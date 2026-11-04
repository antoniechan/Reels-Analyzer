import os
import re
import glob
import json
import time
import tempfile
import argparse
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.models as models
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import DEFAULT_REELS_DIR

# 忽略 librosa 使用 audioread 時的提示警告
warnings.filterwarnings("ignore")

# 載入環境變數與 Gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key


# ==============================================================================
# 0. 萬能音訊載入器 (解決假 MP3 / AAC / MP4 容器導致 mpg123 崩潰問題)
# ==============================================================================

def robust_load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    依序嘗試使用 PyAV (FFmpeg 核心)、librosa、以及同名的 .mp4 視訊檔提取音軌。
    完全避開 soundfile / libmpg123 對非標準 MP3 檔頭的崩潰問題。
    回傳: 1D float32 Tensor (1, num_samples)，採樣率為 target_sr (16000 Hz)
    """
    # 候選讀取來源：優先讀指定 .mp3，若該檔損毀則自動嘗試同名的 .mp4 影片檔
    base_no_ext = os.path.splitext(audio_path)[0]
    mp4_fallback = base_no_ext + ".mp4"
    candidates = [audio_path]
    if os.path.exists(mp4_fallback) and os.path.getsize(mp4_fallback) > 10240:
        candidates.append(mp4_fallback)

    last_err = None

    for src_path in candidates:
        # 方法 1：使用 PyAV (內建 FFmpeg 解碼器，支援所有 AAC/MP4/MP3 容器)
        try:
            import av
            container = av.open(src_path)
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is not None:
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
                chunks = []
                for frame in container.decode(audio_stream):
                    frame.pts = None
                    resampled_frames = resampler.resample(frame)
                    for rf in resampled_frames:
                        chunks.append(rf.to_ndarray())
                container.close()
                if chunks:
                    wav_np = np.concatenate(chunks, axis=-1).astype(np.float32)
                    if wav_np.ndim == 1:
                        wav_np = wav_np[np.newaxis, :]
                    elif wav_np.shape[0] > 1:
                        wav_np = np.mean(wav_np, axis=0, keepdims=True)
                    return torch.from_numpy(wav_np)
        except Exception as e:
            last_err = e

        # 方法 2：使用 librosa (自動調用系統解碼器)
        try:
            import librosa
            wav_np, _ = librosa.load(src_path, sr=target_sr, mono=True)
            if len(wav_np) > 0:
                return torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"無法解碼音訊檔案 (已嘗試 PyAV 與 librosa): {last_err}")


def get_valid_audio_files(root_dir: str) -> list:
    """
    掃描目錄下所有真實有效、非空殼的音訊檔案：
    1. 必須是真實存在的檔案 (os.path.isfile)
    2. 檔案大小必須 > 10 KB (過濾掉爬蟲失敗留下的 0 KB 幽靈檔案)
    3. 排除隱藏檔或暫存檔
    """
    valid_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.lower().endswith(".mp3") or fname.startswith("."):
                continue
            full_path = os.path.join(dirpath, fname)
            if not os.path.isfile(full_path):
                continue

            file_size = os.path.getsize(full_path)
            # 若 .mp3 小於 10KB，檢查是否有對應的正常 .mp4 可供救援
            mp4_companion = os.path.splitext(full_path)[0] + ".mp4"
            if file_size < 10240:
                if os.path.isfile(mp4_companion) and os.path.getsize(mp4_companion) > 10240:
                    valid_files.append(full_path)
                else:
                    print(f"  [略過無效空殼檔] {os.path.relpath(full_path, root_dir)} (大小僅 {file_size} bytes)")
                continue

            valid_files.append(full_path)

    return sorted(valid_files)


# ==============================================================================
# 1. Gemini 3.5 Transcribe 處理器
# ==============================================================================

class GeminiTranscribeProcessor:

    def __init__(self, api_key=None, min_request_interval: float = 5.0):
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY")
        )
        # 每次 API 請求之間的最小安全間隔秒數，避免瞬間塞爆 10,000 TPM
        self.min_request_interval = min_request_interval
        self.last_request_time = 0.0

    @staticmethod
    def _parse_offset_to_seconds(offset_val) -> float:
        if offset_val is None:
            return 0.0
        if isinstance(offset_val, (int, float)):
            return float(offset_val)
        if hasattr(offset_val, "total_seconds"):
            return float(offset_val.total_seconds())
        if isinstance(offset_val, str):
            clean_str = offset_val.strip().rstrip("s")
            try:
                return float(clean_str)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _extract_retry_delay(error_msg: str, attempt: int) -> float:
        """
        從 429 錯誤訊息中自動解析 Google 要求的等待秒數 (例如 'Please retry in 30.37s' 或 'retryDelay': '30s')
        並加上 5 秒安全緩衝。若未找到則採用 35 秒基準遞增。
        """
        # 匹配 "Please retry in 30.3775s"
        match_float = re.search(r"retry in (\d+(?:\.\d+)?)s", error_msg, re.IGNORECASE)
        if match_float:
            return float(match_float.group(1)) + 5.0

        # 匹配 "'retryDelay': '30s'"
        match_delay = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", error_msg, re.IGNORECASE)
        if match_delay:
            return float(match_delay.group(1)) + 5.0

        # 若無明確秒數，對 429 錯誤預設等待 35 秒以上
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return 35.0 + (attempt - 1) * 15.0

        # 一般網路波動錯誤
        return 5.0 * attempt

    def _wait_for_pacing(self):
        """主動節流：確保兩次 API 請求之間至少相隔 min_request_interval 秒"""
        elapsed = time.time() - self.last_request_time
        if self.last_request_time > 0 and elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)

    def transcribe_from_clean_waveform(
        self,
        waveform_16k: torch.Tensor,
        language_codes: list = None,
        enable_diarization: bool = True,
        max_retries: int = 8,
    ):
        """
        將已解碼的乾淨 16kHz 波形存為標準暫存 .wav 後上傳至 Gemini 3.5 Transcribe。
        內建 429 Quota 自動等待與重試補償機制。
        """
        temp_wav_path = None
        uploaded_file = None

        try:
            # 1. 建立標準 PCM 16-bit 暫存 WAV 檔
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            wav_np = waveform_16k.squeeze(0).cpu().numpy()
            sf.write(temp_wav_path, wav_np, 16000, subtype="PCM_16")

            # 2. 主動節流等待
            self._wait_for_pacing()

            # 3. 上傳檔案至 File API (只需上傳一次，重試 generate_content 時可重複使用同一個 uploaded_file)
            uploaded_file = self.client.files.upload(file=temp_wav_path)

            tx_config = types.AudioTranscriptionConfig(
                language_codes=language_codes if language_codes is not None else [],
                word_timestamp=True,
                diarization=enable_diarization,
                mode="VERBATIM",
            )

            # 4. 帶有 429 智慧冷卻補償的請求迴圈
            for attempt in range(1, max_retries + 1):
                try:
                    self.last_request_time = time.time()
                    response = self.client.models.generate_content(
                        model="gemini-3.5-transcribe",
                        contents=[uploaded_file],
                        config=types.GenerateContentConfig(
                            audio_transcription_config=tx_config
                        ),
                    )
                    return response

                except Exception as e:
                    err_str = str(e)
                    wait_sec = self._extract_retry_delay(err_str, attempt)

                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        print(
                            f"  [配額限制 429] 觸發每分鐘 Token 上限 (第 {attempt}/{max_retries} 次)，"
                            f"自動冷卻等待 {wait_sec:.1f} 秒後繼續..."
                        )
                    else:
                        print(
                            f"  [警告] Gemini API 呼叫失敗 (第 {attempt}/{max_retries} 次): {err_str[:120]}... "
                            f"等待 {wait_sec:.1f} 秒後重試"
                        )

                    if attempt < max_retries:
                        time.sleep(wait_sec)
                    else:
                        raise e

        finally:
            # 確保雲端與本地暫存檔在完成或放棄後才被清除
            if uploaded_file is not None:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    def extract_transcription_data(self, response):
        transcribed_chunks = []
        words_list = []

        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            for part in getattr(content, "parts", []) or []:
                transcription = getattr(part, "audio_transcription", None)
                if transcription:
                    speaker = getattr(transcription, "speaker_label", "") or ""
                    seg_text = getattr(transcription, "text", "") or ""
                    if seg_text:
                        transcribed_chunks.append(seg_text)

                    for word_info in getattr(transcription, "words", []) or []:
                        word_str = getattr(word_info, "word", "") or ""
                        start_raw = getattr(word_info, "start_offset", 0.0)
                        end_raw = getattr(word_info, "end_offset", 0.0)

                        start_sec = self._parse_offset_to_seconds(start_raw)
                        end_sec = self._parse_offset_to_seconds(end_raw)

                        words_list.append({
                            "word": word_str,
                            "speaker": speaker,
                            "start_offset": round(start_sec, 3),
                            "end_offset": round(end_sec, 3),
                        })
                elif getattr(part, "text", None):
                    transcribed_chunks.append(part.text)

        full_text = " ".join(transcribed_chunks).strip()
        if not full_text and words_list:
            full_text = " ".join(w["word"] for w in words_list)

        return full_text, words_list

    def build_second_level_semantic_vector(
        self, duration_sec: int, transcribed_text: str, words_info: list, emb_dim: int = 64
    ):
        semantic_curve = np.zeros((duration_sec, emb_dim), dtype=np.float32)
        words_per_second = np.zeros(duration_sec, dtype=np.float32)

        if words_info:
            for idx, item in enumerate(words_info):
                w_start = item["start_offset"]
                w_end = max(w_start + 0.05, item["end_offset"])
                word_len = max(1, len(item["word"].strip()))
                total_w_dur = max(0.05, w_end - w_start)

                sec_start = int(max(0, np.floor(w_start)))
                sec_end = int(min(duration_sec - 1, np.floor(w_end)))

                for s in range(sec_start, sec_end + 1):
                    overlap = max(0.0, min(w_end, s + 1.0) - max(w_start, float(s)))
                    if overlap > 0:
                        words_per_second[s] += (overlap / total_w_dur) * word_len
                        freq_basis = np.linspace(0.5, 3.0, emb_dim)
                        token_encoding = np.sin(freq_basis * (idx % 7 + 1)) + np.cos(
                            freq_basis * word_len
                        )
                        semantic_curve[s, :] += token_encoding * overlap * word_len
        else:
            words = transcribed_text.split() if transcribed_text else []
            density = max(1, len(words)) / max(1, duration_sec)
            for s in range(duration_sec):
                words_per_second[s] = density
                semantic_curve[s, :] = np.sin(np.linspace(s, s + 1, emb_dim)) * float(
                    density * 0.1
                )

        tensor_feat = torch.tensor(
            semantic_curve.T, dtype=torch.float32
        ).unsqueeze(0)
        return tensor_feat, words_per_second


# ==============================================================================
# 2. 預訓練 ResNet-18 聲學流與雙通道能量門控模型
# ==============================================================================

class PretrainedResNetAcousticStream(nn.Module):

    def __init__(self, out_dim=256, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        pretrained_conv1_weight = resnet.conv1.weight.data
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = torch.mean(pretrained_conv1_weight, dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        self.adaptive_freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, mel_spec):
        mean = mel_spec.mean(dim=(-2, -1), keepdim=True).detach()
        std = mel_spec.std(dim=(-2, -1), keepdim=True).detach() + 1e-6
        x = (mel_spec - mean) / std

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.adaptive_freq_pool(x).squeeze(2)
        return x


class GeminiAcousticDualStreamModel(nn.Module):

    def __init__(self, semantic_dim=64, acoustic_dim=256):
        super().__init__()
        self.acoustic_stream = PretrainedResNetAcousticStream(
            out_dim=acoustic_dim, freeze_backbone=True
        )

    def forward(self, semantic_feat, mel_spec, target_seconds):
        acoust_raw = self.acoustic_stream(mel_spec)
        acoust_aligned = F.interpolate(
            acoust_raw, size=target_seconds, mode="linear", align_corners=False
        )
        acoust_texture = torch.mean(torch.abs(acoust_aligned), dim=1)

        core_mel = mel_spec[:, :, 10:90, :]
        frame_energy = torch.mean(core_mel, dim=2)
        sec_energy = F.interpolate(
            frame_energy, size=target_seconds, mode="linear", align_corners=False
        ).squeeze(1)

        e_min, e_max = sec_energy.min().detach(), sec_energy.max().detach()
        energy_gate = (sec_energy - e_min) / (e_max - e_min + 1e-6)

        t_min, t_max = acoust_texture.min().detach(), acoust_texture.max().detach()
        texture_norm = (acoust_texture - t_min) / (t_max - t_min + 1e-6)

        acoust_score = texture_norm * energy_gate

        sem_aligned = F.interpolate(
            semantic_feat, size=target_seconds, mode="linear", align_corners=False
        )
        sem_energy = torch.norm(sem_aligned, p=2, dim=1)
        s_max = sem_energy.max().detach() + 1e-6
        sem_norm = sem_energy / s_max

        per_second_scores = 0.5 * acoust_score + 0.5 * sem_norm
        video_level_score = torch.mean(per_second_scores, dim=-1, keepdim=True)

        return (
            per_second_scores,
            video_level_score,
            acoust_score.detach(),
            sem_norm.detach(),
        )


# ==============================================================================
# 3. 本地音訊 Log-Mel 前處理與單檔分析函數
# ==============================================================================

def prepare_local_log_mel(audio_path, sample_rate=16000):
    waveform = robust_load_audio(audio_path, target_sr=sample_rate)

    duration_sec = int(waveform.shape[-1] / sample_rate)
    if duration_sec < 1:
        duration_sec = 1

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=128,
    )
    log_mel = torch.log(mel_transform(waveform) + 1e-6).unsqueeze(0)

    return waveform, log_mel, duration_sec


def analyze_single_audio(
    audio_path: str,
    model: GeminiAcousticDualStreamModel,
    gemini_proc: GeminiTranscribeProcessor,
    device: str = "cpu",
) -> dict:
    # A. 使用萬能解碼器載入音訊並計算 Log-Mel
    waveform_16k, log_mel_spec, duration_sec = prepare_local_log_mel(audio_path)
    log_mel_spec = log_mel_spec.to(device)
    log_mel_spec.requires_grad = True

    # B. 將乾淨波形轉為標準暫存 WAV 上傳 Gemini 3.5 Transcribe
    transcribe_res = gemini_proc.transcribe_from_clean_waveform(waveform_16k)
    transcribed_text, words_info = gemini_proc.extract_transcription_data(transcribe_res)
    semantic_feat, words_per_sec = gemini_proc.build_second_level_semantic_vector(
        duration_sec, transcribed_text, words_info
    )
    semantic_feat = semantic_feat.to(device)

    # C. 前向推論與反向傳播求 Saliency Map
    model.eval()
    per_sec_scores, video_score, acoust_scores, sem_scores = model(
        semantic_feat, log_mel_spec, duration_sec
    )
    model.zero_grad()
    video_score.backward()

    # D. 整理各項數據指標
    saliency_2d = torch.abs(log_mel_spec.grad).squeeze().cpu().numpy()
    scores_np = np.atleast_1d(per_sec_scores.detach().squeeze().cpu().numpy())
    acoust_np = np.atleast_1d(acoust_scores.squeeze().cpu().numpy())
    sem_np = np.atleast_1d(sem_scores.squeeze().cpu().numpy())

    saliency_temporal_raw = torch.tensor(
        np.mean(saliency_2d, axis=0), dtype=torch.float32
    ).view(1, 1, -1)
    saliency_per_sec = (
        F.interpolate(
            saliency_temporal_raw, size=duration_sec, mode="linear", align_corners=False
        )
        .squeeze()
        .numpy()
    )
    saliency_per_sec = np.atleast_1d(saliency_per_sec)

    hook_window = min(5, duration_sec)
    hook_score = float(np.mean(scores_np[:hook_window]))
    top_k = min(3, duration_sec)
    top_peak_seconds = [int(i + 1) for i in np.argsort(scores_np)[-top_k:][::-1]]

    second_by_second_details = []
    for s in range(duration_sec):
        second_by_second_details.append({
            "second": int(s + 1),
            "engagement_score": round(float(scores_np[s]), 6),
            "acoustic_score": round(float(acoust_np[s]), 6),
            "semantic_score": round(float(sem_np[s]), 6),
            "saliency_grad_mean": round(float(saliency_per_sec[s]), 8),
            "char_density": round(float(words_per_sec[s]), 3),
        })

    account_folder = os.path.basename(os.path.dirname(audio_path))
    file_name = os.path.basename(audio_path)
    video_id = os.path.splitext(file_name)[0]

    return {
        "video_id": video_id,
        "account_id": account_folder,
        "audio_path": os.path.abspath(audio_path),
        "duration_seconds": int(duration_sec),
        "summary_metrics": {
            "mean_engagement_score": round(float(np.mean(scores_np)), 6),
            "std_engagement_score": round(float(np.std(scores_np)), 6),
            "max_engagement_score": round(float(np.max(scores_np)), 6),
            "min_engagement_score": round(float(np.min(scores_np)), 6),
            "hook_first_5s_score": round(hook_score, 6),
            "top_peak_seconds": top_peak_seconds,
            "mean_saliency_gradient": round(float(np.mean(saliency_2d)), 8),
            "max_saliency_gradient": round(float(np.max(saliency_2d)), 8),
        },
        "per_second_scores": [round(float(x), 6) for x in scores_np],
        "second_by_second_metrics": second_by_second_details,
        "transcription": {
            "full_text": transcribed_text,
            "total_characters": len(transcribed_text),
            "word_count": len(words_info),
            "words": words_info,
        },
    }


# ==============================================================================
# 4. 批次遍歷 D:\reels 並輸出 JSON
# ==============================================================================

def run_batch_processing(
    root_dir: str = DEFAULT_REELS_DIR,
    overwrite: bool = False,
    max_files: int = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("【Dual-Stream 音訊吸睛度與顯著性批次分析系統 (PyAV 強化解碼版)】")
    print(f"  - 運算設備 (Device):      {device}")
    print(f"  - 掃描根目錄 (Root Dir):  {root_dir}")
    print(f"  - 覆寫既有 JSON:          {overwrite}")
    print("=" * 70)

    if not os.path.exists(root_dir):
        print(f"錯誤：找不到指定的目錄 {root_dir}，請確認路徑是否正確。")
        return

    mp3_files = get_valid_audio_files(root_dir)

    if not mp3_files:
        print(f"在 {root_dir} 下未找到任何有效的 .mp3 檔案。")
        return

    if max_files is not None:
        mp3_files = mp3_files[:max_files]

    print(f"共驗證通過 {len(mp3_files)} 個有效音訊檔，正在載入模型...\n")

    model = GeminiAcousticDualStreamModel().to(device)
    gemini_proc = GeminiTranscribeProcessor()

    processed_count = 0
    skipped_count = 0
    failed_files = []
    batch_summary = []

    for idx, mp3_path in enumerate(mp3_files, start=1):
        base_path = os.path.splitext(mp3_path)[0]
        output_json_path = f"{base_path}_audio_saliency.json"
        rel_path = os.path.relpath(mp3_path, root_dir)

        if os.path.exists(output_json_path) and not overwrite:
            print(f"[{idx}/{len(mp3_files)}] 略過 (已存在): {rel_path}")
            skipped_count += 1
            try:
                with open(output_json_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                batch_summary.append({
                    "video_id": existing_data.get("video_id"),
                    "account_id": existing_data.get("account_id"),
                    "duration_seconds": existing_data.get("duration_seconds"),
                    **existing_data.get("summary_metrics", {}),
                })
            except Exception:
                pass
            continue

        print(f"[{idx}/{len(mp3_files)}] 正在分析: {rel_path} ...")
        start_t = time.time()

        try:
            result_data = analyze_single_audio(
                audio_path=mp3_path,
                model=model,
                gemini_proc=gemini_proc,
                device=device,
            )

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_t
            mean_sc = result_data["summary_metrics"]["mean_engagement_score"]
            hook_sc = result_data["summary_metrics"]["hook_first_5s_score"]
            peaks = result_data["summary_metrics"]["top_peak_seconds"]
            word_cnt = result_data["transcription"]["word_count"]

            print(
                f"  -> 完成 ({elapsed:.1f}s) | 詞數: {word_cnt} | 平均分: {mean_sc:.4f} | "
                f"前5秒 Hook: {hook_sc:.4f} | 波峰秒數: {peaks}"
            )
            processed_count += 1

            batch_summary.append({
                "video_id": result_data["video_id"],
                "account_id": result_data["account_id"],
                "duration_seconds": result_data["duration_seconds"],
                **result_data["summary_metrics"],
            })

        except Exception as e:
            print(f"  -> [失敗] 處理 {rel_path} 時發生錯誤: {e}")
            failed_files.append({"file": rel_path, "error": str(e)})

    summary_file_path = os.path.join(root_dir, "all_reels_audio_saliency_summary.json")
    summary_payload = {
        "total_valid_mp3": len(mp3_files),
        "newly_processed": processed_count,
        "skipped_existing": skipped_count,
        "failed_count": len(failed_files),
        "failed_files": failed_files,
        "videos": batch_summary,
    }

    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("【批次處理完成！】")
    print(f"  - 成功新運算檔案數: {processed_count}")
    print(f"  - 略過已存在檔案數: {skipped_count}")
    print(f"  - 失敗檔案數:       {len(failed_files)}")
    print(f"  - 總體摘要 JSON:    {os.path.abspath(summary_file_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批次對 D:\\reels 內所有 mp3 執行 Dual-Stream 音訊吸睛度分析並輸出 JSON"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_REELS_DIR,
        help=f"Reels 根目錄路徑 (預設: {DEFAULT_REELS_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若加上此參數，將強制重新計算並覆寫已存在的 *_audio_saliency.json",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="限制最多處理的檔案數量 (例如 --max_files 5)",
    )
    args = parser.parse_args()

    run_batch_processing(
        root_dir=args.dir,
        overwrite=args.overwrite,
        max_files=args.max_files,
    )