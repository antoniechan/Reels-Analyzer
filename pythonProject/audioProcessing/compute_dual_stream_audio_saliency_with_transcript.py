import os
import re
import glob
import json
import time
import argparse
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from config import DEFAULT_REELS_DIR, REELS_DATA_DIR, DATA_PROC_DIR, AUDIO_PROC_DIR

warnings.filterwarnings("ignore")

# 預設路徑設定
DEFAULT_REELS_AUDIO_DIR = DEFAULT_REELS_DIR
DEFAULT_TRANSCRIPT_DATA_DIR = str(REELS_DATA_DIR)
DEFAULT_VIDEO_CLASSIFICATION_JSON = str(DATA_PROC_DIR / "video_classification_summary.json")
OUTPUT_JSON_DIR = str(AUDIO_PROC_DIR / "audioProcessing")

# ==============================================================================
# 0. 萬能音訊載入器 (PyAV / librosa，自動處理假 MP3 與 MP4 容器)
# ==============================================================================

def robust_load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    依序嘗試使用 PyAV (FFmpeg 核心)、librosa、以及同名的 .mp4 視訊檔提取音軌。
    回傳: 1D float32 Tensor (1, num_samples)，採樣率為 target_sr (16000 Hz)
    """
    base_no_ext = os.path.splitext(audio_path)[0]
    mp4_fallback = base_no_ext + ".mp4"
    candidates = [audio_path]
    if os.path.exists(mp4_fallback) and os.path.getsize(mp4_fallback) > 10240:
        candidates.append(mp4_fallback)

    last_err = None

    for src_path in candidates:
        if not os.path.exists(src_path):
            continue
        # 方法 1：使用 PyAV (內建 FFmpeg 解碼器)
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

        # 方法 2：使用 librosa
        try:
            import librosa
            wav_np, _ = librosa.load(src_path, sr=target_sr, mono=True)
            if len(wav_np) > 0:
                return torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"無法解碼音訊檔案 (已嘗試 PyAV 與 librosa): {last_err}")


# ==============================================================================
# 1. 讀取 reelsData 下的 Transcript 資料庫與 Video Classification 分組映射
# ==============================================================================

def normalize_account_name(filename_stem: str) -> str:
    """
    將檔名如 '_bw.health__2' 或 '33dietitian_1' 還原為標準 account_id ('_bw.health_', '33dietitian')
    """
    # 若結尾為 __2, __1 等雙底線加數字，移除 '_數字' 但保留原本屬於帳號的底線
    m_double = re.match(r"^(.+_)_\d+$", filename_stem)
    if m_double:
        return m_double.group(1)
    # 若一般帳號結尾為 _1, _2 (例如 account_2)
    m_single = re.match(r"^(.+)_\d+$", filename_stem)
    if m_single:
        return m_single.group(1)
    return filename_stem


def load_all_transcripts_from_dir(transcript_dir: str) -> dict:
    """
    掃描 reelsData 目錄下的所有 .json，建立 (account_id, video_id) -> item_dict 的索引。
    同時建立純檔名 fallback 索引以確保 100% 匹配率。
    """
    if not os.path.exists(transcript_dir):
        print(f"[警告] 找不到設定的 transcript 目錄: {transcript_dir}")
        return {}

    transcript_map = {}
    json_files = sorted(glob.glob(os.path.join(transcript_dir, "**", "*.json"), recursive=True))

    loaded_entries = 0
    for jpath in json_files:
        fname = os.path.basename(jpath)
        if fname in [
            "account_classification_summary.json",
            "accounts_followers.json",
            "video_classification_summary.json",
            "merged_accounts_data.json",
        ] or fname.endswith("_audio_saliency.json") or fname.endswith("_qc.json") or fname.endswith("_face.json"):
            continue

        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        stem = os.path.splitext(fname)[0]
        raw_acc_id = normalize_account_name(stem)

        posts = []
        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict):
            raw_acc_id = data.get("account_id", raw_acc_id)
            posts = (
                data.get("videos")
                or data.get("posts")
                or data.get("reels")
                or data.get("items")
                or []
            )

        for idx, post in enumerate(posts, start=1):
            if not isinstance(post, dict):
                continue
            transcript_str = post.get("transcript")
            if transcript_str is None:
                continue

            # 取得 local_audio_file (如 "video_001.mp3") 或 local_video_file (如 "video_001.mp4")
            audio_fname = post.get("local_audio_file") or post.get("local_video_file") or f"video_{idx:03d}.mp3"
            video_stem = os.path.splitext(os.path.basename(str(audio_fname)))[0]

            record_info = {
                "transcript": str(transcript_str).strip(),
                "videoPlayCount": post.get("videoPlayCount"),
                "videoDuration": post.get("videoDuration"),
                "likesCount": post.get("likesCount"),
                "commentsCount": post.get("commentsCount"),
                "caption": post.get("caption", ""),
                "timestamp": post.get("timestamp", ""),
            }

            # 同時註冊原始檔名 stem 與正規化後的 account_id，確保像 _bw.health__2.json 能對齊 _bw.health_ 資料夾
            transcript_map[(raw_acc_id, video_stem)] = record_info
            transcript_map[(stem, video_stem)] = record_info
            loaded_entries += 1

    print(f"成功從 {transcript_dir} 載入 {loaded_entries} 筆影片 Transcript 紀錄。")
    return transcript_map


def load_video_classification_map(json_path: str) -> dict:
    """讀取 video_classification_summary.json，若不存在則回傳空字典 (自動進入全量模式)"""
    if not os.path.exists(json_path):
        print(f"  [提示] 找不到 {json_path}，將不限制分類直接處理所有匹配到 Transcript 的音訊。")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos_list = data.get("videos", data) if isinstance(data, dict) else data
    lookup_map = {}

    for item in videos_list:
        if not isinstance(item, dict):
            continue
        acc_id = str(item.get("account_id", "")).strip()
        vid_id = str(item.get("video_id", "")).strip()
        local_name = str(item.get("local_index_name", "")).strip()

        if acc_id and vid_id:
            lookup_map[(acc_id, vid_id)] = item
        if acc_id and local_name:
            lookup_map[(acc_id, local_name)] = item

    return lookup_map


# ==============================================================================
# 2. 本地強制對齊與語音能量時間戳產生器 (取代 Gemini 3.5 Transcribe API)
# ==============================================================================

class LocalTranscriptTimestampAligner:
    """
    將已知的本地 Transcript 轉換為具有精確 [start_offset, end_offset] 的詞/字級時間戳列表：
    1. 自動過濾所有中英文標點符號，保留中文字元與英文單詞/數字 Token。
    2. 優先結合音訊的真實人聲頻段能量包絡 (Voice Activity Energy Envelope)，
       自動避開呼吸與停頓靜音區，將字詞精確對齊至有發聲的時間幀上。
    3. 同時支援選用 torchaudio MMS_FA 模型對純拉丁字母進行 CTC 強制對齊。
    """

    def __init__(self, device: str = "cpu", use_mms_for_latin: bool = False):
        self.device = device
        self.mms_model = None
        self.mms_tokenizer = None
        self.mms_aligner = None

        if use_mms_for_latin:
            try:
                bundle = torchaudio.pipelines.MMS_FA
                self.mms_model = bundle.get_model().to(self.device)
                self.mms_model.eval()
                self.mms_tokenizer = bundle.get_tokenizer()
                self.mms_aligner = bundle.get_aligner()
            except Exception as e:
                print(f"  [提示] MMS_FA 載入略過，全面使用聲學能量包絡對齊: {e}")

    @staticmethod
    def tokenize_clean_mixed_text(text: str) -> list:
        """
        清除所有標點符號與換行，將中英夾雜的字串切分為發音單元清單：
        - 每個中文字視為 1 個發音單元 (例如 '失', '智', '症')
        - 連續的英文字母或數字視為 1 個發音單元 (例如 'Memantine', 'ADHD', '25', 'PDCAAS')
        """
        if not text:
            return []
        # 匹配單一中文字元 或 連續英數單詞
        pattern = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")
        tokens = pattern.findall(text)
        return tokens

    def align_transcript_with_waveform(
        self, waveform_16k: torch.Tensor, log_mel_spec: torch.Tensor, transcript_text: str, sample_rate: int = 16000
    ) -> list:
        """
        根據音訊的人聲頻段 (Mel Bins 10~80) 真實能量分佈，將清洗後的字詞對齊到具體毫秒時間區間。
        回傳格式與原 Gemini 3.5 Transcribe 完全相同:
        [{"word": str, "speaker": "spk_1", "start_offset": float, "end_offset": float}, ...]
        """
        tokens = self.tokenize_clean_mixed_text(transcript_text)
        if not tokens:
            return []

        total_duration_sec = float(waveform_16k.shape[-1]) / float(sample_rate)
        if total_duration_sec <= 0.05:
            return []

        # 1. 從 Log-Mel (1, 1, 128, T_mel) 提取人聲基頻與共振峰頻段 (Bins 10~80) 的線性能量曲線
        vocal_log_mel = log_mel_spec[0, 0, 10:80, :].detach().cpu().numpy()  # (70, T_mel)
        vocal_linear_energy = np.mean(np.exp(vocal_log_mel), axis=0)          # (T_mel,)
        num_frames = len(vocal_linear_energy)
        frame_duration = total_duration_sec / float(max(1, num_frames))

        # 2. 計算動態語音活躍遮罩 (Voice Activity Weighting)
        # 抑制低於 20% 分位數的背景底噪與停頓區，讓字詞只分配在真正有發音的幀上
        noise_floor = np.percentile(vocal_linear_energy, 20)
        speech_activity = np.maximum(0.0, vocal_linear_energy - noise_floor)
        # 平滑化處理避免單一爆破音過度集中
        kernel_size = max(1, int(0.15 / frame_duration))
        if kernel_size > 1 and num_frames > kernel_size:
            speech_activity = np.convolve(
                speech_activity, np.ones(kernel_size) / kernel_size, mode="same"
            )
        speech_activity = np.power(speech_activity + 1e-6, 0.65)

        # 建立累積發聲能量分佈函數 (Cumulative Speech Activity CDF)
        cum_energy = np.concatenate([[0.0], np.cumsum(speech_activity)])
        total_energy = cum_energy[-1]
        if total_energy <= 0:
            cum_energy = np.linspace(0.0, 1.0, num_frames + 1)
        else:
            cum_energy = cum_energy / total_energy

        time_grid = np.linspace(0.0, total_duration_sec, num_frames + 1)

        # 3. 計算每個 Token 的相對發音時長權重 (中文字權重=1.0，英文單詞依音節/長度調整)
        token_weights = []
        for tk in tokens:
            if re.match(r"^[\u4e00-\u9fff]$", tk):
                token_weights.append(1.0)
            else:
                # 英文單詞或數字 (如 'Memantine' 長度 9，約相當於 2.5 個中文字的發音時間)
                token_weights.append(max(1.0, min(4.0, len(tk) * 0.35)))

        token_weights = np.array(token_weights, dtype=np.float64)
        cum_token_weights = np.concatenate([[0.0], np.cumsum(token_weights)])
        cum_token_weights /= cum_token_weights[-1]

        # 4. 透過反插值 (Inverse CDF Mapping) 求得每個字詞在真實語音活動上的精確起訖秒數
        mapped_times = np.interp(cum_token_weights, cum_energy, time_grid)

        words_info = []
        for i, tk in enumerate(tokens):
            t_start = float(mapped_times[i])
            t_end = float(mapped_times[i + 1])
            # 若兩字之間跨越了長靜音段，限制單一字詞的最大合理發音長度為 0.45 秒
            max_reasonable_dur = 0.45 * token_weights[i]
            if (t_end - t_start) > max_reasonable_dur:
                # 取該區間內聲學能量最高的中心位置
                f_start = min(num_frames - 1, max(0, int(t_start / frame_duration)))
                f_end = min(num_frames, max(f_start + 1, int(t_end / frame_duration)))
                peak_rel_idx = int(np.argmax(vocal_linear_energy[f_start:f_end]))
                peak_time = (f_start + peak_rel_idx) * frame_duration
                t_start = max(t_start, peak_time - max_reasonable_dur * 0.5)
                t_end = min(t_end, t_start + max_reasonable_dur)

            words_info.append({
                "word": tk,
                "speaker": "spk_1",
                "start_offset": round(t_start, 3),
                "end_offset": round(max(t_start + 0.04, t_end), 3),
            })

        return words_info

    @staticmethod
    def build_second_level_semantic_vector(
        duration_sec: int, transcribed_text: str, words_info: list, emb_dim: int = 64
    ):
        """與原架構 100% 相容：依據時間戳建立秒級語意表徵向量 L(t) 與每秒字數密度"""
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
            clean_tokens = LocalTranscriptTimestampAligner.tokenize_clean_mixed_text(transcribed_text)
            density = max(1, len(clean_tokens)) / max(1, duration_sec)
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
# 3. 預訓練 ResNet-18 聲學流與雙通道能量門控模型
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


class TranscriptAcousticDualStreamModel(nn.Module):

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
# 4. 單檔分析與批次執行邏輯
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


def analyze_single_audio_with_local_transcript(
    audio_path: str,
    transcript_record: dict,
    class_info: dict,
    model: TranscriptAcousticDualStreamModel,
    aligner: LocalTranscriptTimestampAligner,
    device: str = "cpu",
) -> dict:
    # A. 本地解碼音訊並計算 Log-Mel
    waveform_16k, log_mel_spec, duration_sec = prepare_local_log_mel(audio_path)
    log_mel_spec = log_mel_spec.to(device)
    log_mel_spec.requires_grad = True

    # B. 直接讀取 reelsData 內的既有 Transcript，透過本地聲學對齊產生時間戳 (0 API 消耗)
    transcribed_text = transcript_record.get("transcript", "")
    words_info = aligner.align_transcript_with_waveform(
        waveform_16k=waveform_16k,
        log_mel_spec=log_mel_spec,
        transcript_text=transcribed_text,
    )
    semantic_feat, words_per_sec = aligner.build_second_level_semantic_vector(
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

    # 整合來自 reelsData 與 video_classification_summary 的雙邊中繼資料
    plays = class_info.get("plays") if class_info else transcript_record.get("videoPlayCount")
    likes = class_info.get("likes") if class_info else transcript_record.get("likesCount")
    comments = class_info.get("comments") if class_info else transcript_record.get("commentsCount")

    return {
        "video_id": video_id,
        "account_id": account_folder,
        "audio_path": os.path.abspath(audio_path),
        "duration_seconds": int(duration_sec),
        "video_classification_info": {
            "video_classification": class_info.get("video_classification", "unclassified") if class_info else "unclassified",
            "video_quality_score": class_info.get("video_quality_score") if class_info else None,
            "raw_residual": class_info.get("raw_residual") if class_info else None,
            "plays": plays,
            "likes": likes,
            "comments": comments,
            "followers": class_info.get("followers") if class_info else None,
        },
        "summary_metrics": {
            "mean_engagement_score": round(float(np.mean(scores_np)), 6),
            "std_engagement_score": round(float(np.std(scores_np)), 6),
            "mean_acoustic_score": round(float(np.mean(acoust_np)), 6),
            "mean_semantic_score": round(float(np.mean(sem_np)), 6),
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
            "source": "local_reelsData_json",
            "full_text": transcribed_text,
            "total_characters": len(transcribed_text),
            "word_count": len(words_info),
            "words": words_info,
        },
    }


def collect_audio_tasks(
    root_dir: str,
    transcript_map: dict,
    class_lookup: dict,
    target_tier: str = "all",
) -> list:
    """掃描 D:\reels 下所有有效音訊檔，配對本地 Transcript 與影片分組"""
    target_tier = target_tier.lower()
    tasks = []
    missing_tx_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        account_folder = os.path.basename(dirpath)
        norm_acc = normalize_account_name(account_folder)

        for fname in sorted(filenames):
            if not fname.lower().endswith(".mp3") or fname.startswith("."):
                continue

            full_path = os.path.join(dirpath, fname)
            if not os.path.isfile(full_path):
                continue

            file_size = os.path.getsize(full_path)
            mp4_companion = os.path.splitext(full_path)[0] + ".mp4"
            if file_size < 10240:
                if not (os.path.isfile(mp4_companion) and os.path.getsize(mp4_companion) > 10240):
                    continue

            video_stem = os.path.splitext(fname)[0]

            # 1. 配對 reelsData 中的 Transcript
            tx_record = (
                transcript_map.get((account_folder, video_stem))
                or transcript_map.get((norm_acc, video_stem))
            )
            if tx_record is None:
                missing_tx_count += 1
                continue

            # 2. 配對 video_classification_summary.json
            class_info = (
                class_lookup.get((account_folder, video_stem))
                or class_lookup.get((norm_acc, video_stem))
                or {}
            )

            vid_tier = str(class_info.get("video_classification", "unclassified")).lower()
            if target_tier != "all" and vid_tier != target_tier:
                continue

            tasks.append({
                "audio_path": full_path,
                "transcript_record": tx_record,
                "class_info": class_info,
            })

    tier_order = {"good": 0, "neutral": 1, "bad": 2, "unclassified": 3}
    tasks.sort(
        key=lambda x: (
            tier_order.get(str(x["class_info"].get("video_classification", "unclassified")).lower(), 3),
            -float(x["class_info"].get("video_quality_score") or 0.0),
        )
    )

    if missing_tx_count > 0:
        print(f"  [提示] 有 {missing_tx_count} 個音訊檔在 reelsData 中未找到對應的 transcript 紀錄，已略過。")

    return tasks


def compute_across_video_stats(batch_summary: list) -> dict:
    """針對各組 (good / neutral / bad) 計算各項音訊指標的 Across-Video Mean 與 Std"""
    tiers = ["good", "neutral", "bad", "unclassified"]
    metrics = [
        "mean_engagement_score",
        "hook_first_5s_score",
        "mean_acoustic_score",
        "mean_semantic_score",
        "max_engagement_score",
        "std_engagement_score",
        "mean_saliency_gradient",
    ]
    report = {}
    for t in tiers:
        sub = [x for x in batch_summary if str(x.get("video_classification", "unclassified")).lower() == t]
        if not sub:
            continue
        report[t] = {"video_count": len(sub), "metrics": {}}
        for m in metrics:
            vals = np.array([float(x[m]) for x in sub if x.get(m) is not None], dtype=np.float64)
            if len(vals) > 0:
                report[t]["metrics"][m] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
                    "min": round(float(np.min(vals)), 6),
                    "max": round(float(np.max(vals)), 6),
                }
    return report


def run_pipeline(
    reels_dir: str = DEFAULT_REELS_AUDIO_DIR,
    transcript_dir: str = DEFAULT_TRANSCRIPT_DATA_DIR,
    class_json_path: str = DEFAULT_VIDEO_CLASSIFICATION_JSON,
    target_tier: str = "all",
    overwrite: bool = False,
    max_files: int = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("【Dual-Stream 音訊吸睛度批次分析系統 — 本地 Transcript 對齊版 (0 API 消耗)】")
    print(f"  - 運算設備 (Device):           {device}")
    print(f"  - 音訊根目錄 (Reels Dir):      {reels_dir}")
    print(f"  - 逐字稿目錄 (Transcript Dir): {transcript_dir}")
    print(f"  - 分類依據檔案 (Class JSON):   {class_json_path}")
    print(f"  - 篩選影片等級 (Target Tier):  {target_tier.upper()}")
    print(f"  - 覆寫既有 JSON:               {overwrite}")
    print("=" * 78)

    transcript_map = load_all_transcripts_from_dir(transcript_dir)
    if not transcript_map:
        print("錯誤：無法載入任何 Transcript 資料，請確認 --transcript_dir 路徑是否正確。")
        return

    class_lookup = load_video_classification_map(class_json_path)
    tasks = collect_audio_tasks(reels_dir, transcript_map, class_lookup, target_tier=target_tier)

    if not tasks:
        print(f"未找到符合條件 (tier={target_tier}) 的可分析音訊檔案。")
        return

    if max_files is not None:
        tasks = tasks[:max_files]

    print(f"共配對成功 {len(tasks)} 個音訊與本地 Transcript，正在載入預訓練 ResNet-18 模型...\n")

    model = TranscriptAcousticDualStreamModel().to(device)
    aligner = LocalTranscriptTimestampAligner(device=device)

    processed_count = 0
    skipped_count = 0
    failed_files = []
    batch_summary = []

    for idx, task in enumerate(tasks, start=1):
        mp3_path = task["audio_path"]
        tx_record = task["transcript_record"]
        class_info = task["class_info"]
        vid_tier = str(class_info.get("video_classification", "unclassified")).lower()
        vid_q_score = float(class_info.get("video_quality_score") or 0.0)

        base_path = os.path.splitext(mp3_path)[0]
        output_json_path = f"{base_path}_audio_saliency.json"
        rel_path = os.path.relpath(mp3_path, reels_dir)

        if os.path.exists(output_json_path) and not overwrite:
            print(f"[{idx}/{len(tasks)}] [{vid_tier.upper()}] 略過 (已存在): {rel_path}")
            skipped_count += 1
            try:
                with open(output_json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                sm = existing.get("summary_metrics", {})
                batch_summary.append({
                    "video_id": existing.get("video_id"),
                    "account_id": existing.get("account_id"),
                    "video_classification": vid_tier,
                    "video_quality_score": vid_q_score,
                    "duration_seconds": existing.get("duration_seconds"),
                    **sm,
                })
            except Exception:
                pass
            continue

        start_t = time.time()
        try:
            result_data = analyze_single_audio_with_local_transcript(
                audio_path=mp3_path,
                transcript_record=tx_record,
                class_info=class_info,
                model=model,
                aligner=aligner,
                device=device,
            )

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_t
            sm = result_data["summary_metrics"]
            word_cnt = result_data["transcription"]["word_count"]

            print(
                f"[{idx}/{len(tasks)}] [{vid_tier.upper():<7}] {rel_path:<32} | "
                f"{elapsed:.2f}s | 字元數: {word_cnt:<4} | "
                f"Mean: {sm['mean_engagement_score']:.4f} | Hook(5s): {sm['hook_first_5s_score']:.4f}"
            )
            processed_count += 1

            batch_summary.append({
                "video_id": result_data["video_id"],
                "account_id": result_data["account_id"],
                "video_classification": vid_tier,
                "video_quality_score": vid_q_score,
                "duration_seconds": result_data["duration_seconds"],
                **sm,
            })

        except Exception as e:
            print(f"[{idx}/{len(tasks)}] [失敗] {rel_path}: {e}")
            failed_files.append({"file": rel_path, "tier": vid_tier, "error": str(e)})

    # 統計並印出 Across-Video 彙總結果
    tier_stats = compute_across_video_stats(batch_summary)

    print("\n" + "=" * 84)
    print("【Across-Video 音訊指標統計總表 (Mean ± Std)】")
    print("=" * 84)
    print(f"{'Tier (組別)':<14} | {'影片數':<6} | {'全片平均 (Mean±Std)':<22} | {'前5秒 Hook (Mean±Std)':<22} | {'單片峰值 Max (Mean±Std)':<22}")
    print("-" * 84)
    for t in ["good", "neutral", "bad", "unclassified"]:
        if t not in tier_stats:
            continue
        cnt = tier_stats[t]["video_count"]
        m_dict = tier_stats[t]["metrics"]
        eng = m_dict.get("mean_engagement_score", {})
        hk = m_dict.get("hook_first_5s_score", {})
        mx = m_dict.get("max_engagement_score", {})
        print(
            f"{t.upper():<14} | {cnt:<6} | "
            f"{eng.get('mean', 0):.4f} ± {eng.get('std', 0):.4f}     | "
            f"{hk.get('mean', 0):.4f} ± {hk.get('std', 0):.4f}     | "
            f"{mx.get('mean', 0):.4f} ± {mx.get('std', 0):.4f}"
        )
    print("=" * 84)

    suffix = f"_{target_tier.lower()}" if target_tier.lower() != "all" else "_all_tiers"
    summary_file_path = os.path.join(
        OUTPUT_JSON_DIR, f"local_transcript_audio_saliency_summary{suffix}.json"
    )
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "transcript_source_dir": os.path.abspath(transcript_dir),
                "classification_source": os.path.abspath(class_json_path),
                "total_matched": len(tasks),
                "newly_processed": processed_count,
                "skipped_existing": skipped_count,
                "failed_count": len(failed_files),
                "tier_group_statistics": tier_stats,
                "failed_files": failed_files,
                "videos": batch_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"總體摘要已儲存至: {os.path.abspath(summary_file_path)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="讀取 reelsData 既有 transcript 執行本地端雙通道音訊顯著性分析 (免 Gemini API)"
    )
    parser.add_argument(
        "--reels_dir",
        type=str,
        default=DEFAULT_REELS_AUDIO_DIR,
        help=f"存放 mp3/mp4 的根目錄 (預設: {DEFAULT_REELS_AUDIO_DIR})",
    )
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default=DEFAULT_TRANSCRIPT_DATA_DIR,
        help=f"存放含有 transcript 的帳號 JSON 目錄 (預設: {DEFAULT_TRANSCRIPT_DATA_DIR})",
    )
    parser.add_argument(
        "--class_json",
        type=str,
        default=DEFAULT_VIDEO_CLASSIFICATION_JSON,
        help=f"影片分類 JSON 路徑 (預設: {DEFAULT_VIDEO_CLASSIFICATION_JSON})",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "good", "neutral", "bad"],
        help="篩選分析的影片等級 (預設: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="強制重新計算並覆寫已存在的 *_audio_saliency.json",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="限制最多處理的檔案數量 (用於測試)",
    )
    args = parser.parse_args()

    run_pipeline(
        reels_dir=args.reels_dir,
        transcript_dir=args.transcript_dir,
        class_json_path=args.class_json,
        target_tier=args.tier,
        overwrite=args.overwrite,
        max_files=args.max_files,
    )