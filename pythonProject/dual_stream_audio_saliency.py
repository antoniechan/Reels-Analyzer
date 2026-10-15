import asyncio
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models

# 載入環境變數與 Gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
  os.environ["GEMINI_API_KEY"] = api_key


# ==============================================================================
# 1. Gemini 3.5 Transcribe 處理器 (對齊最新 SDK Config 與 Response Fields)
# ==============================================================================


class GeminiTranscribeProcessor:

  def __init__(self, api_key=None):
    self.client = genai.Client(
        api_key=api_key or os.environ.get("GEMINI_API_KEY")
    )

  @staticmethod
  def _parse_offset_to_seconds(offset_val) -> float:
    """將 API 回傳的 start_offset / end_offset (例如 '0.300s'、float 或 timedelta) 統一轉為 float 秒數"""
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

  def transcribe_with_timestamps(
      self,
      audio_path: str,
      language_codes: list = None,
      enable_diarization: bool = True,
  ):
    """使用 gemini-3.5-transcribe 進行同步檔案轉錄，開啟 word_timestamp 與 diarization"""
    print(f"正在上傳音訊至 Gemini 進行轉錄: {os.path.basename(audio_path)}...")
    audio_file = self.client.files.upload(file=audio_path)

    # 根據官方規範配置 AudioTranscriptionConfig
    # 注意：開啟 word_timestamp 或 diarization 時，mode 必須為 VERBATIM (預設)，且不可同時傳入 custom_vocabulary
    tx_config = types.AudioTranscriptionConfig(
        language_codes=language_codes if language_codes is not None else [],
        word_timestamp=True,
        diarization=enable_diarization,
        mode="VERBATIM",
    )

    response = self.client.models.generate_content(
        model="gemini-3.5-transcribe",
        contents=[audio_file],
        config=types.GenerateContentConfig(
            audio_transcription_config=tx_config
        ),
    )
    return response

  async def transcribe_live_stream(
      self, audio_path: str, language_codes: list = None
  ):
    """使用 gemini-3.5-transcribe-live (Live API) 進行 16kHz PCM 即時串流轉錄

    對齊 server_content.interim_input_transcription 與
    server_content.input_transcription
    """
    model = "gemini-3.5-transcribe-live"
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=language_codes if language_codes is not None else [],
        ),
    )

    # 讀取音訊並轉換為 16kHz 單聲道 16-bit PCM bytes
    wav_data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(wav_data)
    if waveform.ndim > 1:
      waveform = torch.mean(waveform, dim=-1)
    if sr != 16000:
      waveform = T.Resample(sr, 16000)(waveform.unsqueeze(0)).squeeze(0)

    pcm_int16 = (waveform.clamp(-1.0, 1.0).numpy() * 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    final_segments = []

    async with self.client.aio.live.connect(
        model=model, config=config
    ) as session:
      print("已建立 Gemini 3.5 Transcribe Live 連線，開始傳送 PCM 音訊串流...")

      async def send_audio():
        # 每次傳送 100ms (1600 frames = 3200 bytes)
        chunk_size = 3200
        for i in range(0, len(pcm_bytes), chunk_size):
          chunk = pcm_bytes[i : i + chunk_size]
          await session.send_realtime_input(
              audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
          )
          await asyncio.sleep(0.02)
        await session.send_realtime_input(audio_stream_end=True)

      send_task = asyncio.create_task(send_audio())

      async for response in session.receive():
        server_content = getattr(response, "server_content", None)
        if not server_content:
          continue

        # 1. 接收即時預測字幕 (interim_input_transcription)
        interim = getattr(server_content, "interim_input_transcription", None)
        if interim and getattr(interim, "text", None):
          print(f"\r[Interim] {interim.text}", end="", flush=True)

        # 2. 接收最終確認字幕 (input_transcription)
        final = getattr(server_content, "input_transcription", None)
        if final and getattr(final, "text", None):
          print(f"\n[Final] {final.text}")
          final_segments.append(final.text)

      await send_task

    return " ".join(final_segments)

  def extract_transcription_data(self, response):
    """官方標準解析方法：從 candidates -> content -> parts -> audio_transcription 提取所有欄位

    回傳: (完整文字, words_list) words_list 結構: [{"word": str, "speaker": str,
    "start_offset": float, "end_offset": float}]
    """
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
                "start_offset": start_sec,
                "end_offset": end_sec,
            })
        elif getattr(part, "text", None):
          transcribed_chunks.append(part.text)

    full_text = " ".join(transcribed_chunks).strip()
    if not full_text and words_list:
      full_text = " ".join(w["word"] for w in words_list)

    return full_text, words_list

  def build_second_level_semantic_vector(
      self, duration_sec: int, response, emb_dim: int = 64
  ):
    """依據 Gemini 3.5 Transcribe 的精確毫秒級 [start_offset, end_offset] 重疊時長

    建立秒級語意表徵向量 L(t) in R^(1 x emb_dim x duration_sec)
    """
    transcribed_text, words_info = self.extract_transcription_data(response)

    print(
        f"Gemini 轉錄完成 | 總字數: {len(transcribed_text)} | 詞級時間戳數量:"
        f" {len(words_info)}"
    )
    if transcribed_text:
      print(f"轉錄內容預覽: {transcribed_text[:80]}...")

    semantic_curve = np.zeros((duration_sec, emb_dim), dtype=np.float32)

    # 方案 A：使用精確的 [start_offset, end_offset] 計算每秒內的發音覆蓋時長與詞彙密度
    if words_info:
      for idx, item in enumerate(words_info):
        w_start = item["start_offset"]
        w_end = max(w_start + 0.05, item["end_offset"])
        word_len = max(1, len(item["word"].strip()))

        # 找出該字詞橫跨的秒數區間 [t, t+1]
        sec_start = int(max(0, np.floor(w_start)))
        sec_end = int(min(duration_sec - 1, np.floor(w_end)))

        for s in range(sec_start, sec_end + 1):
          # 計算該詞在第 s 秒區間 [s, s+1] 內的實際重疊時間 (Overlap Duration)
          overlap = max(0.0, min(w_end, s + 1.0) - max(w_start, float(s)))
          if overlap > 0:
            freq_basis = np.linspace(0.5, 3.0, emb_dim)
            token_encoding = np.sin(freq_basis * (idx % 7 + 1)) + np.cos(
                freq_basis * word_len
            )
            semantic_curve[s, :] += token_encoding * overlap * word_len

    # 方案 B：若未回傳時間戳，退回使用總字數密度分佈
    else:
      words = transcribed_text.split() if transcribed_text else []
      density = max(1, len(words)) / max(1, duration_sec)
      for s in range(duration_sec):
        semantic_curve[s, :] = np.sin(np.linspace(s, s + 1, emb_dim)) * float(
            density * 0.1
        )

    # 對齊為 PyTorch Tensor: (Batch=1, emb_dim, duration_sec)
    tensor_feat = torch.tensor(
        semantic_curve.T, dtype=torch.float32
    ).unsqueeze(0)
    return tensor_feat


# ==============================================================================
# 2. 雙通道模型結構 (Gemini 語意流 + 本地 Log-Mel 聲學流)
# ==============================================================================

class PretrainedResNetAcousticStream(nn.Module):
    """使用 ImageNet 預訓練的 ResNet-18 提取時頻紋理特徵"""

    def __init__(self, out_dim=64, freeze_backbone=True):
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

        # 只做頻率池化，不加未訓練的隨機 Conv1d 投影層，避免扭曲預訓練特徵！
        self.adaptive_freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, mel_spec):
        mean = mel_spec.mean(dim=(-2, -1), keepdim=True).detach()
        std = mel_spec.std(dim=(-2, -1), keepdim=True).detach() + 1e-6
        x = (mel_spec - mean) / std

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))  # (Batch, 256, F', T')
        x = self.adaptive_freq_pool(x).squeeze(2)     # (Batch, 256, T')
        return x

class AcousticCNNStream(nn.Module):
  """處理 Log-Mel Spectrogram 的聲學卷積 (音效、BGM、重拍)"""

  def __init__(self, out_dim=64):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.adaptive_freq_pool = nn.AdaptiveAvgPool2d((1, None))
    self.proj = nn.Conv1d(64, out_dim, kernel_size=1)

  def forward(self, mel_spec):
    x = F.relu(self.bn1(self.conv1(mel_spec)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.adaptive_freq_pool(x).squeeze(2)
    return self.proj(x)


class GeminiAcousticDualStreamModel(nn.Module):
    """結合 Gemini 語意向量、預訓練 ResNet 紋理與真實聲學能量門控的模型"""

    def __init__(self, semantic_dim=64, acoustic_dim=256, fusion_dim=64):
        super().__init__()
        self.acoustic_stream = PretrainedResNetAcousticStream(out_dim=acoustic_dim, freeze_backbone=True)

    def forward(self, semantic_feat, mel_spec, target_seconds):
        # 1. 提取 ResNet 深度時頻紋理特徵 (256 維)
        acoust_raw = self.acoustic_stream(mel_spec)
        acoust_aligned = F.interpolate(acoust_raw, size=target_seconds, mode='linear', align_corners=False)
        acoust_texture = torch.mean(torch.abs(acoust_aligned), dim=1)  # (Batch, target_seconds)

        # 2. 提取真實物理聲學能量 (專取 Mel Bins 10~90 的人聲與主樂器核心頻段，抑制尾奏靜音假高潮)
        # mel_spec 原值為 log(mel)，轉回線性能量後取對數均值
        core_mel = mel_spec[:, :, 10:90, :]
        frame_energy = torch.mean(core_mel, dim=2)  # (Batch, 1, T_mel)
        sec_energy = F.interpolate(frame_energy, size=target_seconds, mode='linear', align_corners=False).squeeze(1)

        # 重要：使用 .detach() 計算 min/max，防止梯度全部集中到極端值那兩秒！
        e_min, e_max = sec_energy.min().detach(), sec_energy.max().detach()
        energy_gate = (sec_energy - e_min) / (e_max - e_min + 1e-6)  # 真實音量門控 [0, 1]

        t_min, t_max = acoust_texture.min().detach(), acoust_texture.max().detach()
        texture_norm = (acoust_texture - t_min) / (t_max - t_min + 1e-6)

        # 聲學綜合得分 = 預訓練紋理複雜度 * 真實音量門控
        acoust_score = texture_norm * energy_gate

        # 3. 提取 Gemini 語意活躍度 (有人說話時 > 0，尾奏無人說話時 = 0)
        sem_aligned = F.interpolate(semantic_feat, size=target_seconds, mode='linear', align_corners=False)
        sem_energy = torch.norm(sem_aligned, p=2, dim=1)
        s_max = sem_energy.max().detach() + 1e-6
        sem_norm = sem_energy / s_max

        # 4. 雙通道融合 (50% 聲學能量與紋理 + 50% 語意字詞密度)
        per_second_scores = 0.5 * acoust_score + 0.5 * sem_norm
        video_level_score = torch.mean(per_second_scores, dim=-1, keepdim=True)

        return per_second_scores, video_level_score


# ==============================================================================
# 3. 本地音訊 Log-Mel 前處理
# ==============================================================================


def prepare_local_log_mel(audio_path, sample_rate=16000):
  try:
    waveform, sr = torchaudio.load(audio_path, backend="soundfile")
  except Exception:
    wav_data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(wav_data)
    if waveform.ndim == 1:
      waveform = waveform.unsqueeze(0)
    else:
      waveform = waveform.t()

  if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

  if sr != sample_rate:
    waveform = T.Resample(sr, sample_rate)(waveform)

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

  return log_mel, duration_sec


# ==============================================================================
# 4. 主執行與 Saliency 視覺化繪圖
# ==============================================================================


def run_gemini_audio_pipeline(
    audio_path, output_png="gemini_audio_saliency.png"
):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # A. 提取本地聲學特徵
  log_mel_spec, duration_sec = prepare_local_log_mel(audio_path)
  log_mel_spec = log_mel_spec.to(device)
  log_mel_spec.requires_grad = True

  # B. 呼叫 Gemini 3.5 Transcribe (帶入 word_timestamp=True 與 diarization=True)
  gemini_proc = GeminiTranscribeProcessor()
  transcribe_res = gemini_proc.transcribe_with_timestamps(audio_path)
  semantic_feat = gemini_proc.build_second_level_semantic_vector(
      duration_sec, transcribe_res
  ).to(device)

  # C. 模型推論與反向傳播 (計算 Saliency Map)
  model = GeminiAcousticDualStreamModel().to(device)
  model.eval()

  per_second_scores, video_score = model(
      semantic_feat, log_mel_spec, duration_sec
  )
  model.zero_grad()
  video_score.backward()

  # 取得 Saliency 梯度熱力圖
  saliency = torch.abs(log_mel_spec.grad).squeeze().cpu().numpy()
  scores = per_second_scores.detach().squeeze().cpu().numpy()
  log_mel_img = log_mel_spec.detach().squeeze().cpu().numpy()

  # D. 繪圖
  fig, axes = plt.subplots(
      3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.2, 1.2, 1]}
  )

  # 1. Log-Mel Spectrogram
  im0 = axes[0].imshow(
      log_mel_img, aspect="auto", origin="lower", cmap="inferno"
  )
  axes[0].set_title(
      "1. Local Log-Mel Spectrogram (Acoustics + BGM + SFX)",
      fontsize=11,
      fontweight="bold",
  )
  plt.colorbar(im0, ax=axes[0], pad=0.01)

  # 2. Saliency Map
  im1 = axes[1].imshow(saliency, aspect="auto", origin="lower", cmap="hot")
  axes[1].set_title(
      r"2. Audio Saliency Heat Map $|\partial \hat{y} / \partial X(f, t)|$",
      fontsize=11,
      fontweight="bold",
  )
  plt.colorbar(im1, ax=axes[1], pad=0.01)

  # 3. 逐秒分數曲線
  t_axis = np.arange(1, duration_sec + 1)
  axes[2].plot(
      t_axis, scores, color="#1f77b4", linewidth=2.2, marker="o", markersize=4
  )
  axes[2].axhline(
      np.mean(scores),
      color="r",
      linestyle="--",
      label=f"Mean: {np.mean(scores):.3f}",
  )
  axes[2].set_title(
      "3. Second-level Engagement (Gemini 3.5 Semantics + Local Acoustics)",
      fontsize=11,
      fontweight="bold",
  )
  axes[2].set_xlim(1, duration_sec)
  axes[2].set_ylim(0, 1.05)
  axes[2].grid(True, linestyle="--", alpha=0.5)
  axes[2].legend(loc="upper right")

  plt.tight_layout()
  plt.savefig(output_png, dpi=200)
  print(f"完成！已生成 Saliency 熱力圖至: {output_png}")
  plt.show()


if __name__ == "__main__":
  AUDIO_PATH = r"C:\Users\Anthony\Desktop\video_001.mp3"

  if not os.path.exists(AUDIO_PATH):
    AUDIO_PATH = "demo_test.wav"
    t = torch.linspace(0, 10, 160000)
    wav = (0.5 * torch.sin(2 * np.pi * 440 * t)).numpy()
    sf.write(AUDIO_PATH, wav, 16000)

  run_gemini_audio_pipeline(AUDIO_PATH)