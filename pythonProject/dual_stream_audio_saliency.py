import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from dotenv import load_dotenv
import soundfile as sf


# 設定你的 Gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key


# ==============================================================================
# 1. Gemini 3.5 Transcribe 前處理器 (取得文字與精確秒級時間戳)
# ==============================================================================

class GeminiTranscribeProcessor:

  def __init__(self, api_key=None):
    self.client = genai.Client(
        api_key=api_key or os.environ.get("GEMINI_API_KEY")
    )

  def transcribe_with_timestamps(self, audio_path):
    print(f"正在上傳音訊至 Gemini 進行轉錄: {os.path.basename(audio_path)}...")
    audio_file = self.client.files.upload(file=audio_path)

    response = self.client.models.generate_content(
        model="gemini-3.5-transcribe",
        contents=[audio_file],
    )
    return response

  def extract_transcription_data(self, response):
    """從 candidates.content.parts 中提取 audio_transcription 或文字內容"""
    transcribed_text = ""
    words_info = []

    try:
      candidate = response.candidates[0]
      for part in candidate.content.parts:
        # 1. 檢查是否存在專屬的 audio_transcription 物件
        if hasattr(part, "audio_transcription") and part.audio_transcription:
          transcription_obj = part.audio_transcription

          # 取得完整文字
          if hasattr(transcription_obj, "text"):
            transcribed_text += transcription_obj.text + " "

          # 取得詞級時間戳記 (Word-level timestamps)
          if hasattr(transcription_obj, "words"):
            for w in transcription_obj.words:
              words_info.append({
                  "word": getattr(w, "word", ""),
                  "start": getattr(w, "start_offset", 0.0),
                  "end": getattr(w, "end_offset", 0.0),
              })

        # 2. 備選：若為一般 text part
        elif hasattr(part, "text") and part.text:
          transcribed_text += part.text + " "

    except Exception as e:
      print(f"解析 Gemini 回傳格式時出現例外: {e}")

    # 若上述皆未命中，退回檢查 response.text
    if not transcribed_text.strip():
      transcribed_text = getattr(response, "text", "") or ""

    return transcribed_text.strip(), words_info

  def build_second_level_semantic_vector(
      self, duration_sec, response, emb_dim=64
  ):
    """依據 Gemini 回傳的逐字文字或時間戳記，建立秒級語意表徵向量 (1, emb_dim, duration_sec)"""
    transcribed_text, words_info = self.extract_transcription_data(response)

    print(
        f"Gemini 轉錄成功 (共 {len(transcribed_text)} 字):"
        f" {transcribed_text[:60]}..."
    )

    semantic_curve = np.zeros((duration_sec, emb_dim), dtype=np.float32)

    # 方案 A：如果有精確的詞級時間戳 (Word-level timestamps)
    if words_info:
      for item in words_info:
        start_sec = int(min(max(0, item["start"]), duration_sec - 1))
        # 語音活躍的秒數賦予正弦語意權重
        semantic_curve[start_sec, :] += np.sin(
            np.linspace(0.1, 1.0, emb_dim)
        ) * len(item["word"])

    # 方案 B：若無時間戳，使用總字數均勻分配
    else:
      words = transcribed_text.split() if transcribed_text else []
      density = max(1, len(words) // max(1, duration_sec))
      for s in range(duration_sec):
        semantic_curve[s, :] = np.sin(
            np.linspace(s, s + 1, emb_dim)
        ) * float(density * 0.1)

    # 轉為 PyTorch Tensor: (Batch=1, emb_dim, duration_sec)
    tensor_feat = torch.tensor(
        semantic_curve.T, dtype=torch.float32
    ).unsqueeze(0)
    return tensor_feat


# ==============================================================================
# 2. 雙通道模型結構 (Gemini 語意流 + 本地 Log-Mel 聲學流)
# ==============================================================================

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
    """結合 Gemini 語意向量與原生 Log-Mel 頻譜的模型"""

    def __init__(self, semantic_dim=64, acoustic_dim=64, fusion_dim=64):
        super().__init__()
        self.semantic_proj = nn.Conv1d(semantic_dim, fusion_dim, kernel_size=1)
        self.acoustic_stream = AcousticCNNStream(out_dim=acoustic_dim)
        self.acoustic_proj = nn.Conv1d(acoustic_dim, fusion_dim, kernel_size=1)

        # 時序融合頭
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_dim * 2, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Conv1d(fusion_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, semantic_feat, mel_spec, target_seconds):
        # 1. 語意分支對齊到秒數
        sem_aligned = F.interpolate(self.semantic_proj(semantic_feat), size=target_seconds, mode='linear')

        # 2. 聲學分支對齊到秒數
        acoust_aligned = F.interpolate(self.acoustic_proj(self.acoustic_stream(mel_spec)), size=target_seconds,
                                       mode='linear')

        # 3. 拼接並計算逐秒評分
        combined = torch.cat([sem_aligned, acoust_aligned], dim=1)
        per_second_scores = self.fusion(combined).squeeze(1)  # (Batch, target_seconds)
        video_level_score = torch.mean(per_second_scores, dim=-1, keepdim=True)
        return per_second_scores, video_level_score


# ==============================================================================
# 3. 本地音訊 Log-Mel 前處理
# ==============================================================================

def prepare_local_log_mel(audio_path, sample_rate=16000):
    # 新版 torchaudio 指定 backend 的標準寫法
    try:
        waveform, sr = torchaudio.load(audio_path, backend="soundfile")
    except Exception:
        # 備選方案：若 soundfile 後端仍有問題，改用 soundfile 原生套件直接讀取
        import soundfile as sf

        wav_data, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(wav_data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

    # 轉為單聲道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 重採樣至 16000 Hz
    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)

    duration_sec = int(waveform.shape[-1] / sample_rate)
    if duration_sec < 1:
        duration_sec = 1

    # 計算 Log-Mel Spectrogram
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

def run_gemini_audio_pipeline(audio_path, output_png="gemini_audio_saliency.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A. 提取本地聲學特徵
    log_mel_spec, duration_sec = prepare_local_log_mel(audio_path)
    log_mel_spec = log_mel_spec.to(device)
    log_mel_spec.requires_grad = True

    # B. 呼叫 Gemini 3.5 Transcribe
    gemini_proc = GeminiTranscribeProcessor()
    transcribe_res = gemini_proc.transcribe_with_timestamps(audio_path)
    semantic_feat = gemini_proc.build_second_level_semantic_vector(duration_sec, transcribe_res).to(device)

    # C. 模型推論與反向傳播 (計算 Saliency Map)
    model = GeminiAcousticDualStreamModel().to(device)
    model.eval()

    per_second_scores, video_score = model(semantic_feat, log_mel_spec, duration_sec)
    model.zero_grad()
    video_score.backward()

    # 取得 Saliency 梯度熱力圖
    saliency = torch.abs(log_mel_spec.grad).squeeze().cpu().numpy()
    scores = per_second_scores.detach().squeeze().cpu().numpy()
    log_mel_img = log_mel_spec.detach().squeeze().cpu().numpy()

    # D. 繪圖
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1.2, 1.2, 1]})

    # 1. Log-Mel Spectrogram
    im0 = axes[0].imshow(log_mel_img, aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title("1. Local Log-Mel Spectrogram (Acoustics + BGM + SFX)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], pad=0.01)

    # 2. Saliency Map
    im1 = axes[1].imshow(saliency, aspect='auto', origin='lower', cmap='hot')
    axes[1].set_title(r"2. Audio Saliency Heat Map $|\partial \hat{y} / \partial X(f, t)|$", fontsize=11,
                      fontweight="bold")
    plt.colorbar(im1, ax=axes[1], pad=0.01)

    # 3. 逐秒分數曲線
    t_axis = np.arange(1, duration_sec + 1)
    axes[2].plot(t_axis, scores, color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    axes[2].axhline(np.mean(scores), color="r", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
    axes[2].set_title("3. Second-level Engagement (Gemini 3.5 Semantics + Local Acoustics)", fontsize=11,
                      fontweight="bold")
    axes[2].set_xlim(1, duration_sec)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"完成！已生成 Saliency 熱力圖至: {output_png}")
    plt.show()


if __name__ == "__main__":
    AUDIO_PATH = r"C:\Users\Anthony\Desktop\video_002.mp3"

    # 測試執行 (若無音訊檔會自動生成 demo 音訊)
    if not os.path.exists(AUDIO_PATH):
        AUDIO_PATH = "demo_test.wav"
        t = torch.linspace(0, 10, 160000)
        wav = (0.5 * torch.sin(2 * np.pi * 440 * t)).numpy()
        sf.write(AUDIO_PATH, wav, 16000)

    run_gemini_audio_pipeline(AUDIO_PATH)
