import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from transformers import WhisperModel, WhisperFeatureExtractor


# ==============================================================================
# 1. 雙通道模型結構定義
# ==============================================================================

class AcousticCNNStream(nn.Module):
    """處理 Log-Mel Spectrogram 的聲學卷積分支 (捕捉音效、BGM、能量躍遷)"""

    def __init__(self, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 沿頻率維度做自適應平均池化，保留時間軸
        self.adaptive_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Conv1d(128, out_dim, kernel_size=1)

    def forward(self, mel_spec):
        # mel_spec shape: (Batch, 1, n_mels, T_mel)
        x = F.relu(self.bn1(self.conv1(mel_spec)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.adaptive_freq_pool(x).squeeze(2)  # shape: (Batch, 128, T_compressed)
        x = self.proj(x)  # shape: (Batch, out_dim, T_compressed)
        return x


class DualStreamAudioEngagementModel(nn.Module):
    """雙通道時序吸睛度分析模型"""

    def __init__(self, whisper_model_name="openai/whisper-tiny", acoustic_dim=128, fusion_dim=128):
        super().__init__()
        # Stream 1: Whisper Encoder (語音內容與語意)
        print(f"正在載入 Whisper 預訓練權重: {whisper_model_name}...")
        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.whisper_encoder = whisper.encoder
        # 凍結部分 Whisper 權重以進行遷移學習
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        whisper_hidden_dim = self.whisper_encoder.config.d_model
        self.whisper_proj = nn.Conv1d(whisper_hidden_dim, fusion_dim, kernel_size=1)

        # Stream 2: Log-Mel Acoustic CNN (背景音樂與音效)
        self.acoustic_stream = AcousticCNNStream(out_dim=acoustic_dim)
        self.acoustic_proj = nn.Conv1d(acoustic_dim, fusion_dim, kernel_size=1)

        # 時序融合層 (Temporal Fusion Head)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(fusion_dim * 2, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(fusion_dim, 1, kernel_size=1),
            nn.Sigmoid()  # 輸出 0~1 的吸睛度數值
        )

    def forward(self, whisper_features, mel_spec, target_seconds):
        """
        :param whisper_features: (Batch, 80, 3000) Whisper 輸入特徵
        :param mel_spec: (Batch, 1, 128, T_mel) 原生 Log-Mel 頻譜圖
        :param target_seconds: 影片音訊長度 (秒數)
        :return: per_second_scores (Batch, target_seconds), video_engagement_score (Batch, 1)
        """
        # 1. 提取 Whisper Encoder 表徵
        whisper_out = self.whisper_encoder(whisper_features).last_hidden_state  # (Batch, T_w, D)
        whisper_out = whisper_out.transpose(1, 2)  # (Batch, D, T_w)
        whisper_feat = self.whisper_proj(whisper_out)  # (Batch, fusion_dim, T_w)

        # 2. 提取 Log-Mel 聲學表徵
        acoustic_feat = self.acoustic_stream(mel_spec)  # (Batch, acoustic_dim, T_a)
        acoustic_feat = self.acoustic_proj(acoustic_feat)  # (Batch, fusion_dim, T_a)

        # 3. 雙通道插值對齊到秒級尺度 (target_seconds)
        whisper_aligned = F.interpolate(whisper_feat, size=target_seconds, mode='linear', align_corners=False)
        acoustic_aligned = F.interpolate(acoustic_feat, size=target_seconds, mode='linear', align_corners=False)

        # 4. 特徵拼接與時序卷積
        combined = torch.cat([whisper_aligned, acoustic_aligned], dim=1)  # (Batch, fusion_dim*2, target_seconds)
        per_second_scores = self.fusion_conv(combined).squeeze(1)  # (Batch, target_seconds)

        # 整體音訊吸睛度 (宏觀純量，用於梯度反向傳播)
        video_level_score = torch.mean(per_second_scores, dim=-1, keepdim=True)

        return per_second_scores, video_level_score


# ==============================================================================
# 2. 音訊前處理與特徵提取
# ==============================================================================

class AudioPreprocessor:
    def __init__(self, whisper_model_name="openai/whisper-tiny", sample_rate=16000):
        self.sample_rate = sample_rate
        self.whisper_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)

        # 建立 Log-Mel 頻譜轉換器 (128 mel-bins)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128
        )

    def process(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)

        # 轉為單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 統一採樣率至 16000 Hz
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        duration_sec = int(waveform.shape[-1] / self.sample_rate)
        if duration_sec < 1:
            raise ValueError("音訊長度小於 1 秒，無法分析。")

        # 1. 取得 Whisper 標準輸入 (Pad/Truncate 至 30 秒區塊)
        wav_np = waveform.squeeze(0).numpy()
        whisper_inputs = self.whisper_extractor(wav_np, sampling_rate=self.sample_rate, return_tensors="pt")
        whisper_features = whisper_inputs.input_features  # shape: (1, 80, 3000)

        # 2. 取得全音訊長度的 Log-Mel 頻譜圖 (可保留梯度)
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-6).unsqueeze(0)  # shape: (1, 1, 128, T_mel)

        return whisper_features, log_mel_spec, duration_sec


# ==============================================================================
# 3. 梯度顯著性熱力圖 (Saliency Map) 與逐秒評分計算
# ==============================================================================

def compute_audio_saliency_and_scores(model, whisper_feat, log_mel_spec, duration_sec, device="cpu"):
    """
    計算逐秒評分並反向求導生成類似論文的時頻 Saliency Map
    """
    model.eval()
    whisper_feat = whisper_feat.to(device)
    log_mel_spec = log_mel_spec.to(device)

    # 開啟頻譜圖的梯度追蹤
    log_mel_spec.requires_grad = True

    # 前向傳播
    per_second_scores, video_score = model(whisper_feat, log_mel_spec, target_seconds=duration_sec)

    # 對整體預測值反向求導，獲得輸入時頻圖各頻段/時點的梯度偏導
    model.zero_grad()
    video_score.backward()

    # Saliency Map = 梯度的絕對值 (Simonyan et al., 2014)
    saliency = torch.abs(log_mel_spec.grad).squeeze().cpu().numpy()  # (128, T_mel)

    scores = per_second_scores.detach().squeeze().cpu().numpy()
    log_mel_img = log_mel_spec.detach().squeeze().cpu().numpy()

    return scores, saliency, log_mel_img


# ==============================================================================
# 4. 視覺化繪圖：時頻熱力圖與秒級吸睛曲線
# ==============================================================================

def plot_audio_heatmaps(scores, saliency, log_mel_img, duration_sec, output_path="audio_saliency_heatmap.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False, gridspec_kw={'height_ratios': [1.2, 1.2, 1]})

    # 1. 原始 Log-Mel 頻譜圖
    im0 = axes[0].imshow(log_mel_img, aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title("1. Log-Mel Spectrogram (Full Audio: Voice + BGM + SFX)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mel Frequency Bins", fontsize=10)
    plt.colorbar(im0, ax=axes[0], pad=0.01)

    # 2. 頻譜顯著性熱力圖 (Saliency Map: 哪些頻段和時間最吸睛)
    im1 = axes[1].imshow(saliency, aspect='auto', origin='lower', cmap='hot')
    axes[1].set_title(r"2. Audio Saliency Heat Map $|\partial \hat{y} / \partial X(f, t)|$ (Gradient-based)",
                      fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mel Frequency Bins", fontsize=10)
    plt.colorbar(im1, ax=axes[1], pad=0.01)

    # 3. 逐秒吸睛度得分曲線 (Per-Second Engagement Curve)
    time_axis = np.arange(1, duration_sec + 1)
    axes[2].plot(time_axis, scores, color='#1f77b4', linewidth=2.2, marker='o', markersize=4,
                 label="Second-level Score")
    axes[2].axhline(np.mean(scores), color='r', linestyle='--', label=f"Mean Score ({np.mean(scores):.3f})")

    # 標記吸睛高峰秒數 (Top-3 Peaks)
    top_peaks = np.argsort(scores)[-3:]
    for peak_idx in top_peaks:
        axes[2].annotate(f"{scores[peak_idx]:.2f}",
                         (time_axis[peak_idx], scores[peak_idx]),
                         textcoords="offset points", xytext=(0, 7), ha='center',
                         fontweight='bold', color='darkred', fontsize=9)

    axes[2].set_title("3. Predicted Per-Second Engagement Curve", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (Seconds)", fontsize=10)
    axes[2].set_ylabel("Score [0, 1]", fontsize=10)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlim(1, duration_sec)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"\n視覺化圖表已成功輸出至: {os.path.abspath(output_path)}")
    plt.show()


# ==============================================================================
# 主執行程序入口
# ==============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"運行設備: {device}")

    # 測試用音訊檔案路徑 (可替換為你 database 中的任一影片提取的 mp3/wav 檔)
    AUDIO_FILE = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\reelsData\video_001.mp3"

    # 若音訊檔不存在，自動生成 15 秒合成測試訊號
    if not os.path.exists(AUDIO_FILE):
        print(f"提示：找不到指定路徑 {AUDIO_FILE}，建立 15 秒白噪音/正弦混合音訊進行流程測試...")
        AUDIO_FILE = "demo_test_audio.wav"
        sr = 16000
        t = torch.linspace(0, 15, 15 * sr)
        # 模擬語音正弦波 + 模擬節奏點
        mock_wav = 0.5 * torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn_like(t)
        torchaudio.save(AUDIO_FILE, mock_wav.unsqueeze(0), sr)

    # 1. 前處理音訊
    preprocessor = AudioPreprocessor(whisper_model_name="openai/whisper-tiny")
    whisper_feat, log_mel_spec, duration_sec = preprocessor.process(AUDIO_FILE)
    print(f"成功處理音訊，時長: {duration_sec} 秒")

    # 2. 初始化模型
    model = DualStreamAudioEngagementModel(whisper_model_name="openai/whisper-tiny").to(device)

    # 3. 計算逐秒分數與 Saliency Map
    scores, saliency, log_mel_img = compute_audio_saliency_and_scores(
        model, whisper_feat, log_mel_spec, duration_sec, device=device
    )

    print("\n【逐秒吸睛度分數 (前 10 秒)】")
    for sec, sc in enumerate(scores[:10], 1):
        print(f"  - 第 {sec:02d} 秒: {sc:.4f}")

    # 4. 繪製並保存熱力圖
    plot_audio_heatmaps(scores, saliency, log_mel_img, duration_sec, output_path="audio_saliency_heatmap.png")