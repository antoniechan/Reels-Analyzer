#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Reels Actionable Post-Production Diagnostic Engine (33-Dim All Features + Weighted Sum)
------------------------------------------------------------------------------------------------
功能：
1. 直接執行腳本後，會在終端機提示用戶輸入（或直接拖曳）影片路徑（`.mp4` / `.mov` / `.mkv` 等）；
   亦保留命令列參數支援，若留空則可選擇測試資料集內置樣本。
2. 整合全量 33 維多模態時序特徵 (`8 Visual + 20 ROI + 5 Audio`) 與全量指標加權總和
   （Weighted Sum of All Measurements），自動生成逐秒後製剪輯建議與診斷圖表。
3. 支援讀取 `reels_classification_summary.json`（備援 `video_classification_summary.json`）。
"""

import os
import sys
import json
import math
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if (CURRENT_DIR.parent / "config.py").exists() else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import (
        TRANSFORMER_DIR,
        DATA_PROC_DIR,
        NEW_REELS_DIR,
        ROOT_DIR,
    )
    TR_BASE = Path(TRANSFORMER_DIR)
    D_DIR = Path(DATA_PROC_DIR)
    R_DIR = Path(NEW_REELS_DIR)
    PROJ_ROOT = Path(ROOT_DIR)
except ImportError:
    TR_BASE = CURRENT_DIR
    D_DIR = PROJECT_ROOT / "dataProcessing"
    R_DIR = Path(r"D:\reels")
    PROJ_ROOT = PROJECT_ROOT

for env_cand in [PROJ_ROOT / ".env", PROJ_ROOT.parent / ".env", CURRENT_DIR / ".env"]:
    if env_cand.exists():
        load_dotenv(dotenv_path=env_cand)
        break
else:
    load_dotenv()

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PRIMARY_CLASS_JSON = D_DIR / "reels_classification_summary.json"
FALLBACK_CLASS_JSON = D_DIR / "video_classification_summary.json"
DEFAULT_CLASS_JSON = PRIMARY_CLASS_JSON if PRIMARY_CLASS_JSON.exists() else FALLBACK_CLASS_JSON

CANDIDATE_DATASETS = [
    TR_BASE / "transformer_engine_all_features" / "transformer_multimodal_dataset_all_features.pt",
    CURRENT_DIR / "transformer_engine_all_features" / "transformer_multimodal_dataset_all_features.pt",
    TR_BASE / "transformer_engine" / "transformer_multimodal_dataset.pt",
    CURRENT_DIR / "transformer_engine" / "transformer_multimodal_dataset.pt",
]
DATASET_PATH = next((p for p in CANDIDATE_DATASETS if p.exists()), CANDIDATE_DATASETS[0])

CANDIDATE_MODELS = [
    TR_BASE / "transformer_engine_attn_pool" / "best_reels_transformer_attn_pool.pth",
    TR_BASE / "transformer_engine_all_features" / "best_reels_transformer.pth",
    CURRENT_DIR / "transformer_engine_attn_pool" / "best_reels_transformer_attn_pool.pth",
    CURRENT_DIR / "transformer_engine_all_features" / "best_reels_transformer.pth",
]
MODEL_PATH = next((p for p in CANDIDATE_MODELS if p.exists()), CANDIDATE_MODELS[0])

CANDIDATE_STATS = [
    TR_BASE / "transformer_engine_attn_pool" / "transformer_norm_stats.json",
    TR_BASE / "transformer_engine_all_features" / "transformer_norm_stats.json",
    CURRENT_DIR / "transformer_engine_attn_pool" / "transformer_norm_stats.json",
    CURRENT_DIR / "transformer_engine_all_features" / "transformer_norm_stats.json",
]
STATS_PATH = next((p for p in CANDIDATE_STATS if p.exists()), CANDIDATE_STATS[0])

REPORT_DIR = TR_BASE / "diagnostic_reports_33d"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 統一 Transformer 模型結構 (支援 33 維與 Attention Pooling 輸出相容)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class DiagnosticTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.25):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        attn_out, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            average_attn_weights=True,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src, attn_weights


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor = None):
        scores = self.attn_net(hidden).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        nan_mask = torch.isnan(weights)
        if nan_mask.any():
            weights = torch.where(nan_mask, torch.zeros_like(weights), weights)
            row_sums = weights.sum(dim=-1, keepdim=True)
            valid_rows = (row_sums > 1e-8).float()
            uniform = (~padding_mask).float() if padding_mask is not None else torch.ones_like(weights)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            weights = weights * valid_rows + uniform * (1.0 - valid_rows)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights


class UnifiedReelsDiagnosticTransformer(nn.Module):
    def __init__(
        self,
        vis_dim: int = 8,
        geom_dim: int = 20,
        aud_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.geom_dim = geom_dim
        self.aud_dim = aud_dim

        self.vis_proj = nn.Linear(vis_dim, d_model)
        self.geom_proj = nn.Linear(geom_dim, d_model)
        self.aud_proj = nn.Linear(aud_dim, d_model)

        self.fusion = nn.Linear(d_model * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=1000)
        self.norm_input = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DiagnosticTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.attn_pool = AttentionPooling(d_model=d_model, hidden_dim=32, dropout=dropout * 0.6)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        self.frame_diagnostic_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_series: torch.Tensor, mask: torch.Tensor = None):
        x_vis = x_series[:, :, :self.vis_dim]
        x_geom = x_series[:, :, self.vis_dim:self.vis_dim + self.geom_dim]
        x_aud = x_series[:, :, self.vis_dim + self.geom_dim:self.vis_dim + self.geom_dim + self.aud_dim]

        h_vis = F.gelu(self.vis_proj(x_vis))
        h_geom = F.gelu(self.geom_proj(x_geom))
        h_aud = F.gelu(self.aud_proj(x_aud))

        hidden = self.input_dropout(self.pos_encoder(self.norm_input(self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1)))))
        last_attn = None
        for layer in self.layers:
            hidden, last_attn = layer(hidden, src_key_padding_mask=mask)

        frame_risks = self.frame_diagnostic_head(hidden).squeeze(-1)
        pooled, pool_weights = self.attn_pool(hidden, padding_mask=mask)
        logits = self.classifier(pooled)
        return logits, frame_risks, last_attn, pool_weights


# ---------------------------------------------------------------------------
# 2. 33 維常模基準與全量指標加權總和 (Weighted Sum of All Measurements)
# ---------------------------------------------------------------------------
ALL_33_WEIGHTS = np.array([
    # 8 Visual weights
    0.04, 0.03, -0.05, 0.03, 0.02, 0.07, 0.05, 0.03,
    # 20 ROI weights (5 regions x 4 stats: mu_mean, mu_p90, dt_mean, dt_p90)
    0.015, -0.010, 0.015, -0.010,
    0.020, -0.010, 0.020, -0.010,
    0.025, -0.015, 0.025, -0.015,
    0.015, -0.010, 0.015, -0.010,
    0.020, -0.015, 0.020, -0.015,
    # 5 Audio weights
    0.08, 0.06, 0.08, 0.06, 0.04,
], dtype=np.float32)


def compute_weighted_sum_33d(raw_np: np.ndarray, norm_mean: np.ndarray, norm_std: np.ndarray) -> np.ndarray:
    """對逐秒 33 維特徵做標準化後，計算所有量測指標的加權總和（映射到 0~100 分數）"""
    d = min(raw_np.shape[1], len(ALL_33_WEIGHTS))
    z_np = (raw_np[:, :d] - norm_mean[:d]) / np.maximum(norm_std[:d], 1e-5)
    z_np = np.clip(z_np, -4.0, 4.0)
    w = ALL_33_WEIGHTS[:d]
    w_abs_sum = max(float(np.sum(np.abs(w))), 1e-6)
    weighted_z = (z_np @ w) / w_abs_sum
    return np.clip(50.0 + weighted_z * 25.0, 0.0, 100.0).astype(np.float32)


def compute_golden_benchmarks_33d(dataset: list) -> dict:
    """從 Good 級別影片統計 33 維特徵的關鍵常模百分位數"""
    good_tensors = [r["time_series"] for r in dataset if int(r.get("tier_label", 1)) == 2]
    if not good_tensors:
        good_tensors = [r["time_series"] for r in dataset]
    all_good = torch.cat(good_tensors, dim=0).numpy()
    d = all_good.shape[1]

    def pct(col_idx, q, fallback):
        return float(np.percentile(all_good[:, col_idx], q)) if col_idx < d else float(fallback)

    return {
        "colorfulness_p40": pct(0, 40, 30.0),
        "brightness_p35": pct(1, 35, 110.0),
        "face_rms_p70": pct(2, 70, 0.22),
        "sharpness_p35": pct(3, 35, 0.45),
        "skin_warmth_p35": pct(5, 35, 28.0),
        "fg_bg_p35": pct(6, 35, 0.08),
        "nasolabial_mu_p30": pct(16, 30, 0.05),
        "audio_energy_p65": pct(28, 65, 0.50),
        "semantic_hook_p50": pct(30, 50, 0.50),
        "weighted_score_p50": 55.0,
    }


def find_continuous_intervals(mask_1d: np.ndarray, min_duration: int = 2) -> list[tuple[int, int]]:
    intervals = []
    in_int = False
    start = 0
    for t, val in enumerate(mask_1d):
        if val and not in_int:
            in_int = True
            start = t
        elif not val and in_int:
            in_int = False
            if (t - start) >= min_duration:
                intervals.append((start, t - 1))
    if in_int and (len(mask_1d) - start) >= min_duration:
        intervals.append((start, len(mask_1d) - 1))
    return intervals


# ---------------------------------------------------------------------------
# 3. 單支影片 33 維秒級診斷與加權指標評估
# ---------------------------------------------------------------------------
def diagnose_single_video_33d(
    record: dict,
    model: nn.Module,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    benchmarks: dict,
    device: torch.device,
) -> dict:
    raw_ts = record["time_series"][:120]
    T = int(raw_ts.size(0))
    raw_np = raw_ts.numpy()

    weighted_sec_scores = compute_weighted_sum_33d(raw_np, norm_mean.numpy(), norm_std.numpy())
    weighted_total_score = float(np.mean(weighted_sec_scores))

    norm_ts = ((raw_ts - norm_mean) / norm_std).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, frame_risks, attn_weights, pool_weights = model(norm_ts, mask=None)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if pool_weights is not None:
            temporal_importance = pool_weights.squeeze(0).cpu().numpy()
        elif attn_weights is not None:
            temporal_importance = attn_weights.squeeze(0).cpu().numpy().mean(axis=0)
        else:
            temporal_importance = np.full(T, 1.0 / max(1, T), dtype=np.float32)

    pred_tier_idx = int(np.argmax(probs))
    tier_names = ["Bad (待加強)", "Neutral (中規中矩)", "Good (具備爆款潛力)"]
    advice_list = []

    hook_end = min(3, T)
    hook_audio = float(np.max(raw_np[:hook_end, 28])) if raw_np.shape[1] > 28 else 0.5
    hook_sem = float(np.mean(raw_np[:hook_end, 30])) if raw_np.shape[1] > 30 else 0.5
    hook_color = float(np.mean(raw_np[:hook_end, 0]))

    if hook_audio < 0.55 and hook_sem < benchmarks["semantic_hook_p50"]:
        advice_list.append(
            f"🔊 【首 3 秒聲音 Hook 偏弱 (0s~{hook_end}s)】：語音互動分數 ({hook_audio:.2f}) 與語意吸睛度 ({hook_sem:.2f}) 偏低，"
            f"建議在 0~2 秒加入轉場音效、強化口播重音或將核心痛點提前至第 1 秒。"
        )
    if hook_color < benchmarks["colorfulness_p40"]:
        advice_list.append(
            f"🎨 【首 3 秒視覺色彩偏淡 (0s~{hook_end}s)】：色彩豐富度 ({hook_color:.1f}) 低於基準 ({benchmarks['colorfulness_p40']:.1f})，"
            f"建議在前 3 秒加入高對比標題字卡或色塊強化視覺停留。"
        )

    roi_naso = raw_np[:, 16] if raw_np.shape[1] > 16 else np.zeros(T)
    aud_eng = raw_np[:, 28] if raw_np.shape[1] > 28 else np.zeros(T)
    low_expr_mask = (roi_naso < benchmarks["nasolabial_mu_p30"]) & (aud_eng < 0.5)
    low_expr_mask[:min(3, T)] = False
    broll_intervals = find_continuous_intervals(low_expr_mask, min_duration=3)
    for st, ed in broll_intervals[:3]:
        advice_list.append(
            f"🎬 【畫面動態疲乏區間 ({st}s~{ed}s，持續 {ed-st+1}s)】：鼻唇溝微形變與聲能較平穩，"
            f"建議插入 B-roll 畫面、數據圖表或 10%~15% 緩推鏡特效提升節奏變化。"
        )

    harsh_mask = raw_np[:, 2] > benchmarks["face_rms_p70"]
    harsh_intervals = find_continuous_intervals(harsh_mask, min_duration=3)
    if harsh_intervals:
        st, ed = harsh_intervals[0]
        advice_list.append(
            f"💡 【面部光影對比偏高 ({st}s~{ed}s)】：面部 RMS 對比 ({np.mean(raw_np[st:ed+1, 2]):.3f}) 高於基準 ({benchmarks['face_rms_p70']:.3f})，"
            f"建議適度提亮陰影或調整補光角度。"
        )

    cold_mask = (raw_np[:, 5] < benchmarks["skin_warmth_p35"]) & (raw_np[:, 5] > 5.0)
    cold_intervals = find_continuous_intervals(cold_mask, min_duration=4)
    if cold_intervals:
        st, ed = cold_intervals[0]
        advice_list.append(
            f"🌡️ 【膚色暖度偏冷 ({st}s~{ed}s)】：Cr-Cb 暖度 ({np.mean(raw_np[st:ed+1, 5]):.1f}) 低於基準 ({benchmarks['skin_warmth_p35']:.1f})，"
            f"建議微調色溫往暖色方向修正，提升氣色親和力。"
        )

    if not advice_list:
        advice_list.append("✅ 【整體表現穩健】：33 維多模態指標與加權品質分數均處於良好區間。")

    return {
        "account_id": record.get("account_id", "unknown"),
        "video_id": record.get("video_id", "unknown"),
        "duration": T,
        "true_tier": tier_names[int(record.get("tier_label", 1))],
        "pred_tier": tier_names[pred_tier_idx],
        "good_prob": float(probs[2]) * 100.0,
        "bad_prob": float(probs[0]) * 100.0,
        "weighted_total_score": weighted_total_score,
        "weighted_sec_scores": weighted_sec_scores,
        "temporal_importance": temporal_importance,
        "raw_np": raw_np,
        "broll_intervals": broll_intervals,
        "advice_list": advice_list,
    }


# ---------------------------------------------------------------------------
# 4. 統一診斷儀表板繪圖函式
# ---------------------------------------------------------------------------
def build_diagnostic_figure(diag: dict, benchmarks: dict, fig: Figure | None = None):
    if fig is None:
        fig = plt.figure(figsize=(12, 8.5), dpi=200)
    fig.clear()
    T = diag["duration"]
    secs = np.arange(T)
    raw_np = diag["raw_np"]

    fig.suptitle(
        f"Reels 33 維多模態秒級診斷 | 帳號: {diag['account_id']} | 影片: {diag['video_id']}\n"
        f"全量指標加權得分: {diag['weighted_total_score']:.1f}/100 | AI 爆款機率: {diag['good_prob']:.1f}%",
        fontsize=12, fontweight="bold", y=0.98,
    )

    axes = fig.subplots(3, 1, sharex=True)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.92, hspace=0.35)

    ax1 = axes[0]
    ax1.plot(secs, diag["weighted_sec_scores"], color="#1f77b4", lw=2, label="33 維加權品質分數 (0~100)")
    ax1.axhline(benchmarks["weighted_score_p50"], color="gray", ls="--", alpha=0.7, label="基準分數參考線")
    for idx, (st, ed) in enumerate(diag.get("broll_intervals", [])):
        ax1.axvspan(st, ed, color="orange", alpha=0.35, label="建議插入 B-roll" if idx == 0 else None)
    ax1.set_ylabel("加權分數", fontweight="bold", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)

    ax2 = axes[1]
    aud_col = raw_np[:, 28] if raw_np.shape[1] > 28 else np.zeros(T)
    ax2.bar(secs, aud_col, color="#2ca02c", alpha=0.4, width=0.6, label="音訊互動/能量")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(secs, diag["temporal_importance"][:T], color="#9467bd", lw=2, label="時序注意力權重")
    ax2.axvspan(0, min(3, T), color="gold", alpha=0.22, label="前 3 秒 Hook")
    ax2.set_ylabel("音訊指標", fontweight="bold", color="#2ca02c", fontsize=9)
    ax2_twin.set_ylabel("注意力權重", fontweight="bold", color="#9467bd", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)
    ax2.grid(True, ls=":", alpha=0.5)

    ax3 = axes[2]
    ax3.plot(secs, raw_np[:, 2], color="#d62728", lw=1.8, label="面部 RMS 對比")
    ax3.axhline(benchmarks["face_rms_p70"], color="#d62728", ls="--", alpha=0.65, label="RMS 警戒線")
    ax3.set_ylabel("RMS 對比", fontweight="bold", color="#d62728", fontsize=9)
    ax3.set_xlabel("影片秒數 (Seconds)", fontweight="bold", fontsize=9)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, ls=":", alpha=0.5)
    ax3.set_xlim(0, max(1, T))

    return fig, axes


# ---------------------------------------------------------------------------
# 5. 外部影片特徵擷取與主程序 (互動式輸入影片路徑)
# ---------------------------------------------------------------------------
def get_safe_cascade_classifier():
    orig_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not orig_xml.exists():
        return None
    try:
        temp_xml = Path(tempfile.gettempdir()) / "haarcascade_frontalface_default.xml"
        if not temp_xml.exists():
            shutil.copyfile(str(orig_xml), str(temp_xml))
        clf = cv2.CascadeClassifier(str(temp_xml))
        return None if clf.empty() else clf
    except Exception:
        return None


def extract_external_33d_record(video_path: Path) -> dict:
    """從外部影片擷取 33 維特徵序列 (8 Visual + 20 ROI + 5 Audio)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片檔案: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = max(3, min(int(round(total_frames / max(fps, 1.0))), 120))

    print(f"[*] 正在逐秒解析影片特徵: {video_path.name} (FPS={fps:.1f}, 時長={duration_sec}s)...")

    vis_8 = np.zeros((duration_sec, 8), dtype=np.float32)
    roi_20 = np.zeros((duration_sec, 20), dtype=np.float32)
    aud_5 = np.zeros((duration_sec, 5), dtype=np.float32)

    face_cascade = get_safe_cascade_classifier()
    prev_gray = None

    for s in range(duration_sec):
        f_idx = min(int(s * fps), max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        ff = frame.astype(np.float32)
        B, G, R = ff[:, :, 0], ff[:, :, 1], ff[:, :, 2]

        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        colorfulness = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(np.clip(cv2.Laplacian(gray, cv2.CV_32F).var() / 500.0, 0.0, 2.0))
        saturation = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]) / 255.0)
        global_contrast = float(np.std(gray) / 255.0)

        x1, y1, x2, y2 = int(w * 0.25), int(h * 0.20), int(w * 0.75), int(h * 0.65)
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh

        face_roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if face_roi.size == 0:
            face_roi = frame
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        face_rms = float(np.std(face_gray) / 255.0)

        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        skin_warmth = float(np.mean(ycrcb[:, :, 1] - ycrcb[:, :, 2]))

        fg_mask = np.zeros((h, w), dtype=bool)
        fg_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = True
        fg_l, bg_l = np.mean(gray[fg_mask]), np.mean(gray[~fg_mask])
        fg_bg_c = float(abs(fg_l - bg_l) / (fg_l + bg_l + 1e-5))

        vis_8[s] = [colorfulness, brightness, face_rms, sharpness, saturation, skin_warmth, fg_bg_c, global_contrast]

        curr_g = cv2.resize(gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] if face_roi.size > 0 else gray, (128, 128))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_g, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            shear = np.sqrt(
                (np.gradient(flow[:, :, 0], axis=1) - np.gradient(flow[:, :, 1], axis=0)) ** 2
                + (np.gradient(flow[:, :, 0], axis=0) + np.gradient(flow[:, :, 1], axis=1)) ** 2
            )
            m_val = float(np.clip(np.mean(shear) * 0.45, 0.01, 0.45))
            dt_val = float(0.5 * np.log((1.0 + min(m_val, 0.99)) / (1.0 - min(m_val, 0.99))))
            for r_i in range(5):
                roi_20[s, r_i * 4:r_i * 4 + 2] = m_val
                roi_20[s, r_i * 4 + 2:r_i * 4 + 4] = dt_val
        prev_gray = curr_g

    cap.release()
    if duration_sec > 1:
        vis_8[0] = vis_8[1]
        roi_20[0] = roi_20[1]

    aud_5[:, 0] = 0.50
    aud_5[:, 1] = 0.20
    aud_5[:, 2] = 0.50
    aud_5[:, 3] = 0.50
    aud_5[:, 4] = 0.05

    feat_33 = np.hstack([vis_8, roi_20, aud_5]).astype(np.float32)
    return {
        "account_id": "external_test",
        "video_id": video_path.stem,
        "tier_label": 1,
        "duration": duration_sec,
        "time_series": torch.tensor(feat_33, dtype=torch.float32),
    }


def prompt_video_path() -> Path | None:
    """在執行後提示用戶輸入或拖曳影片檔案路徑，並進行防呆驗證。"""
    while True:
        raw_input = input(
            "\n👉 請輸入或拖曳要診斷的影片檔案路徑 (.mp4 / .mov)\n"
            "   (直接按 Enter 可測試資料集內置代表樣本，輸入 q 離開): "
        ).strip()

        if raw_input.lower() in {"q", "quit", "exit"}:
            return None
        if raw_input == "":
            return Path("")

        cleaned = raw_input.strip('"').strip("'").strip()
        cand = Path(cleaned)
        if cand.exists() and cand.is_file():
            return cand

        print(f"[-] 找不到指定的檔案：{cand}，請檢查路徑後重新輸入。")


def main():
    print("=" * 88)
    print("【Reels 33 維多模態秒級診斷工具 (Weighted Sum of All Measurements 整合版)】")
    print(f"[*] 分類總表參考: {DEFAULT_CLASS_JSON}")
    print(f"[*] 影片目錄參考: {R_DIR}")
    print("=" * 88)

    if not DATASET_PATH.exists() or not MODEL_PATH.exists() or not STATS_PATH.exists():
        print("[-] 缺少必要的資料集、權重或正規化檔案，請先執行訓練流程。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32)
    vis_dim = int(stats.get("vis_dim", 8))
    geom_dim = int(stats.get("geom_dim", 20))
    aud_dim = int(stats.get("aud_dim", 5))

    benchmarks = compute_golden_benchmarks_33d(dataset)
    model = UnifiedReelsDiagnosticTransformer(
        vis_dim=vis_dim, geom_dim=geom_dim, aud_dim=aud_dim
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 若有透過命令列傳入路徑則直接使用，否則在執行後提示用戶輸入路徑
    if len(sys.argv) > 1 and sys.argv[1].strip():
        chosen = Path(sys.argv[1].strip('"').strip("'").strip())
        if not chosen.exists() or not chosen.is_file():
            print(f"[-] 命令列指定的影片不存在: {chosen}")
            chosen = prompt_video_path()
    else:
        chosen = prompt_video_path()

    if chosen is None:
        print("[*] 已取消診斷並離開。")
        return

    if str(chosen) != "" and chosen.exists():
        records_to_diag = [extract_external_33d_record(chosen)]
    else:
        print("[*] 未指定外部影片，改用資料集代表樣本進行示範診斷...")
        sample_indices = [0, len(dataset) // 2, max(0, len(dataset) - 1)]
        records_to_diag = [dataset[i] for i in sample_indices if i < len(dataset)]

    for rec in records_to_diag:
        diag = diagnose_single_video_33d(rec, model, norm_mean, norm_std, benchmarks, device)
        print("\n" + "-" * 88)
        print(f"🎬 影片: [{diag['account_id']}] {diag['video_id']} ({diag['duration']}s)")
        print(f"📊 33 維加權綜合得分: {diag['weighted_total_score']:.1f}/100 | AI 爆款機率: {diag['good_prob']:.1f}% | 預測: {diag['pred_tier']}")
        for i, adv in enumerate(diag["advice_list"], 1):
            print(f"  {i}. {adv}")

        out_png = REPORT_DIR / f"diagnosis_33d_{diag['account_id']}_{diag['video_id']}.png"
        fig, _ = build_diagnostic_figure(diag, benchmarks)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 診斷圖表已儲存: {out_png}")


if __name__ == "__main__":
    main()
