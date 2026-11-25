#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze & Visualize Temporal Attention Pooling Weights (pool_weights)
----------------------------------------------------------------------
功能：
1. 載入 `best_reels_transformer_attn_pool.pth` 與 `transformer_norm_stats.json`。
2. 還原訓練時的隨機種子 (seed=42) 與 80/20 切分，分別評估 Validation Set (N=600) 與全量資料集 (N=3000)。
3. 萃取每支影片的逐秒 `pool_weights`，統計：
   - 黃金前 3 秒 (0~2s)、前 5 秒 (0~4s)、中段、最後 3 秒的注意力佔比與相對於均勻分佈 (1/T) 的倍率。
   - Bad / Neutral / Good 三類影片在絕對秒數與歸一化進度 (0%~100%) 上的注意力差異。
4. 輸出圖表 `pool_weights_temporal_distribution.png` 與明細表 `pool_weights_summary.csv`。
"""

import os
import sys
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import TRANSFORMER_DIR
    TR_BASE = Path(TRANSFORMER_DIR)
except ImportError:
    TR_BASE = CURRENT_DIR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. 模型定義 (與 train_transformer_engine_attn_pool.py 完全一致)
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
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.3):
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


class ReelsDiagnosticTransformerAttnPool(nn.Module):
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
        dropout: float = 0.30,
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

        h_fused = self.norm_input(self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1)))
        hidden = self.input_dropout(self.pos_encoder(h_fused))

        last_attn_weights = None
        for layer in self.layers:
            hidden, last_attn_weights = layer(hidden, src_key_padding_mask=mask)

        frame_risks = self.frame_diagnostic_head(hidden).squeeze(-1)
        pooled, pool_weights = self.attn_pool(hidden, padding_mask=mask)
        logits = self.classifier(pooled)

        return logits, frame_risks, last_attn_weights, pool_weights


# ---------------------------------------------------------------------------
# 2. 推論與分析主程序
# ---------------------------------------------------------------------------
def resolve_paths():
    model_candidates = [
        TR_BASE / "transformer_engine_attn_pool" / "best_reels_transformer_attn_pool.pth",
        CURRENT_DIR / "transformer_engine_attn_pool" / "best_reels_transformer_attn_pool.pth",
        CURRENT_DIR / "best_reels_transformer_attn_pool.pth",
    ]
    stats_candidates = [
        TR_BASE / "transformer_engine_attn_pool" / "transformer_norm_stats.json",
        CURRENT_DIR / "transformer_engine_attn_pool" / "transformer_norm_stats.json",
        CURRENT_DIR / "transformer_norm_stats.json",
    ]
    data_candidates = [
        TR_BASE / "transformer_engine_all_features" / "transformer_multimodal_dataset_all_features.pt",
        CURRENT_DIR / "transformer_engine_all_features" / "transformer_multimodal_dataset_all_features.pt",
        TR_BASE / "transformer_engine" / "transformer_multimodal_dataset.pt",
        CURRENT_DIR / "transformer_engine" / "transformer_multimodal_dataset.pt",
    ]

    model_path = next((p for p in model_candidates if p.exists()), model_candidates[0])
    stats_path = next((p for p in stats_candidates if p.exists()), stats_candidates[0])
    data_path = next((p for p in data_candidates if p.exists()), data_candidates[0])
    return model_path, stats_path, data_path


def resample_weights_to_bins(w: np.ndarray, num_bins: int = 20) -> np.ndarray:
    """將任意長度的注意力權重序列重採樣到固定比例區間 (0%~100%)，並保持總和為 1"""
    T = len(w)
    if T == 0:
        return np.zeros(num_bins, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, num_bins)
    interp_w = np.interp(x_new, x_old, w)
    s = interp_w.sum()
    return interp_w / s if s > 0 else np.full(num_bins, 1.0 / num_bins)


def main():
    print("=" * 88)
    print("【Attention Pooling 時序權重 (pool_weights) 解析與視覺化分析】")
    print("=" * 88)

    model_path, stats_path, data_path = resolve_paths()
    for label, p in [("模型權重", model_path), ("正規化參數", stats_path), ("時序資料集", data_path)]:
        if not p.exists():
            print(f"[-] 找不到{label}: {p}")
            return

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32)
    vis_dim = int(stats.get("vis_dim", 8))
    geom_dim = int(stats.get("geom_dim", 20))
    aud_dim = int(stats.get("aud_dim", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReelsDiagnosticTransformerAttnPool(
        vis_dim=vis_dim,
        geom_dim=geom_dim,
        aud_dim=aud_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=3,
        dropout=0.30,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"[*] 載入資料集: {data_path}")
    raw_data = torch.load(data_path, weights_only=False)

    # 還原訓練時的 80/20 切分標記 (seed=42)
    set_seed(42)
    indices = list(range(len(raw_data)))
    random.shuffle(indices)
    split_idx = int(len(raw_data) * 0.8)
    val_index_set = set(indices[split_idx:])

    tier_names = {0: "Bad", 1: "Neutral", 2: "Good"}
    records_summary = []

    # 收集前 30 秒的逐秒權重與均勻倍率 (w_t / (1/T))
    max_sec_plot = 30
    sec_weights_by_tier = {0: {s: [] for s in range(max_sec_plot)},
                           1: {s: [] for s in range(max_sec_plot)},
                           2: {s: [] for s in range(max_sec_plot)}}
    sec_lift_all = {s: [] for s in range(max_sec_plot)}
    binned_by_tier = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for idx, item in enumerate(raw_data):
            raw_ts = item["time_series"][:120]
            T = int(raw_ts.size(0))
            if T < 3:
                continue

            ts_norm = ((raw_ts - norm_mean) / norm_std).unsqueeze(0).to(device)  # (1, T, D)
            logits, frame_risks, _, pool_weights = model(ts_norm, mask=None)

            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_label = int(np.argmax(probs))
            true_label = int(item["tier_label"])
            w = pool_weights.squeeze(0).cpu().numpy()  # (T,), sum == 1.0
            is_val = idx in val_index_set

            uniform_w = 1.0 / T
            lift = w / uniform_w  # >1 代表比平均分配更受關注

            hook_3s_share = float(w[:min(3, T)].sum())
            hook_5s_share = float(w[:min(5, T)].sum())
            outro_3s_share = float(w[max(0, T - 3):].sum())
            mid_share = float(max(0.0, 1.0 - hook_3s_share - outro_3s_share)) if T > 6 else 0.0

            # Top-3 最受關注秒數 (1-indexed 秒數更直觀)
            top_indices = np.argsort(w)[::-1][:3]
            top_secs_str = ", ".join([f"{int(s)+1}s({w[s]*100:.1f}%)" for s in top_indices])

            # 正規化注意力熵 (0=完全集中單一秒, 1=完全平均分散)
            ent = -np.sum(w * np.log(np.clip(w, 1e-12, 1.0))) / max(np.log(T), 1e-8)

            records_summary.append({
                "split": "val" if is_val else "train",
                "account_id": item.get("account_id", ""),
                "video_id": item.get("video_id", ""),
                "duration_sec": T,
                "true_tier": tier_names[true_label],
                "pred_tier": tier_names[pred_label],
                "prob_good": round(float(probs[2]), 4),
                "hook_3s_weight_pct": round(hook_3s_share * 100, 2),
                "hook_3s_lift": round(float(np.mean(lift[:min(3, T)])), 3),
                "hook_5s_weight_pct": round(hook_5s_share * 100, 2),
                "outro_3s_weight_pct": round(outro_3s_share * 100, 2),
                "outro_3s_lift": round(float(np.mean(lift[max(0, T - 3):])), 3),
                "peak_second_1idx": int(top_indices[0]) + 1,
                "peak_weight_pct": round(float(w[top_indices[0]]) * 100, 2),
                "peak_lift": round(float(lift[top_indices[0]]), 3),
                "norm_entropy": round(float(ent), 4),
                "top_3_seconds": top_secs_str,
            })

            if is_val:
                for s in range(min(T, max_sec_plot)):
                    sec_weights_by_tier[true_label][s].append(float(w[s]))
                    sec_lift_all[s].append(float(lift[s]))
                binned_by_tier[true_label].append(resample_weights_to_bins(w, num_bins=20))

    df = pd.DataFrame(records_summary)
    val_df = df[df["split"] == "val"]

    out_dir = model_path.parent / "attention_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pool_weights_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "-" * 88)
    print(f"【1. 驗證集 (N = {len(val_df)}) 各時間區段注意力聚焦度 (Lift = 實際權重 / 均勻基準 1/T)】")
    print("-" * 88)
    for tier in ["Good", "Neutral", "Bad"]:
        sub = val_df[val_df["true_tier"] == tier]
        print(
            f"  [{tier:7s} | n={len(sub):3d}] "
            f"前3秒佔比: {sub['hook_3s_weight_pct'].mean():5.2f}% (Lift={sub['hook_3s_lift'].mean():.2f}x) | "
            f"前5秒佔比: {sub['hook_5s_weight_pct'].mean():5.2f}% | "
            f"最後3秒佔比: {sub['outro_3s_weight_pct'].mean():5.2f}% (Lift={sub['outro_3s_lift'].mean():.2f}x) | "
            f"熵值: {sub['norm_entropy'].mean():.3f}"
        )

    print("\n" + "-" * 88)
    print("【2. 驗證集前 15 秒逐秒平均注意力倍率 (Lift > 1.0 代表模型特別注重該秒)】")
    print("-" * 88)
    for s in range(15):
        vals = sec_lift_all[s]
        if vals:
            mean_lift = np.mean(vals)
            bar = "█" * int(round(mean_lift * 15))
            print(f"  第 {s+1:02d} 秒 ({s}~{s+1}s): Lift = {mean_lift:.3f}x  {bar}")

    print("\n" + "-" * 88)
    print("【3. 驗證集最具代表性的 Good 影片 Top 8 關注秒數明細】")
    print("-" * 88)
    top_good = val_df[val_df["true_tier"] == "Good"].sort_values("prob_good", ascending=False).head(8)
    for _, row in top_good.iterrows():
        print(
            f"  @{row['account_id']} | {row['video_id']} (片長 {row['duration_sec']}s, P(Good)={row['prob_good']:.2f}) "
            f"-> Top秒數: {row['top_3_seconds']}"
        )

    # -----------------------------------------------------------------------
    # 3. 繪製 3 合 1 視覺化圖表
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # 子圖 A: 前 25 秒平均注意力倍率 (Lift)
    secs_x = np.arange(1, 26)
    lifts_y = [np.mean(sec_lift_all[s]) if sec_lift_all[s] else 0.0 for s in range(25)]
    colors_bar = ["#d9534f" if v >= 1.05 else "#4a90d9" for v in lifts_y]
    axes[0].bar(secs_x, lifts_y, color=colors_bar, alpha=0.88)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="Uniform Baseline (1.0x)")
    axes[0].set_title("Val Set: Attention Lift by Second (1s–25s)")
    axes[0].set_xlabel("Video Second (1-indexed)")
    axes[0].set_ylabel("Attention Lift (Weight / (1/T))")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # 子圖 B: Bad / Neutral / Good 前 20 秒絕對權重曲線
    tier_styles = [
        (2, "Good", "#2ca02c", "o"),
        (1, "Neutral", "#7f7f7f", "s"),
        (0, "Bad", "#d62728", "^"),
    ]
    secs_20 = np.arange(1, 21)
    for t_id, t_name, col, mk in tier_styles:
        y_vals = [
            np.mean(sec_weights_by_tier[t_id][s]) * 100 if sec_weights_by_tier[t_id][s] else np.nan
            for s in range(20)
        ]
        axes[1].plot(secs_20, y_vals, label=t_name, color=col, marker=mk, markersize=4, linewidth=1.8)
    axes[1].set_title("Mean Attention Weight (%) by Tier (1s–20s)")
    axes[1].set_xlabel("Video Second (1-indexed)")
    axes[1].set_ylabel("Mean Attention Weight (%)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # 子圖 C: 歸一化影片進度 (0% -> 100%) 注意力分佈
    progress_x = np.linspace(5, 100, 20)
    for t_id, t_name, col, mk in tier_styles:
        if binned_by_tier[t_id]:
            mat = np.vstack(binned_by_tier[t_id])
            mean_curve = mat.mean(axis=0) * 100
            axes[2].plot(progress_x, mean_curve, label=t_name, color=col, marker=mk, markersize=4, linewidth=1.8)
    axes[2].axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="Uniform (5.0% / bin)")
    axes[2].set_title("Normalized Timeline Attention (0%–100% Duration)")
    axes[2].set_xlabel("Video Progress (%)")
    axes[2].set_ylabel("Share of Attention per 5% Segment (%)")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "pool_weights_temporal_distribution.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 88)
    print(f"[+] 明細 CSV 已儲存至: {csv_path}")
    print(f"[+] 視覺化圖表已儲存至: {fig_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()