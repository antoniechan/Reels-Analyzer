#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze & Visualize Temporal Attention Pooling Weights (_vfr)
---------------------------------------------------------------
核心功能：
1. 讀取同層資料夾 `transformer_engine_vfr/` 下的 `best_reels_transformer_vfr.pth`、
   `transformer_norm_stats_vfr.json` 與 `transformer_multimodal_dataset_vfr.pt`。
2. 輸出至同層子資料夾 `transformer_engine_vfr/attention_analysis_vfr/` 的 `*_vfr` 檔案：
   - `pool_weights_summary_vfr.csv`
   - `pool_weights_temporal_distribution_vfr.png`
   絕不覆蓋原有檔案。
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

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRANSFORMER_DIR

SUFFIX = "_vfr"
ENGINE_DIR = Path(TRANSFORMER_DIR) / f"transformer_engine{SUFFIX}"
MODEL_PATH = ENGINE_DIR / f"best_reels_transformer{SUFFIX}.pth"
STATS_PATH = ENGINE_DIR / f"transformer_norm_stats{SUFFIX}.json"
DATA_PATH = ENGINE_DIR / f"transformer_multimodal_dataset{SUFFIX}.pt"
OUT_DIR = ENGINE_DIR / f"attention_analysis{SUFFIX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights


class ReelsDiagnosticTransformerAttnPool(nn.Module):
    def __init__(self, vis_dim=8, geom_dim=20, aud_dim=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=3, dropout=0.30):
        super().__init__()
        self.vis_dim, self.geom_dim, self.aud_dim = vis_dim, geom_dim, aud_dim
        self.vis_proj = nn.Linear(vis_dim, d_model)
        self.geom_proj = nn.Linear(geom_dim, d_model)
        self.aud_proj = nn.Linear(aud_dim, d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=1000)
        self.norm_input = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DiagnosticTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionPooling(d_model=d_model, hidden_dim=32, dropout=dropout * 0.6)
        self.classifier = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, num_classes))
        self.frame_diagnostic_head = nn.Sequential(nn.Linear(d_model, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x_series: torch.Tensor, mask: torch.Tensor = None):
        h_vis = F.gelu(self.vis_proj(x_series[:, :, :self.vis_dim]))
        h_geom = F.gelu(self.geom_proj(x_series[:, :, self.vis_dim:self.vis_dim + self.geom_dim]))
        h_aud = F.gelu(self.aud_proj(x_series[:, :, self.vis_dim + self.geom_dim:self.vis_dim + self.geom_dim + self.aud_dim]))
        hidden = self.input_dropout(self.pos_encoder(self.norm_input(self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1)))))
        last_attn = None
        for layer in self.layers:
            hidden, last_attn = layer(hidden, src_key_padding_mask=mask)
        frame_risks = self.frame_diagnostic_head(hidden).squeeze(-1)
        pooled, pool_weights = self.attn_pool(hidden, padding_mask=mask)
        return self.classifier(pooled), frame_risks, last_attn, pool_weights


def resample_weights_to_bins(w: np.ndarray, num_bins: int = 20) -> np.ndarray:
    if len(w) == 0:
        return np.zeros(num_bins, dtype=np.float32)
    interp_w = np.interp(np.linspace(0.0, 1.0, num_bins), np.linspace(0.0, 1.0, len(w)), w)
    s = interp_w.sum()
    return interp_w / s if s > 0 else np.full(num_bins, 1.0 / num_bins)


def main():
    for label, p in [("模型權重", MODEL_PATH), ("標準化參數", STATS_PATH), ("資料集", DATA_PATH)]:
        if not p.exists():
            print(f"[-] 找不到{label}: {p}")
            return

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReelsDiagnosticTransformerAttnPool(
        vis_dim=int(stats.get("vis_dim", 8)),
        geom_dim=int(stats.get("geom_dim", 20)),
        aud_dim=int(stats.get("aud_dim", 5)),
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    raw_data = torch.load(DATA_PATH, weights_only=False)
    set_seed(42)
    indices = list(range(len(raw_data)))
    random.shuffle(indices)
    val_set = set(indices[int(len(raw_data) * 0.8):])

    tier_names = {0: "Bad", 1: "Neutral", 2: "Good"}
    sec_weights_by_tier = {0: {s: [] for s in range(30)}, 1: {s: [] for s in range(30)}, 2: {s: [] for s in range(30)}}
    sec_lift_all = {s: [] for s in range(30)}
    binned_by_tier = {0: [], 1: [], 2: []}
    rows = []

    with torch.no_grad():
        for idx, item in enumerate(raw_data):
            raw_ts = item["time_series"][:120]
            T = int(raw_ts.size(0))
            if T < 3:
                continue
            ts_norm = ((raw_ts - norm_mean) / norm_std).unsqueeze(0).to(device)
            logits, _, _, pool_weights = model(ts_norm, mask=None)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            w = pool_weights.squeeze(0).cpu().numpy()
            lift = w / (1.0 / T)
            true_label = int(item["tier_label"])
            is_val = idx in val_set
            top_idx = np.argsort(w)[::-1][:3]
            ent = -np.sum(w * np.log(np.clip(w, 1e-12, 1.0))) / max(np.log(T), 1e-8)

            rows.append({
                "criterion": f"vfr{SUFFIX}",
                "split": "val" if is_val else "train",
                "account_id": item.get("account_id", ""),
                "video_id": item.get("video_id", ""),
                "duration_sec": T,
                "true_tier": tier_names[true_label],
                "pred_tier": tier_names[int(np.argmax(probs))],
                "prob_good": round(float(probs[2]), 4),
                "hook_3s_weight_pct": round(float(w[:min(3, T)].sum()) * 100, 2),
                "hook_3s_lift": round(float(np.mean(lift[:min(3, T)])), 3),
                "outro_3s_weight_pct": round(float(w[max(0, T - 3):].sum()) * 100, 2),
                "peak_second_1idx": int(top_idx[0]) + 1,
                "peak_lift": round(float(lift[top_idx[0]]), 3),
                "norm_entropy": round(float(ent), 4),
                "top_3_seconds": ", ".join([f"{int(s)+1}s({w[s]*100:.1f}%)" for s in top_idx]),
            })
            if is_val:
                for s in range(min(T, 30)):
                    sec_weights_by_tier[true_label][s].append(float(w[s]))
                    sec_lift_all[s].append(float(lift[s]))
                binned_by_tier[true_label].append(resample_weights_to_bins(w, 20))

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"pool_weights_summary{SUFFIX}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=150)
    fig.patch.set_facecolor("white")

    secs_25 = np.arange(1, 26)
    lifts_25 = [np.mean(sec_lift_all[s]) if sec_lift_all[s] else 0.0 for s in range(25)]
    axes[0].bar(secs_25, lifts_25, color=["#d9534f" if v >= 1.05 else "#4a90d9" for v in lifts_25], alpha=0.88)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="Uniform (1.0x)")
    axes[0].set_title(f"Val Set Attention Lift (1s-25s) [{SUFFIX}]")
    axes[0].set_xlabel("Video Second (1-indexed)")
    axes[0].set_ylabel("Lift (Weight / (1/T))")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    secs_20 = np.arange(1, 21)
    for t_id, t_name, col, mk in [(2, "Good", "#2ca02c", "o"), (1, "Neutral", "#7f7f7f", "s"), (0, "Bad", "#d62728", "^")]:
        y_vals = [np.mean(sec_weights_by_tier[t_id][s]) * 100 if sec_weights_by_tier[t_id][s] else np.nan for s in range(20)]
        axes[1].plot(secs_20, y_vals, label=t_name, color=col, marker=mk, markersize=4, linewidth=1.8)
    axes[1].set_title(f"Mean Weight (%) by Tier (1s-20s) [{SUFFIX}]")
    axes[1].set_xlabel("Video Second (1-indexed)")
    axes[1].set_ylabel("Weight (%)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    prog_x = np.linspace(5, 100, 20)
    for t_id, t_name, col, mk in [(2, "Good", "#2ca02c", "o"), (1, "Neutral", "#7f7f7f", "s"), (0, "Bad", "#d62728", "^")]:
        if binned_by_tier[t_id]:
            axes[2].plot(prog_x, np.vstack(binned_by_tier[t_id]).mean(axis=0) * 100, label=t_name, color=col, marker=mk, markersize=4, linewidth=1.8)
    axes[2].axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="Uniform (5.0%)")
    axes[2].set_title(f"Normalized Timeline (0%-100%) [{SUFFIX}]")
    axes[2].set_xlabel("Progress (%)")
    axes[2].set_ylabel("Weight Share (%)")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / f"pool_weights_temporal_distribution{SUFFIX}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print(f"[+] 完成 ({SUFFIX})！已儲存: {csv_path} 與 {fig_path}")


if __name__ == "__main__":
    main()
