#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (v4 - Full 33-Dim All Modality Features): Train Reels Multimodal Temporal Transformer Engine
-----------------------------------------------------------------------------------------------------
核心升級：
1. 預設納入全部 33 維特徵（`vis_dim=8, geom_dim=20, aud_dim=5`），並支援從資料集的 `modality_dims` 或張量維度自動適配。
2. 自動相容 `TRANSFORMER_DIR / "transformer_engine"` 與舊版 `CURRENT_DIR / "transformer_engine"` 路徑。
3. 訓練具備 Temporal Self-Attention 的多模態 Transformer 模型，儲存最佳權重 (`best_reels_transformer.pth`)、
   標準化參數 (`transformer_norm_stats.json`) 與訓練歷程 (`training_history.json`)。
"""

import os
import sys
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRANSFORMER_DIR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

CANDIDATE_DATASET_PATHS = [
    Path(TRANSFORMER_DIR) / "transformer_engine_all_features" / "transformer_multimodal_dataset_all_features.pt",
    CURRENT_DIR / "transformer_engine_all_features" / "transformer_multimodal_dataset_all_features.pt",
]
DATASET_PATH = next((p for p in CANDIDATE_DATASET_PATHS if p.exists()), CANDIDATE_DATASET_PATHS[0])
OUTPUT_DIR = Path(TRANSFORMER_DIR) / "transformer_engine_all_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class DiagnosticTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
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
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src, attn_weights


class ReelsDiagnosticTransformer(nn.Module):
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
        dropout: float = 0.15,
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

        self.layers = nn.ModuleList([
            DiagnosticTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

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

        h_fused = self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1))
        h_fused = self.norm_input(h_fused)
        h_tokens = self.pos_encoder(h_fused)

        last_attn_weights = None
        hidden = h_tokens
        for layer in self.layers:
            hidden, last_attn_weights = layer(hidden, src_key_padding_mask=mask)

        frame_risks = self.frame_diagnostic_head(hidden).squeeze(-1)

        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
            pooled = (hidden * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.classifier(pooled)
        return logits, frame_risks, last_attn_weights


class ReelsPaddedDataset(Dataset):
    def __init__(self, records, norm_mean=None, norm_std=None, max_len: int = 120):
        self.records = records
        self.max_len = max_len

        if norm_mean is None or norm_std is None:
            all_frames = torch.cat([r["time_series"][:self.max_len] for r in records], dim=0)
            self.norm_mean = all_frames.mean(dim=0)
            self.norm_std = all_frames.std(dim=0).clamp(min=1e-5)
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        raw_ts = item["time_series"][:self.max_len]
        ts_normalized = (raw_ts - self.norm_mean) / self.norm_std
        seq_len = ts_normalized.size(0)

        f_mask = item.get("face_valid_mask")
        if f_mask is not None and len(f_mask) >= seq_len:
            face_valid = f_mask[:seq_len].to(torch.bool)
        else:
            face_valid = torch.ones(seq_len, dtype=torch.bool)

        return {
            "series": ts_normalized,
            "face_valid": face_valid,
            "tier_label": int(item["tier_label"]),
            "quality_score": float(item.get("quality_score", 50.0)),
            "shares": float(item.get("shares", 0.0)),
            "SRV": float(item.get("SRV", 0.0)),
            "video_id": item.get("video_id", ""),
        }


def collate_fn(batch):
    max_len = max(b["series"].size(0) for b in batch)
    batch_size = len(batch)
    feat_dim = batch[0]["series"].size(1)

    padded_series = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    padded_face_valid = torch.zeros(batch_size, max_len, dtype=torch.bool)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["series"].size(0)
        padded_series[i, :seq_len] = b["series"]
        padded_face_valid[i, :seq_len] = b["face_valid"]
        mask[i, :seq_len] = False
        labels[i] = b["tier_label"]

    return {
        "series": padded_series,
        "face_valid": padded_face_valid,
        "mask": mask,
        "labels": labels,
    }


def main():
    print("=" * 88)
    print("【啟動 Reels 多模態時序 Transformer 診斷模型訓練 v4 (全量 33 維：8 Visual + 20 ROI + 5 Audio)】")
    print("=" * 88)

    if not DATASET_PATH.exists():
        print(f"[-] 找不到素材檔: {DATASET_PATH}，請先執行 prepare_transformer_dataset_all_features。")
        return

    print(f"[*] 正在載入素材庫: {DATASET_PATH} ...")
    raw_data = torch.load(DATASET_PATH, weights_only=False)
    if not raw_data or len(raw_data) == 0:
        print("[-] 素材庫為空 (0 支影片)，請先確認資料打包腳本是否成功產生非空資料集。")
        return

    print(f"[*] 成功載入 {len(raw_data)} 支影片的時序特徵。")

    total_dim = int(raw_data[0]["time_series"].size(1))
    mod_dims = raw_data[0].get("modality_dims")
    if isinstance(mod_dims, dict) and sum(int(v) for v in mod_dims.values()) == total_dim:
        vis_dim = int(mod_dims["vis_dim"])
        geom_dim = int(mod_dims["geom_dim"])
        aud_dim = int(mod_dims["aud_dim"])
    elif total_dim == 33:
        vis_dim, geom_dim, aud_dim = 8, 20, 5
    elif total_dim == 26:
        vis_dim, geom_dim, aud_dim = 4, 20, 2
    else:
        vis_dim = min(8, total_dim)
        aud_dim = min(5, max(0, total_dim - vis_dim))
        geom_dim = max(1, total_dim - vis_dim - aud_dim)

    print(f"[*] 自動偵測特徵維度: 總維度={total_dim} (Visual={vis_dim}, ROI QC={geom_dim}, Audio={aud_dim})")

    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_records = raw_data[:split_idx]
    val_records = raw_data[split_idx:]

    train_ds = ReelsPaddedDataset(train_records)
    val_ds = ReelsPaddedDataset(val_records, norm_mean=train_ds.norm_mean, norm_std=train_ds.norm_std)

    stats_dict = {
        "mean": train_ds.norm_mean.tolist(),
        "std": train_ds.norm_std.tolist(),
        "vis_dim": vis_dim,
        "geom_dim": geom_dim,
        "aud_dim": aud_dim,
        "total_dim": total_dim,
        "feature_names": raw_data[0].get("feature_names", []),
        "num_train": len(train_records),
        "num_val": len(val_records),
    }
    stats_path = OUTPUT_DIR / "transformer_norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[+] 特徵正規化參數已儲存至: {stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用運算裝置: {device}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = ReelsDiagnosticTransformer(
        vis_dim=vis_dim,
        geom_dim=geom_dim,
        aud_dim=aud_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=3,
        dropout=0.15,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    epochs = 30
    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / "best_reels_transformer.pth"
    history = []

    print("\n[*] 開始進入 Epoch 迭代訓練 (共 30 輪)...")
    print("-" * 88)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            series = batch["series"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(series, mask=mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        model.eval()
        v_correct = 0
        v_total = 0
        v_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                series = batch["series"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _, _ = model(series, mask=mask)
                loss = criterion(logits, labels)
                v_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == labels).sum().item()
                v_total += len(labels)

        val_acc = v_correct / max(1, v_total)
        val_loss = v_loss / max(1, v_total)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_mark = "★ (Saved Best)"
        else:
            saved_mark = ""

        if epoch % 5 == 0 or epoch == 1 or saved_mark != "":
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc * 100:.2f}% {saved_mark}"
            )

    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 88)
    print(f"[+] 訓練全數完成！最佳驗證準確率: {best_val_acc * 100:.2f}%")
    print(f"[+] 最佳模型權重已儲存至: {best_model_path}")
    print(f"[+] 訓練歷程已儲存至: {history_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()