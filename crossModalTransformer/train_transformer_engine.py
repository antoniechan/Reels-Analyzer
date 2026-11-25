#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Train Reels Multimodal Temporal Transformer Engine (NewReels + Compact 8-Dim)
----------------------------------------------------------------------------------------
功能：
1. 載入 transformer_multimodal_dataset.pt（輸入維度 8：4 視覺 + 2 QC 形變 + 2 音訊）。
2. 進行特徵標準化與動態 Padding Mask。
3. 訓練具備 Temporal Self-Attention 的多模態 Transformer 模型並儲存權重與正規化參數。
"""

import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import TRANSFORMER_DIR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

DATASET_PATH = Path(TRANSFORMER_DIR) / "transformer_engine" / "transformer_multimodal_dataset.pt"
OUTPUT_DIR = Path(TRANSFORMER_DIR) / "transformer_engine"
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
        return x + self.pe[:, :x.size(1)]


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
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src, attn_weights


class ReelsDiagnosticTransformer(nn.Module):
    def __init__(
        self,
        vis_dim: int = 4,
        geom_dim: int = 2,
        aud_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
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
        x_vis = x_series[:, :, 0:4]
        x_geom = x_series[:, :, 4:6]
        x_aud = x_series[:, :, 6:8]

        h_vis = F.gelu(self.vis_proj(x_vis))
        h_geom = F.gelu(self.geom_proj(x_geom))
        h_aud = F.gelu(self.aud_proj(x_aud))

        h_fused = self.norm_input(self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1)))
        hidden = self.pos_encoder(h_fused)

        last_attn_weights = None
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
    def __init__(self, records, norm_mean=None, norm_std=None):
        self.records = records
        if norm_mean is None or norm_std is None:
            all_frames = torch.cat([r["time_series"] for r in records], dim=0)
            self.norm_mean = all_frames.mean(dim=0)
            self.norm_std = all_frames.std(dim=0).clamp(min=1e-5)
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        raw_ts = item["time_series"][:120]
        ts_normalized = (raw_ts - self.norm_mean) / self.norm_std
        return {
            "series": ts_normalized,
            "tier_label": item["tier_label"],
        }


def collate_fn(batch):
    max_len = max(b["series"].size(0) for b in batch)
    batch_size = len(batch)
    feat_dim = batch[0]["series"].size(1)

    padded_series = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["series"].size(0)
        padded_series[i, :seq_len] = b["series"]
        mask[i, :seq_len] = False
        labels[i] = b["tier_label"]

    return {"series": padded_series, "mask": mask, "labels": labels}


def main():
    if not DATASET_PATH.exists():
        print(f"[-] 找不到素材檔: {DATASET_PATH}")
        return

    raw_data = torch.load(DATASET_PATH, weights_only=False)
    if not raw_data:
        print("[-] 素材庫為空 (0 支影片)，請先執行 prepare_transformer_dataset.py。")
        return

    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_ds = ReelsPaddedDataset(raw_data[:split_idx])
    val_ds = ReelsPaddedDataset(raw_data[split_idx:], norm_mean=train_ds.norm_mean, norm_std=train_ds.norm_std)

    with open(OUTPUT_DIR / "transformer_norm_stats.json", "w", encoding="utf-8") as f:
        json.dump({"mean": train_ds.norm_mean.tolist(), "std": train_ds.norm_std.tolist()}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = ReelsDiagnosticTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / "best_reels_transformer.pth"

    for epoch in range(1, 31):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
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
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

        scheduler.step()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                series = batch["series"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)
                logits, _, _ = model(series, mask=mask)
                v_correct += (logits.argmax(dim=-1) == labels).sum().item()
                v_total += len(labels)

        val_acc = v_correct / max(1, v_total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:02d}/30] | Train Loss: {total_loss / max(1, total):.4f} "
                f"Acc: {correct / max(1, total) * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%"
            )

    print(f"[+] 訓練完成！最佳驗證準確率: {best_val_acc * 100:.2f}% -> {best_model_path}")


if __name__ == "__main__":
    main()