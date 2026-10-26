#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Train Reels Multimodal Temporal Transformer Engine
-----------------------------------------------------------
功能：
1. 載入先前打包好的 transformer_multimodal_dataset.pt。
2. 進行特徵標準化與動態 Padding Mask。
3. 訓練具備 Temporal Self-Attention 的多模態 Transformer 模型。
4. 儲存最佳模型權重 (best_transformer_model.pth) 與特徵正規化參數 (norm_stats.json)。
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
from config import TRANSFORMER_DIR


# 設定隨機種子以確保實驗可重複性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

DATASET_PATH = TRANSFORMER_DIR / "transformer_engine" / "transformer_multimodal_dataset.pt"
OUTPUT_DIR = TRANSFORMER_DIR / "transformer_engine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 位置編碼層 (Positional Encoding)
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
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ---------------------------------------------------------------------------
# 2. 自定義多頭注意力 Transformer Encoder (支援提取 Attention Weights)
# ---------------------------------------------------------------------------
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
        # 提取注意力矩陣 attn_weights: (Batch, Seq_Len, Seq_Len)
        attn_out, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            average_attn_weights=True
        )
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src, attn_weights


# ---------------------------------------------------------------------------
# 3. 核心模型：多模態時序 Transformer (ReelsDiagnosticTransformer)
# ---------------------------------------------------------------------------
class ReelsDiagnosticTransformer(nn.Module):
    def __init__(
            self,
            vis_dim: int = 4,  # [colorfulness, face_rms, skin_warmth, fg_bg_contrast]
            geom_dim: int = 2,  # [key_mus, all_mus]
            aud_dim: int = 2,  # [audio_energy, semantic_score]
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            num_classes: int = 3,  # 0: Bad, 1: Neutral, 2: Good
            dropout: float = 0.15,
    ):
        super().__init__()

        # 各模態獨立投影
        self.vis_proj = nn.Linear(vis_dim, d_model)
        self.geom_proj = nn.Linear(geom_dim, d_model)
        self.aud_proj = nn.Linear(aud_dim, d_model)

        # 融合層
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=1000)
        self.norm_input = nn.LayerNorm(d_model)

        # 多層 Encoder
        self.layers = nn.ModuleList([
            DiagnosticTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 1. 全片分類頭 (Global Classifier: Good/Neutral/Bad)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

        # 2. 逐秒流失風險/診斷頭 (Frame-level Risk Head: 輸出每秒的流失風險指標)
        self.frame_diagnostic_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_series: torch.Tensor, mask: torch.Tensor = None):
        """
        x_series: (Batch, T, 8)
        mask: (Batch, T) - True 代表無效 Padding 秒數
        """
        # 拆分模態子通道
        x_vis = x_series[:, :, 0:4]
        x_geom = x_series[:, :, 4:6]
        x_aud = x_series[:, :, 6:8]

        # 獨立投影並激活
        h_vis = F.gelu(self.vis_proj(x_vis))
        h_geom = F.gelu(self.geom_proj(x_geom))
        h_aud = F.gelu(self.aud_proj(x_aud))

        # 跨模態融合
        h_fused = self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1))
        h_fused = self.norm_input(h_fused)
        h_tokens = self.pos_encoder(h_fused)

        # 逐層 Attention 傳遞並記錄最後一層注意力
        last_attn_weights = None
        hidden = h_tokens
        for layer in self.layers:
            hidden, last_attn_weights = layer(hidden, src_key_padding_mask=mask)

        # 計算逐秒流失風險分 (Batch, T)
        frame_risks = self.frame_diagnostic_head(hidden).squeeze(-1)

        # 全域時間池化 (Masked Global Average Pooling)
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
            pooled = (hidden * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)

        # 全域評級預測
        logits = self.classifier(pooled)

        return logits, frame_risks, last_attn_weights


# ---------------------------------------------------------------------------
# 4. 資料集載入、特徵標準化與動態 Collate
# ---------------------------------------------------------------------------
class ReelsPaddedDataset(Dataset):
    def __init__(self, records, norm_mean=None, norm_std=None):
        self.records = records

        # 若無提供標準化參數，則從訓練集中統計 (Channel-wise Z-score)
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
        # 限制最長分析前 120 秒 (涵蓋 99% 正常 Reels 長度，大幅加速 CPU 訓練)
        raw_ts = item["time_series"][:120]
        ts_normalized = (raw_ts - self.norm_mean) / self.norm_std
        return {
            "series": ts_normalized,
            "tier_label": item["tier_label"],
            "video_id": item["video_id"],
            "duration": ts_normalized.size(0)
        }


def collate_fn(batch):
    max_len = max([b["series"].size(0) for b in batch])
    batch_size = len(batch)
    feat_dim = batch[0]["series"].size(1)

    padded_series = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["series"].size(0)
        padded_series[i, :seq_len] = b["series"]
        mask[i, :seq_len] = False  # False 表示真實有效秒數
        labels[i] = b["tier_label"]

    return {
        "series": padded_series,
        "mask": mask,
        "labels": labels
    }


# ---------------------------------------------------------------------------
# 5. 訓練與驗證循環主程序
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【啟動 Reels 多模態時序 Transformer 診斷模型訓練】")
    print("=" * 80)

    if not DATASET_PATH.exists():
        print(f"[-] 找不到素材檔: {DATASET_PATH}，請先執行素材準備腳本。")
        return

    # 1. 讀取打包檔案
    print(f"[*] 正在載入素材庫: {DATASET_PATH} ...")
    raw_data = torch.load(DATASET_PATH, weights_only=False)
    print(f"[*] 成功載入 {len(raw_data)} 支影片的時序特徵。")

    # 2. 劃分訓練集 (80%) 與驗證集 (20%)
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_records = raw_data[:split_idx]
    val_records = raw_data[split_idx:]

    train_ds = ReelsPaddedDataset(train_records)
    val_ds = ReelsPaddedDataset(val_records, norm_mean=train_ds.norm_mean, norm_std=train_ds.norm_std)

    # 儲存特徵標準化均值與標準差 (供未來單片診斷時使用)
    stats_dict = {
        "mean": train_ds.norm_mean.tolist(),
        "std": train_ds.norm_std.tolist()
    }
    stats_path = OUTPUT_DIR / "transformer_norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[+] 特徵正規化參數已儲存至: {stats_path}")

    # DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用運算裝置: {device}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 3. 初始化模型
    model = ReelsDiagnosticTransformer(
        vis_dim=4,
        geom_dim=2,
        aud_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=3,
        dropout=0.15
    ).to(device)

    # 類別權重平衡 (Bad:0, Neutral:1, Good:2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    epochs = 30
    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / "best_reels_transformer.pth"

    print("\n[*] 開始進入 Epoch 迭代訓練 (共 30 輪)...")
    print("-" * 80)

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

            # 梯度裁剪防止震盪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # 驗證階段
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

        val_acc = v_correct / v_total
        val_loss = v_loss / v_total

        # 若驗證表現更好則儲存權重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_mark = "★ (Saved Best)"
        else:
            saved_mark = ""

        if epoch % 5 == 0 or epoch == 1 or saved_mark != "":
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc * 100:.2f}% {saved_mark}")

    print("-" * 80)
    print(f"[+] 訓練全數完成！最佳驗證準確率: {best_val_acc * 100:.2f}%")
    print(f"[+] 最佳模型權重已儲存至: {best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
