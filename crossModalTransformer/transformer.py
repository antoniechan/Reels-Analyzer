#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Time-Series Transformer for Reels Video Quality Diagnosis
---------------------------------------------------------------------
特點：
1. 支援不固定長度影片 (以 Pad Mask 遮蔽無效秒數)。
2. 跨模態時序自注意力機制 (Self-Attention over Time)。
3. 提供 get_attention_weights() 匯出逐秒診斷分數。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ---------------------------------------------------------------------------
# 1. 位置編碼 (Positional Encoding)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 120):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ---------------------------------------------------------------------------
# 2. 核心模型：多模態時序 Transformer (ReelsMultimodalTransformer)
# ---------------------------------------------------------------------------
class ReelsMultimodalTransformer(nn.Module):
    def __init__(
        self,
        vis_dim: int = 4,    # colorfulness, face_rms, skin_warmth, fg_bg_contrast
        geom_dim: int = 2,   # key_mus, all_mus (Beltrami QC)
        aud_dim: int = 2,    # acoustic_energy, semantic_score
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 各模態獨立投影至共同隱藏維度
        self.vis_proj = nn.Linear(vis_dim, d_model)
        self.geom_proj = nn.Linear(geom_dim, d_model)
        self.aud_proj = nn.Linear(aud_dim, d_model)

        # 模態融合融合層 (特徵拼接後投影回 d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=120)
        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer 編碼器層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 輸出預測頭 (MLP)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 預測連續品質得分
        )

    def forward(self, x_vis, x_geom, x_aud, mask=None):
        """
        x_vis:  (Batch, T, vis_dim)
        x_geom: (Batch, T, geom_dim)
        x_aud:  (Batch, T, aud_dim)
        mask:   (Batch, T) - True 代表該秒是 padding (無效秒數)
        """
        # 1. 模態投影
        h_vis = F.gelu(self.vis_proj(x_vis))
        h_geom = F.gelu(self.geom_proj(x_geom))
        h_aud = F.gelu(self.aud_proj(x_aud))

        # 2. 模態拼接融合
        h_fused = self.fusion(torch.cat([h_vis, h_geom, h_aud], dim=-1))
        h_fused = self.layer_norm(h_fused)

        # 3. 加上位置編碼
        h_tokens = self.pos_encoder(h_fused)

        # 4. 時序自注意力運算 (Temporal Self-Attention)
        encoded = self.transformer_encoder(h_tokens, src_key_padding_mask=mask)

        # 5. 時間維度聚合 (Masked Average Pooling)
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()  # (Batch, T, 1)
            pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)

        # 6. 回歸預測
        out = self.regressor(pooled).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# 3. 模態時序資料集建構器 (Dataset & Collate)
# ---------------------------------------------------------------------------
class ReelsTimeSeriesDataset(Dataset):
    """
    從整合後的 Master 資料庫中讀取各影片的逐秒時序陣列
    """
    def __init__(self, sample_records: list):
        self.samples = sample_records

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "vis": torch.tensor(item["vis_series"], dtype=torch.float32),      # (T, 4)
            "geom": torch.tensor(item["geom_series"], dtype=torch.float32),    # (T, 2)
            "aud": torch.tensor(item["aud_series"], dtype=torch.float32),      # (T, 2)
            "quality_score": torch.tensor(item["quality_score"], dtype=torch.float32),
            "video_stem": item.get("video_stem", "")
        }


def pad_collate_fn(batch):
    """
    動態對齊同一個 Batch 內不同長度 (秒數) 的影片，產生 Padding Mask
    """
    max_len = max([b["vis"].size(0) for b in batch])
    batch_size = len(batch)

    vis_dim = batch[0]["vis"].size(1)
    geom_dim = batch[0]["geom"].size(1)
    aud_dim = batch[0]["aud"].size(1)

    padded_vis = torch.zeros(batch_size, max_len, vis_dim)
    padded_geom = torch.zeros(batch_size, max_len, geom_dim)
    padded_aud = torch.zeros(batch_size, max_len, aud_dim)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # 預設全為 True (Masked)
    targets = torch.zeros(batch_size)

    for i, b in enumerate(batch):
        seq_len = b["vis"].size(0)
        padded_vis[i, :seq_len] = b["vis"]
        padded_geom[i, :seq_len] = b["geom"]
        padded_aud[i, :seq_len] = b["aud"]
        mask[i, :seq_len] = False  # 有效秒數標記為 False
        targets[i] = b["quality_score"]

    return {
        "vis": padded_vis,
        "geom": padded_geom,
        "aud": padded_aud,
        "mask": mask,
        "targets": targets,
    }