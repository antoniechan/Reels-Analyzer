#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomical ROI Pooling & Multimodal Feature Sequence Builder for Transformer
----------------------------------------------------------------------------
功能：
1. 將全臉 468 個 MediaPipe 頂點依據解剖功能劃分為 4 大表情肌群 ROI（並保留全臉 Global 基準）：
   - ROI_1: Brow & Forehead (眉心與額肌區 — 皺眉肌 Corrugator / 額肌 Frontalis，反映認知專注與嚴肅強調)
   - ROI_2: Eyes & Duchenne Zone (眼周與外眼角區 — 眼輪匝肌 Orbicularis Oculi，反映真誠微笑與親和力)
   - ROI_3: Nasolabial & Mid-Face Shear Boundary (鼻唇溝與面中提肌區 — 含 #458, #461 等高靈敏度應力交界帶)
   - ROI_4: Mouth & Zygomaticus Zone (口周與嘴角區 — 顴大肌與口輪匝肌，反映語音活力與積極情緒)
2. 從每支影片的 `*_qc.json` 中，逐秒（Time-step $t = 1 \\dots T$）執行空間 ROI Pooling：
   - 針對每個 ROI 內的三角形面片，計算 `abs_mu`、`teichmuller_distance` 與 `dilation` 的空間均值（Spatial Mean）與 90 分位數（Spatial P90）。
3. 對齊並融合多模態時間序列（Multimodal Temporal Alignment）：
   - 視覺光影序列（`*_color_lighting.json`）：色彩豐富度、臉部亮度、RMS 對比度、膚色暖度（Cr-Cb）、前景背景反差。
   - 聲學與語意序列（`*_audio_saliency.json`）：綜合吸睛度、純聲學能量、純語意密度、梯度顯著性、每秒字數密度。
   - 標籤與去偏誤元數據（`reels_classification_summary.json`）：`good/neutral/bad` 分級、`video_quality_score`、`raw_residual`、`SRV`、`ERV`、`VFR`。
4. 輸出標準化張量與結構化檔案，直接對接 PyTorch Multimodal Cross-Attention Transformer 訓練：
   - 單片對齊多模態序列：`<account_id>/<video_stem>_multimodal_seq.npz`（含 `qc_roi_seq`, `visual_seq`, `audio_seq`, `qc_valid_mask`）
   - 全庫彙總特徵與 PyTorch Dataset 載入器：`multimodal_transformer_dataset_manifest.json` 與 `multimodal_video_level_summary.csv`
"""

import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    NEW_REELS_DIR,
    DATA_PROC_DIR,
    VIDEO_PROC_DIR,
    AUDIO_PROC_DIR,
)

# ---------------------------------------------------------------------------
# 1. 解剖功能 ROI 頂點定義 (MediaPipe 468 Canonical Face Mesh)
# ---------------------------------------------------------------------------
ANATOMICAL_ROI_VERTICES = {
    "brow_forehead": {
        # 眉心皺眉肌、眉峰與額肌區
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
        9, 8, 168, 151, 108, 337, 109, 338,
    },
    "eyes_duchenne": {
        # 眼周眼輪匝肌與外眼角（Duchenne 微笑區）
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 130, 25, 110, 24, 23, 22, 26, 112,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 359, 255, 339, 254, 253, 252, 256, 341,
    },
    "nasolabial_shear": {
        # 鼻翼溝與面中提上唇肌應力交界帶（納入全臉統計最顯著之 #458, #461, #370 及對稱點）
        458, 461, 459, 460, 457, 438, 439, 370, 326, 327, 358, 279, 294, 455,
        238, 241, 239, 240, 237, 218, 219, 141, 97, 98, 129, 49, 64, 235,
        203, 206, 205, 36, 142, 423, 426, 425, 266, 371,
    },
    "mouth_zygomaticus": {
        # 口輪匝肌與顴大肌微笑嘴角區
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        95, 88, 178, 87, 14, 317, 402, 318, 324,
        57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287,
    },
}

ROI_NAMES = ["brow_forehead", "eyes_duchenne", "nasolabial_shear", "mouth_zygomaticus", "global_face"]

VISUAL_FEATURE_KEYS = [
    "colorfulness",
    "face_luminance_mean",
    "face_rms_contrast",
    "face_dynamic_range",
    "skin_exposure",
    "skin_warmth_cbcr",
    "figure_ground_lum_contrast",
    "figure_ground_sat_contrast",
]

AUDIO_FEATURE_KEYS = [
    "engagement_score",
    "acoustic_score",
    "semantic_score",
    "saliency_grad_mean",
    "char_density",
]


# ---------------------------------------------------------------------------
# 2. 預先建立「三角形面片 -> ROI 歸屬遮罩」加速查詢表
# ---------------------------------------------------------------------------
def build_triangle_roi_masks(canonical_triangles: np.ndarray) -> dict[str, np.ndarray]:
    """
    依據三角形的 3 個頂點是否落入對應解剖區域，預先建立布林遮罩 (num_triangles,)。
    可將每支影片的 ROI Pooling 運算加速 20 倍以上。
    """
    num_tris = len(canonical_triangles)
    masks = {"global_face": np.ones(num_tris, dtype=bool)}

    for roi_name, v_set in ANATOMICAL_ROI_VERTICES.items():
        roi_mask = np.zeros(num_tris, dtype=bool)
        for t_idx, tri in enumerate(canonical_triangles):
            # 只要三角形有至少 2 個頂點屬於該 ROI，即歸入該區域
            overlap = sum(1 for v in tri if int(v) in v_set)
            if overlap >= 2:
                roi_mask[t_idx] = True
        masks[roi_name] = roi_mask

    return masks


# ---------------------------------------------------------------------------
# 3. 單支影片多模態時序特徵抽取與 ROI Pooling
# ---------------------------------------------------------------------------
def pool_qc_to_roi_timeseries(
    qc_json_path: Path,
    duration_sec: int,
    roi_masks: dict[str, np.ndarray],
    zero_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    將單支影片的 `*_qc.json` 轉換為對齊秒級時間軸的 ROI 特徵張量：
    回傳:
      - qc_seq: shape (T, 20) -> 5 個區域 (4 ROI + Global) x 4 個指標 (mu_mean, mu_p90, dt_mean, dt_p90)
      - qc_valid_mask: shape (T,) -> 1.0 代表該秒有有效臉部微應變，0.0 代表該秒無臉或 B-roll 遮擋
      - video_level_qc_summary: 單片彙總字典（供表格迴歸或分類器使用）
    """
    feat_dim = len(ROI_NAMES) * 4  # 5 * 4 = 20 維
    qc_seq = np.zeros((duration_sec, feat_dim), dtype=np.float32)
    qc_valid_mask = np.zeros(duration_sec, dtype=np.float32)

    if not qc_json_path.exists():
        return qc_seq, qc_valid_mask, {}

    try:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = json.load(f)
    except Exception:
        return qc_seq, qc_valid_mask, {}

    transitions = qc_data.get("consecutive_qc_metrics", [])
    for trans in transitions:
        if not trans.get("valid_transition", False):
            continue
        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            continue

        sec_from = int(trans.get("from_second", 0))
        sec_to = int(trans.get("to_second", sec_from + 1))
        if sec_from < 0 or sec_from >= duration_sec:
            continue

        # 提取該過渡幀所有面片的 abs_mu 陣列（按 triangle_id 對齊）
        num_tris = len(roi_masks["global_face"])
        mu_vec = np.zeros(num_tris, dtype=np.float64)
        for fm in facet_metrics:
            tid = int(fm.get("triangle_id", -1))
            if 0 <= tid < num_tris:
                mu_vec[tid] = float(fm.get("abs_mu", 0.0))

        # 計算 Teichmüller 距離向量 d_T = 0.5 * ln((1+|mu|)/(1-|mu|))
        mu_clipped = np.clip(mu_vec, 0.0, 0.999)
        dt_vec = 0.5 * np.log((1.0 + mu_clipped) / (1.0 - mu_clipped))

        row_feats = []
        has_valid_signal = False
        for r_name in ROI_NAMES:
            r_mask = roi_masks[r_name]
            sub_mu = mu_vec[r_mask]
            sub_dt = dt_vec[r_mask]

            valid_sub = sub_mu > zero_eps
            if np.any(valid_sub):
                has_valid_signal = True
                v_mu = sub_mu[valid_sub]
                v_dt = sub_dt[valid_sub]
                row_feats.extend([
                    float(np.mean(v_mu)),
                    float(np.percentile(v_mu, 90)),
                    float(np.mean(v_dt)),
                    float(np.percentile(v_dt, 90)),
                ])
            else:
                row_feats.extend([0.0, 0.0, 0.0, 0.0])

        if has_valid_signal:
            # 若存在跨秒過渡 (例如 max_gap=2 秒)，將訊號填滿 [sec_from, min(duration_sec, sec_to))
            end_fill = min(duration_sec, max(sec_from + 1, sec_to))
            for s in range(sec_from, end_fill):
                qc_seq[s, :] = np.array(row_feats, dtype=np.float32)
                qc_valid_mask[s] = 1.0

    # 計算整部影片的 ROI 彙總指標（僅對有效人臉秒數求平均與標準差）
    summary = {
        "qc_valid_seconds": int(np.sum(qc_valid_mask)),
        "qc_valid_ratio": round(float(np.mean(qc_valid_mask)), 4),
    }
    valid_indices = np.where(qc_valid_mask > 0.5)[0]
    for r_idx, r_name in enumerate(ROI_NAMES):
        base = r_idx * 4
        if len(valid_indices) > 0:
            mu_mean_series = qc_seq[valid_indices, base]
            mu_p90_series = qc_seq[valid_indices, base + 1]
            dt_mean_series = qc_seq[valid_indices, base + 2]
            dt_p90_series = qc_seq[valid_indices, base + 3]

            summary[f"qc_{r_name}_mu_mean"] = round(float(np.mean(mu_mean_series)), 6)
            summary[f"qc_{r_name}_mu_std"] = round(float(np.std(mu_mean_series, ddof=1)) if len(valid_indices) > 1 else 0.0, 6)
            summary[f"qc_{r_name}_mu_p90_mean"] = round(float(np.mean(mu_p90_series)), 6)
            summary[f"qc_{r_name}_dt_mean"] = round(float(np.mean(dt_mean_series)), 6)
            summary[f"qc_{r_name}_dt_std"] = round(float(np.std(dt_mean_series, ddof=1)) if len(valid_indices) > 1 else 0.0, 6)
            summary[f"qc_{r_name}_dt_p90_mean"] = round(float(np.mean(dt_p90_series)), 6)
        else:
            for k in ["mu_mean", "mu_std", "mu_p90_mean", "dt_mean", "dt_std", "dt_p90_mean"]:
                summary[f"qc_{r_name}_{k}"] = 0.0

    return qc_seq, qc_valid_mask, summary


def load_visual_timeseries(cl_json_path: Path, duration_sec: int) -> tuple[np.ndarray, dict]:
    """讀取 `*_color_lighting.json` 並對齊至 (T, 8) 視覺光影張量"""
    vis_seq = np.zeros((duration_sec, len(VISUAL_FEATURE_KEYS)), dtype=np.float32)
    vis_summary = {}

    if not cl_json_path.exists():
        return vis_seq, vis_summary

    try:
        with open(cl_json_path, "r", encoding="utf-8") as f:
            cl_data = json.load(f)
    except Exception:
        return vis_seq, vis_summary

    ts_list = cl_data.get("second_level_time_series", [])
    for idx, item in enumerate(ts_list):
        s = int(round(float(item.get("second", idx))))
        if 0 <= s < duration_sec:
            for k_idx, k_name in enumerate(VISUAL_FEATURE_KEYS):
                vis_seq[s, k_idx] = float(item.get(k_name, 0.0))

    whole_sum = cl_data.get("whole_video_summary", {})
    hook_sum = cl_data.get("hook_window_summary", {})
    for k, v in whole_sum.items():
        vis_summary[f"vis_whole_{k}"] = round(float(v), 6)
    for k, v in hook_sum.items():
        vis_summary[f"vis_{k}"] = round(float(v), 6)

    return vis_seq, vis_summary


def load_audio_timeseries(audio_json_path: Path, duration_sec: int) -> tuple[np.ndarray, dict]:
    """讀取 `*_audio_saliency.json` 並對齊至 (T, 5) 聲學語意張量"""
    aud_seq = np.zeros((duration_sec, len(AUDIO_FEATURE_KEYS)), dtype=np.float32)
    aud_summary = {}

    if not audio_json_path.exists():
        return aud_seq, aud_summary

    try:
        with open(audio_json_path, "r", encoding="utf-8") as f:
            aud_data = json.load(f)
    except Exception:
        return aud_seq, aud_summary

    ts_list = aud_data.get("second_by_second_metrics", [])
    for item in ts_list:
        s = int(item.get("second", 1)) - 1  # audio second 從 1 開始索引
        if 0 <= s < duration_sec:
            for k_idx, k_name in enumerate(AUDIO_FEATURE_KEYS):
                aud_seq[s, k_idx] = float(item.get(k_name, 0.0))

    sm = aud_data.get("summary_metrics", {})
    for k, v in sm.items():
        if isinstance(v, (int, float)):
            aud_summary[f"aud_{k}"] = round(float(v), 6)

    return aud_seq, aud_summary


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset 載入器（可直接匯入至 Cross-Attention Transformer 訓練腳本）
# ---------------------------------------------------------------------------
def get_pytorch_dataset_class():
    """動態建立 PyTorch Dataset 類別，避免未安裝 torch 時影響特徵前處理執行"""
    import torch
    from torch.utils.data import Dataset

    class TrustLensMultimodalDataset(Dataset):
        def __init__(self, manifest_json_path: str, max_seq_len: int = 90):
            with open(manifest_json_path, "r", encoding="utf-8") as f:
                self.items = json.load(f)["samples"]
            self.max_seq_len = max_seq_len
            self.label_map = {"bad": 0, "neutral": 1, "good": 2}

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            meta = self.items[idx]
            npz = np.load(meta["npz_path"])
            qc_seq = npz["qc_roi_seq"]       # (T, 20): 4 解剖 ROI + Global 的微應變序列
            vis_seq = npz["visual_seq"]      # (T, 8) : 色彩豐富度、膚色暖度、對比度序列
            aud_seq = npz["audio_seq"]       # (T, 5) : 聲學能量、語意密度、梯度顯著性序列
            qc_mask = npz["qc_valid_mask"]   # (T,)   : 1=有講者人臉, 0=B-roll/無臉遮罩

            T = min(len(qc_seq), self.max_seq_len)
            pad_len = self.max_seq_len - T

            def pad_2d(arr):
                sub = arr[:T]
                return np.pad(sub, ((0, pad_len), (0, 0)), mode="constant") if pad_len > 0 else sub

            attn_mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
            qc_valid_pad = np.concatenate([qc_mask[:T], np.zeros(pad_len, dtype=np.float32)])

            return {
                "qc_roi": torch.tensor(pad_2d(qc_seq), dtype=torch.float32),       # (L, 20)
                "visual": torch.tensor(pad_2d(vis_seq), dtype=torch.float32),      # (L, 8)
                "audio": torch.tensor(pad_2d(aud_seq), dtype=torch.float32),       # (L, 5)
                "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),            # (L,)
                "qc_face_mask": torch.tensor(qc_valid_pad, dtype=torch.bool),      # (L,)
                "label": torch.tensor(self.label_map.get(meta["tier"], 1), dtype=torch.long),
                "quality_score": torch.tensor(meta["video_quality_score"], dtype=torch.float32),
                "srv": torch.tensor(meta["SRV"], dtype=torch.float32),
            }

    return TrustLensMultimodalDataset


# ---------------------------------------------------------------------------
# 5. 批次建構主流程
# ---------------------------------------------------------------------------
def build_multimodal_dataset(
    reels_dir: str,
    class_json_path: str,
    output_dir: str,
    max_videos: int = None,
):
    from compute_qc_deformation import get_canonical_triangles

    os.makedirs(output_dir, exist_ok=True)
    reels_root = Path(reels_dir)

    if not os.path.exists(class_json_path):
        raise FileNotFoundError(f"找不到分類清單檔案: {class_json_path}")

    with open(class_json_path, "r", encoding="utf-8") as f:
        c_data = json.load(f)
    videos_list = c_data.get("videos", c_data) if isinstance(c_data, dict) else c_data

    if max_videos is not None:
        videos_list = videos_list[:max_videos]

    print("=" * 90)
    print("🚀 【Multimodal ROI Pooling & 時序特徵對齊建構引擎 (Transformer Ready)】")
    print(f"  - 影片特徵根目錄:   {reels_root}")
    print(f"  - 標籤與指標來源:   {class_json_path}")
    print(f"  - 輸出彙總目錄:     {output_dir}")
    print(f"  - 目標處理影片數:   {len(videos_list)}")
    print("=" * 90 + "\n")

    canonical_triangles = get_canonical_triangles()
    roi_masks = build_triangle_roi_masks(canonical_triangles)
    for r_name, r_mask in roi_masks.items():
        print(f"  • ROI [{r_name:<18}]: 包含 {int(np.sum(r_mask)):>3} 個三角形面片")
    print("-" * 90)

    manifest_samples = []
    tabular_rows = []
    skipped_missing = 0
    start_t = time.time()

    for idx, v_info in enumerate(videos_list, start=1):
        acc_id = str(v_info.get("account_id", "")).strip()
        vid_id = str(v_info.get("video_id", "")).strip()
        loc_name = str(v_info.get("local_index_name", "")).strip()
        tier = str(v_info.get("video_classification", "neutral")).lower().strip()

        acc_dir = reels_root / acc_id
        if not acc_dir.exists():
            skipped_missing += 1
            continue

        # 定位檔案前綴 (優先 local_index_name 如 video_001，次選 video_id)
        stem = loc_name if (loc_name and (acc_dir / f"{loc_name}_qc.json").exists()) else vid_id
        qc_path = acc_dir / f"{stem}_qc.json"
        cl_path = acc_dir / f"{stem}_color_lighting.json"
        aud_path = acc_dir / f"{stem}_audio_saliency.json"

        if not qc_path.exists():
            skipped_missing += 1
            continue

        # 決定統一時間軸長度 T (秒)
        duration_sec = 0
        for p_check, k_dur in [(aud_path, "duration_seconds"), (cl_path, "duration_sec")]:
            if p_check.exists():
                try:
                    with open(p_check, "r", encoding="utf-8") as f:
                        d_tmp = json.load(f)
                    duration_sec = max(duration_sec, int(round(float(d_tmp.get(k_dur, 0)))))
                except Exception:
                    pass

        if duration_sec <= 0:
            try:
                with open(qc_path, "r", encoding="utf-8") as f:
                    q_tmp = json.load(f)
                duration_sec = max(1, int(q_tmp.get("total_transitions", 1)) + 1)
            except Exception:
                duration_sec = 1

        # 1. 提取三模態對齊時間序列
        qc_seq, qc_mask, qc_sum = pool_qc_to_roi_timeseries(qc_path, duration_sec, roi_masks)
        vis_seq, vis_sum = load_visual_timeseries(cl_path, duration_sec)
        aud_seq, aud_sum = load_audio_timeseries(aud_path, duration_sec)

        # 2. 儲存單片壓縮多模態張量 (.npz)
        npz_path = acc_dir / f"{stem}_multimodal_seq.npz"
        np.savez_compressed(
            npz_path,
            qc_roi_seq=qc_seq,
            visual_seq=vis_seq,
            audio_seq=aud_seq,
            qc_valid_mask=qc_mask,
        )

        # 3. 寫入 Manifest 與表格彙總列
        sample_meta = {
            "account_id": acc_id,
            "video_id": vid_id,
            "video_stem": stem,
            "tier": tier,
            "video_quality_score": float(v_info.get("video_quality_score") or 0.0),
            "raw_residual": float(v_info.get("raw_residual") or 0.0),
            "VFR": float(v_info.get("VFR") or 0.0),
            "ERV": float(v_info.get("ERV") or 0.0),
            "CLR": float(v_info.get("CLR") or 0.0),
            "SRV": float(v_info.get("SRV") or 0.0),
            "SLR": float(v_info.get("SLR") or 0.0),
            "duration_sec": duration_sec,
            "has_visual": cl_path.exists(),
            "has_audio": aud_path.exists(),
            "npz_path": str(npz_path.resolve()),
        }
        manifest_samples.append(sample_meta)

        tab_row = {**sample_meta, **qc_sum, **vis_sum, **aud_sum}
        tabular_rows.append(tab_row)

        if idx % 250 == 0 or idx == len(videos_list):
            print(f"  進度: [{idx}/{len(videos_list)}] ({time.time() - start_t:.1f}s) | 已封裝: {len(manifest_samples)} 支")

    # 4. 儲存全庫 Manifest JSON 與 Tabular CSV
    qc_col_names = []
    for r_name in ROI_NAMES:
        for stat_k in ["mu_mean", "mu_p90", "dt_mean", "dt_p90"]:
            qc_col_names.append(f"{r_name}_{stat_k}")

    manifest_payload = {
        "schema_info": {
            "qc_roi_seq_dim": len(qc_col_names),
            "qc_roi_seq_columns": qc_col_names,
            "visual_seq_dim": len(VISUAL_FEATURE_KEYS),
            "visual_seq_columns": VISUAL_FEATURE_KEYS,
            "audio_seq_dim": len(AUDIO_FEATURE_KEYS),
            "audio_seq_columns": AUDIO_FEATURE_KEYS,
        },
        "total_samples": len(manifest_samples),
        "skipped_missing": skipped_missing,
        "samples": manifest_samples,
    }

    manifest_path = os.path.join(output_dir, "multimodal_transformer_dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)

    df_tab = pd.DataFrame(tabular_rows)
    csv_path = os.path.join(output_dir, "multimodal_video_level_summary.csv")
    df_tab.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("🎉 多模態 ROI Pooling 與時序對齊建構完成！")
    print(f"  • 成功封裝樣本數:   {len(manifest_samples)} 支 (略過缺失: {skipped_missing})")
    print(f"  • Transformer 清單: {os.path.abspath(manifest_path)}")
    print(f"  • 計量/表格特徵表:  {os.path.abspath(csv_path)}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行解剖 ROI Pooling 並建構 Multimodal Transformer 時序數據集")
    parser.add_argument("--reels_dir", type=str, default=str(NEW_REELS_DIR), help="影片與特徵檔案根目錄 (預設 NEW_REELS_DIR)")
    parser.add_argument(
        "--class_json",
        type=str,
        default=str(DATA_PROC_DIR / "reels_classification_summary.json"),
        help="影片分類與去偏誤分數 JSON 路徑",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DATA_PROC_DIR / "multimodal_dataset"),
        help="輸出 Manifest 與彙總 CSV 目錄",
    )
    parser.add_argument("--max_videos", type=int, default=None, help="限制處理影片數 (測試用)")
    args = parser.parse_args()

    build_multimodal_dataset(
        reels_dir=args.reels_dir,
        class_json_path=args.class_json,
        output_dir=args.out_dir,
        max_videos=args.max_videos,
    )
