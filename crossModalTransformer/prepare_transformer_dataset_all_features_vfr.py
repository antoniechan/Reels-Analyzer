#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 (_vfr): Multimodal Time-Series Dataset Builder for Transformer (Viral / Reach Criterion)
------------------------------------------------------------------------------------------------
核心功能：
1. 優先讀取同層資料夾的 `reels_classification_summary_vfr.json`（若尚未執行重算，則退回 `reels_classification_summary.json` / `video_classification_summary.json`）。
2. 完整保留 33 維多模態時序特徵（8 Visual + 20 ROI + 5 Audio），並記錄 `criterion="vfr"` 與各項傳播指標。
3. 輸出至同層子資料夾 `transformer_engine_vfr/transformer_multimodal_dataset_vfr.pt`，絕不覆蓋原有檔案。
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROC_DIR,
    AUDIO_PROC_DIR,
    NEW_REELS_DIR,
    TRANSFORMER_DIR,
)

SUFFIX = "_vfr"
VFR_CLASS_JSON = Path(DATA_PROC_DIR) / f"reels_classification_summary{SUFFIX}.json"
PRIMARY_CLASS_JSON = Path(DATA_PROC_DIR) / "reels_classification_summary.json"
FALLBACK_CLASS_JSON = Path(DATA_PROC_DIR) / "video_classification_summary.json"
DEFAULT_CLASS_JSON = (
    VFR_CLASS_JSON if VFR_CLASS_JSON.exists()
    else (PRIMARY_CLASS_JSON if PRIMARY_CLASS_JSON.exists() else FALLBACK_CLASS_JSON)
)
DEFAULT_AUDIO_JSON = Path(AUDIO_PROC_DIR) / "dual_stream_audio_saliency_all_records.json"

OUTPUT_DIR = Path(TRANSFORMER_DIR) / f"transformer_engine{SUFFIX}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"transformer_multimodal_dataset{SUFFIX}.pt"

VIS_8_COLS = [f"vis_dim_{i}" for i in range(8)]
ROI_NAMES = ["brow_forehead", "eyes_duchenne", "nasolabial_shear", "mouth_zygomaticus", "global_face"]
ROI_STAT_SUFFIXES = ["mu_mean", "mu_p90", "dt_mean", "dt_p90"]
ROI_20_COLS = [f"{r}_{s}" for r in ROI_NAMES for s in ROI_STAT_SUFFIXES]
AUD_5_COLS = [f"aud_dim_{i}" for i in range(5)]
ALL_33_COLS = VIS_8_COLS + ROI_20_COLS + AUD_5_COLS

EXPRESSION_KEY_VERTICES = {
    0, 13, 14, 17, 61, 291, 33, 263, 130, 359,
    458, 461, 370, 238, 241, 141,
}


def build_media_index(root_dir: Path) -> tuple[dict, dict, dict]:
    print(f"[*] 正在索引 {root_dir} 下的多模態張量與特徵檔（排除 _quarantined*）...")
    npz_index, color_index, qc_index = {}, {}, {}

    for dirpath, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_quarantined")]
        acc_folder = os.path.basename(dirpath).lower().strip()
        if not acc_folder:
            continue

        for fname in filenames:
            f_low = fname.lower()
            full_p = os.path.join(dirpath, fname)
            if f_low.endswith("_multimodal_seq.npz"):
                npz_index[(acc_folder, f_low[:-19])] = full_p
            elif f_low.endswith("_color_lighting.json"):
                color_index[(acc_folder, f_low[:-20])] = full_p
            elif f_low.endswith("_qc.json"):
                qc_index[(acc_folder, f_low[:-8])] = full_p

    print(f"[*] 索引完成: .npz={len(npz_index)}, _color_lighting={len(color_index)}, _qc={len(qc_index)}")
    return npz_index, color_index, qc_index


def extract_visual_8d_timeseries(cl_series: list, target_t: int) -> np.ndarray:
    vis_matrix = np.zeros((target_t, 8), dtype=np.float32)
    actual_points = len(cl_series)
    if actual_points == 0:
        return vis_matrix

    for t in range(target_t):
        start_idx = min(max(0, int((t / target_t) * actual_points)), actual_points - 1)
        end_idx = min(max(start_idx + 1, int(((t + 1) / target_t) * actual_points)), actual_points)
        chunk = cl_series[start_idx:end_idx]
        vis_matrix[t, 0] = np.mean([float(c.get("colorfulness", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 1] = np.mean([float(c.get("brightness", c.get("luminance", 0.0)) or 0.0) for c in chunk])
        vis_matrix[t, 2] = np.mean([float(c.get("face_rms_contrast", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 3] = np.mean([float(c.get("sharpness", c.get("laplacian_var", 0.0)) or 0.0) for c in chunk])
        vis_matrix[t, 4] = np.mean([float(c.get("saturation", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 5] = np.mean([float(c.get("skin_warmth_cbcr", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 6] = np.mean([float(c.get("figure_ground_lum_contrast", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 7] = np.mean([float(c.get("global_contrast", c.get("rms_contrast", 0.0)) or 0.0) for c in chunk])

    return vis_matrix


def extract_qc_20d_fallback(qc_path: str, max_t: int) -> tuple[np.ndarray, np.ndarray]:
    roi_20 = np.zeros((max_t, 20), dtype=np.float32)
    valid_mask = np.zeros(max_t, dtype=np.float32)
    try:
        with open(qc_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return roi_20, valid_mask

    transitions = d.get("consecutive_qc_metrics", [])
    if not transitions:
        return roi_20, valid_mask

    sec_all_collect = {s: [] for s in range(max_t)}
    total_transitions = len(transitions)
    for idx, trans in enumerate(transitions):
        if not trans.get("valid_transition", False):
            continue
        sec_idx = min(max(0, int((idx / max(total_transitions, 1)) * max_t)), max_t - 1)
        for fm in trans.get("facet_metrics", []):
            mu_val = float(fm.get("abs_mu", 0.0) or 0.0)
            if mu_val > 1e-6:
                sec_all_collect[sec_idx].append(mu_val)

    prev_val = 0.0
    for s in range(max_t):
        if sec_all_collect[s]:
            valid_mask[s] = 1.0
            m_val = float(np.mean(sec_all_collect[s]))
            dt_val = float(0.5 * np.log((1.0 + np.clip(m_val, 0.0, 0.999)) / (1.0 - np.clip(m_val, 0.0, 0.999))))
            for r_i in range(5):
                roi_20[s, r_i * 4:r_i * 4 + 2] = m_val
                roi_20[s, r_i * 4 + 2:r_i * 4 + 4] = dt_val
            prev_val = m_val
        elif prev_val > 0.0:
            prev_dt = float(0.5 * np.log((1.0 + np.clip(prev_val, 0.0, 0.999)) / (1.0 - np.clip(prev_val, 0.0, 0.999))))
            for r_i in range(5):
                roi_20[s, r_i * 4:r_i * 4 + 2] = prev_val
                roi_20[s, r_i * 4 + 2:r_i * 4 + 4] = prev_dt

    return roi_20, valid_mask


def extract_audio_5d_timeseries(audio_info: dict, max_t: int) -> np.ndarray:
    aud_5 = np.zeros((max_t, 5), dtype=np.float32)
    mean_eng = float(audio_info.get("mean_engagement_score") or 0.45)
    hook_5s = float(audio_info.get("hook_first_5s_score") or mean_eng)
    mean_sem = float(audio_info.get("mean_semantic_score") or mean_eng)
    pacing_std = float(audio_info.get("std_engagement_score") or 0.0)

    aud_5[:, 0] = mean_eng
    aud_5[:, 1] = 0.2
    aud_5[:, 2] = mean_sem
    aud_5[:min(5, max_t), 2] = hook_5s
    aud_5[:, 3] = hook_5s
    aud_5[:, 4] = pacing_std

    for p_sec in audio_info.get("top_peak_seconds", []):
        sec = int(round(float(p_sec))) - 1
        if 0 <= sec < max_t:
            aud_5[sec, 1] = 1.0
            if sec + 1 < max_t:
                aud_5[sec + 1, 1] = max(aud_5[sec + 1, 1], 0.6)
            if sec - 1 >= 0:
                aud_5[sec - 1, 1] = max(aud_5[sec - 1, 1], 0.6)

    return aud_5


def pad_or_slice_cols(arr: np.ndarray, target_cols: int) -> np.ndarray:
    t_len, cur_cols = arr.shape
    if cur_cols == target_cols:
        return arr.astype(np.float32)
    if cur_cols > target_cols:
        return arr[:, :target_cols].astype(np.float32)
    out = np.zeros((t_len, target_cols), dtype=np.float32)
    out[:, :cur_cols] = arr.astype(np.float32)
    return out


def main():
    print("=" * 88)
    print(f"【啟動時序 Transformer 多模態資料集建置管線 ({SUFFIX} 傳播導向版)】")
    print(f"[*] 分類總表來源: {DEFAULT_CLASS_JSON}")
    print(f"[*] 輸出檔案路徑: {OUTPUT_FILE}")
    print("=" * 88)

    npz_index, color_index, qc_index = build_media_index(Path(NEW_REELS_DIR))

    audio_dict = {}
    if DEFAULT_AUDIO_JSON.exists():
        with open(DEFAULT_AUDIO_JSON, "r", encoding="utf-8") as f:
            adata = json.load(f)
        for v in adata.get("videos", []):
            acc = str(v.get("account_id", "")).lower().strip()
            vid = str(v.get("video_id", "")).lower().strip()
            audio_dict[(acc, vid)] = v

    videos_meta = []
    if DEFAULT_CLASS_JSON.exists():
        with open(DEFAULT_CLASS_JSON, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        videos_meta = cdata.get("videos", cdata) if isinstance(cdata, dict) else cdata

    dataset_records = []
    tier_map = {"bad": 0, "neutral": 1, "good": 2}
    from_npz_count, from_json_count = 0, 0

    for item in videos_meta:
        if not isinstance(item, dict):
            continue
        acc = str(item.get("account_id", "")).lower().strip()
        vid = str(item.get("video_id", "")).lower().strip()
        loc_stem = str(item.get("local_index_name", "")).lower().strip()
        tier_str = str(
            item.get(f"video_classification{SUFFIX}")
            or item.get("video_classification")
            or item.get("tier", "")
        ).lower().strip()
        if tier_str not in tier_map:
            continue

        cand_stems = [s for s in [loc_stem, vid] if s]
        npz_path = next((npz_index.get((acc, s)) for s in cand_stems if npz_index.get((acc, s))), None)
        cl_path = next((color_index.get((acc, s)) for s in cand_stems if color_index.get((acc, s))), None)
        qc_path = next((qc_index.get((acc, s)) for s in cand_stems if qc_index.get((acc, s))), None)
        a_info = audio_dict.get((acc, vid)) or audio_dict.get((acc, loc_stem), {})

        if npz_path:
            try:
                npz = np.load(npz_path)
                qc_roi_seq = npz["qc_roi_seq"]
                vis_seq = npz["visual_seq"]
                aud_seq = npz["audio_seq"]
                qc_mask = npz["qc_valid_mask"]
                T = min(int(len(qc_roi_seq)), 180)
                if T < 3:
                    continue
                feature_matrix = np.hstack([
                    pad_or_slice_cols(vis_seq[:T], 8),
                    pad_or_slice_cols(qc_roi_seq[:T], 20),
                    pad_or_slice_cols(aud_seq[:T], 5),
                ]).astype(np.float32)

                dataset_records.append({
                    "criterion": "vfr",
                    "account_id": acc,
                    "video_id": vid,
                    "tier_label": tier_map[tier_str],
                    "quality_score": float(item.get(f"video_quality_score{SUFFIX}") or item.get("video_quality_score") or 50.0),
                    "shares": float(item.get("shares") or a_info.get("shares") or 0.0),
                    "SRV": float(item.get("SRV") or a_info.get("SRV") or 0.0),
                    "ERV": float(item.get("ERV") or a_info.get("ERV") or 0.0),
                    "VFR": float(item.get("VFR") or a_info.get("VFR") or 0.0),
                    "CLR": float(item.get("CLR") or a_info.get("CLR") or 0.0),
                    "SLR": float(item.get("SLR") or a_info.get("SLR") or 0.0),
                    "duration": T,
                    "feature_names": ALL_33_COLS,
                    "modality_dims": {"vis_dim": 8, "geom_dim": 20, "aud_dim": 5},
                    "face_valid_mask": torch.tensor(qc_mask[:T] > 0.5, dtype=torch.bool),
                    "time_series": torch.tensor(feature_matrix, dtype=torch.float32),
                })
                from_npz_count += 1
                continue
            except Exception:
                pass

        if not cl_path:
            continue
        try:
            with open(cl_path, "r", encoding="utf-8") as f:
                cl_data = json.load(f)
        except Exception:
            continue
        cl_series = cl_data.get("second_level_time_series", [])
        if len(cl_series) < 3:
            continue
        raw_dur = float(cl_data.get("duration_sec", len(cl_series)))
        T = min(int(round(raw_dur)) if raw_dur > 0 else len(cl_series), 180)
        if T < 3:
            continue

        vis_8 = extract_visual_8d_timeseries(cl_series, target_t=T)
        roi_20, valid_mask = extract_qc_20d_fallback(qc_path, max_t=T) if qc_path else (
            np.zeros((T, 20), dtype=np.float32), np.zeros(T, dtype=np.float32)
        )
        aud_5 = extract_audio_5d_timeseries(a_info, max_t=T)
        feature_matrix = np.hstack([vis_8, roi_20, aud_5]).astype(np.float32)

        dataset_records.append({
            "criterion": "vfr",
            "account_id": acc,
            "video_id": vid,
            "tier_label": tier_map[tier_str],
            "quality_score": float(item.get(f"video_quality_score{SUFFIX}") or item.get("video_quality_score") or 50.0),
            "shares": float(item.get("shares") or a_info.get("shares") or 0.0),
            "SRV": float(item.get("SRV") or a_info.get("SRV") or 0.0),
            "ERV": float(item.get("ERV") or a_info.get("ERV") or 0.0),
            "VFR": float(item.get("VFR") or a_info.get("VFR") or 0.0),
            "CLR": float(item.get("CLR") or a_info.get("CLR") or 0.0),
            "SLR": float(item.get("SLR") or a_info.get("SLR") or 0.0),
            "duration": T,
            "feature_names": ALL_33_COLS,
            "modality_dims": {"vis_dim": 8, "geom_dim": 20, "aud_dim": 5},
            "face_valid_mask": torch.tensor(valid_mask > 0.5, dtype=torch.bool),
            "time_series": torch.tensor(feature_matrix, dtype=torch.float32),
        })
        from_json_count += 1

    torch.save(dataset_records, OUTPUT_FILE)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"[+] 成功打包 {len(dataset_records)} 支影片 (.npz: {from_npz_count}, JSON: {from_json_count}) -> {OUTPUT_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
