#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Multimodal Time-Series Dataset Builder for Transformer (NewReels + Compact 8-Dim)
--------------------------------------------------------------------------------------------
核心功能：
1. 針對 `NEW_REELS_DIR`（自動排除 `_quarantined*` 隔離資料夾），支援 `(account_id, local_index_name)` 與 `(account_id, video_id)` 雙鍵查找。
2. 預設讀取 `DATA_PROC_DIR / "reels_classification_summary.json"`（若不存在則退回 `video_classification_summary.json`）與 `AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"`。
3. 僅組裝原來的 8 維時序特徵 `(T, 8)`：[4 視覺光影, 2 QC 形變 (key_mus, all_mus), 2 音訊語意]。
4. 打包輸出為 `transformer_multimodal_dataset.pt`。
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

DEFAULT_CLASS_JSON = DATA_PROC_DIR / "reels_classification_summary.json"
DEFAULT_AUDIO_JSON = AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"
OUTPUT_DIR = Path(TRANSFORMER_DIR) / "transformer_engine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_KEY_VERTICES = {
    0, 13, 14, 17, 61, 291, 33, 263, 130, 359,
    458, 461, 370, 238, 241, 141,
}


def build_newreels_index(root_dir: Path) -> tuple[dict, dict, dict]:
    """掃描 NEW_REELS_DIR（自動排除 _quarantined*），建立 .npz、_color_lighting.json 與 _qc.json 索引。"""
    print(f"[*] 正在索引 {root_dir} 下的多模態檔案（自動排除 _quarantined*）...")
    npz_index = {}
    color_index = {}
    qc_index = {}

    for dirpath, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_quarantined")]
        acc_folder = os.path.basename(dirpath).lower().strip()
        if not acc_folder:
            continue

        for fname in filenames:
            f_low = fname.lower()
            full_p = os.path.join(dirpath, fname)
            if f_low.endswith("_multimodal_seq.npz"):
                stem = f_low[:-19]
                npz_index[(acc_folder, stem)] = full_p
            elif f_low.endswith("_color_lighting.json"):
                stem = f_low[:-20]
                color_index[(acc_folder, stem)] = full_p
            elif f_low.endswith("_qc.json"):
                stem = f_low[:-8]
                qc_index[(acc_folder, stem)] = full_p

    print(
        f"[*] 索引完成: .npz={len(npz_index)}, "
        f"_color_lighting={len(color_index)}, _qc={len(qc_index)}"
    )
    return npz_index, color_index, qc_index


def extract_visual_timeseries(cl_series: list, target_t: int) -> np.ndarray:
    """安全地將 cl_series 對齊至 target_t 秒，輸出 (T, 4) 視覺光影特徵。"""
    vis_matrix = np.zeros((target_t, 4), dtype=np.float32)
    actual_points = len(cl_series)
    if actual_points == 0:
        return vis_matrix

    for t in range(target_t):
        start_idx = min(max(0, int((t / target_t) * actual_points)), actual_points - 1)
        end_idx = min(max(start_idx + 1, int(((t + 1) / target_t) * actual_points)), actual_points)
        chunk = cl_series[start_idx:end_idx]
        vis_matrix[t, 0] = np.mean([float(c.get("colorfulness", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 1] = np.mean([float(c.get("face_rms_contrast", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 2] = np.mean([float(c.get("skin_warmth_cbcr", 0.0) or 0.0) for c in chunk])
        vis_matrix[t, 3] = np.mean([float(c.get("figure_ground_lum_contrast", 0.0) or 0.0) for c in chunk])

    return vis_matrix


def extract_qc_timeseries(qc_path: str, max_t: int) -> tuple[np.ndarray, np.ndarray]:
    """從 _qc.json 提取每秒的 key_mus 與 all_mus，輸出各 (T,)。"""
    key_ts = np.zeros(max_t, dtype=np.float32)
    all_ts = np.zeros(max_t, dtype=np.float32)
    try:
        with open(qc_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return key_ts, all_ts

    transitions = d.get("consecutive_qc_metrics", [])
    if not transitions:
        return key_ts, all_ts

    sec_key_collect = {s: [] for s in range(max_t)}
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
                verts = fm.get("vertices", [])
                if any(v in EXPRESSION_KEY_VERTICES for v in verts):
                    sec_key_collect[sec_idx].append(mu_val)

    for s in range(max_t):
        key_ts[s] = np.mean(sec_key_collect[s]) if sec_key_collect[s] else (key_ts[s - 1] if s > 0 else 0.0)
        all_ts[s] = np.mean(sec_all_collect[s]) if sec_all_collect[s] else (all_ts[s - 1] if s > 0 else 0.0)

    return key_ts, all_ts


def extract_audio_timeseries(audio_info: dict, max_t: int) -> tuple[np.ndarray, np.ndarray]:
    """提取每秒聲能 peak 脈衝與文本語意吸睛度，輸出各 (T,)。"""
    audio_energy_ts = np.full(max_t, 0.2, dtype=np.float32)
    semantic_ts = np.zeros(max_t, dtype=np.float32)

    mean_eng = float(audio_info.get("mean_engagement_score") or 0.45)
    hook_5s = float(audio_info.get("hook_first_5s_score") or mean_eng)
    semantic_ts[:] = mean_eng
    semantic_ts[:min(5, max_t)] = hook_5s

    for p_sec in audio_info.get("top_peak_seconds", []):
        sec = int(round(float(p_sec))) - 1
        if 0 <= sec < max_t:
            audio_energy_ts[sec] = 1.0
            if sec + 1 < max_t:
                audio_energy_ts[sec + 1] = max(audio_energy_ts[sec + 1], 0.6)
            if sec - 1 >= 0:
                audio_energy_ts[sec - 1] = max(audio_energy_ts[sec - 1], 0.6)

    return audio_energy_ts, semantic_ts


def main():
    print("=" * 84)
    print("【啟動時序 Transformer 多模態資料集建置管線 (NEW_REELS_DIR + 原 8 維特徵)】")
    print(f"[*] 分類總表來源: {DEFAULT_CLASS_JSON}")
    print(f"[*] 目標影片目錄: {NEW_REELS_DIR}")
    print("=" * 84)

    npz_index, color_index, qc_index = build_newreels_index(Path(NEW_REELS_DIR))

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
    abnormal_len_records = []
    from_npz_count = 0
    from_json_count = 0

    for item in videos_meta:
        if not isinstance(item, dict):
            continue
        acc = str(item.get("account_id", "")).lower().strip()
        vid = str(item.get("video_id", "")).lower().strip()
        loc_stem = str(item.get("local_index_name", "")).lower().strip()
        tier_str = str(item.get("video_classification", item.get("tier", ""))).lower().strip()
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
                qc_roi_seq = npz["qc_roi_seq"]     # (T, 20)
                vis_seq = npz["visual_seq"]       # (T, 8)
                aud_seq = npz["audio_seq"]        # (T, 5)

                T = int(len(qc_roi_seq))
                if T > 180:
                    abnormal_len_records.append((acc, vid, T, T))
                    T = 180
                    qc_roi_seq = qc_roi_seq[:T]
                    vis_seq = vis_seq[:T]
                    aud_seq = aud_seq[:T]
                if T < 3:
                    continue

                # 僅取原 8 維：4 視覺 + 2 幾何 (nasolabial mu_mean, global_face mu_mean) + 2 音訊
                vis_4 = vis_seq[:, [0, 2, 5, 6]].astype(np.float32)
                key_mus = qc_roi_seq[:, 8].astype(np.float32)
                all_mus = qc_roi_seq[:, 16].astype(np.float32)
                aud_energy = aud_seq[:, 0].astype(np.float32)
                sem_saliency = aud_seq[:, 2].astype(np.float32)

                feature_matrix = np.column_stack([
                    vis_4,
                    key_mus[:, None],
                    all_mus[:, None],
                    aud_energy[:, None],
                    sem_saliency[:, None],
                ]).astype(np.float32)

                dataset_records.append({
                    "account_id": acc,
                    "video_id": vid,
                    "tier_label": tier_map[tier_str],
                    "quality_score": float(item.get("video_quality_score") or 50.0),
                    "duration": T,
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
        T = int(round(raw_dur)) if raw_dur > 0 else len(cl_series)
        if T > 180:
            abnormal_len_records.append((acc, vid, T, len(cl_series)))
            T = 180
        if T < 3:
            continue

        vis_matrix = extract_visual_timeseries(cl_series, target_t=T)
        if qc_path:
            key_mus, all_mus = extract_qc_timeseries(qc_path, max_t=T)
        else:
            key_mus = np.zeros(T, dtype=np.float32)
            all_mus = np.zeros(T, dtype=np.float32)

        aud_energy, sem_saliency = extract_audio_timeseries(a_info, max_t=T)

        feature_matrix = np.column_stack([
            vis_matrix,
            key_mus[:, None],
            all_mus[:, None],
            aud_energy[:, None],
            sem_saliency[:, None],
        ]).astype(np.float32)

        dataset_records.append({
            "account_id": acc,
            "video_id": vid,
            "tier_label": tier_map[tier_str],
            "quality_score": float(item.get("video_quality_score") or 50.0),
            "duration": T,
            "time_series": torch.tensor(feature_matrix, dtype=torch.float32),
        })
        from_json_count += 1

    if abnormal_len_records:
        print(f"\n[!] 偵測到 {len(abnormal_len_records)} 支影片時長超過 180 秒 (已截斷上限至 180s)")

    output_file = OUTPUT_DIR / "transformer_multimodal_dataset.pt"
    torch.save(dataset_records, output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(
        f"\n[+] 成功打包 {len(dataset_records)} 支影片 (來自 .npz: {from_npz_count}, 來自 JSON: {from_json_count}) "
        f"-> {output_file} ({file_size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()