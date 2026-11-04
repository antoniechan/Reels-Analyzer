#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Comprehensive Multi-Modal Master Dataset (Visual + QC + Audio + Cross-Modal)
---------------------------------------------------------------------------------
功能：
1. 整合四大維度完整特徵：
   - 視覺光影 (Colorfulness, Face RMS, Skin Warmth, Fg-Bg Contrast)
   - 純幾何微表情 QC (Key Facet Mu, Mean Mu, Mu Std, Valid Ratio)
   - 純聲學與語意 Saliency (Hook 5s Score, Mean/Max Saliency, Peak Counts)
   - 跨模態視聽協同 (Optimal Lag Max Corr, Peak Coincidence Rate, Lag Seconds)
2. 匯出 comprehensive_master_dataset.csv 供 XGBoost / ML 模型進行全特徵訓練。
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from config import VIDEO_PROC_DIR, DATA_PROC_DIR, AUDIO_PROC_DIR, DEFAULT_REELS_DIR, CROSS_MODAL_PROC_DIR

DEFAULT_AUDIO_JSON = AUDIO_PROC_DIR / "local_transcript_audio_saliency_summary_all_tiers.json"
CROSS_MODAL_CSV = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_dataset.csv"
COLOR_CSV_DIR = VIDEO_PROC_DIR / "color_lighting_analysis" / "visual_color_lighting_dataset.csv"
OUTPUT_DIR = CROSS_MODAL_PROC_DIR / "master_dataset_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_KEY_VERTICES = set([0, 13, 14, 17, 61, 291, 33, 263, 130, 359])


def extract_qc_static_features(qc_path: str) -> dict:
    """從 *_qc.json 提取純幾何微表情靜態指標"""
    try:
        with open(qc_path, "r", encoding="utf-8") as f:
            qc = json.load(f)
    except Exception:
        return {}

    transitions = qc.get("consecutive_qc_metrics", [])
    if not transitions:
        return {}

    all_mus = []
    key_mus = []
    valid_count = 0

    for t in transitions:
        if not t.get("valid_transition", False):
            continue
        valid_count += 1
        for fm in t.get("facet_metrics", []):
            mu_val = float(fm.get("abs_mu", 0.0))
            if mu_val > 1e-6:
                all_mus.append(mu_val)
                verts = fm.get("vertices", [])
                if any(v in EXPRESSION_KEY_VERTICES for v in verts):
                    key_mus.append(mu_val)

    if not all_mus:
        return {}

    arr_all = np.array(all_mus)
    arr_key = np.array(key_mus) if key_mus else arr_all

    return {
        "qc_valid_transition_ratio": valid_count / len(transitions),
        "qc_key_facets_mu_mean": float(np.mean(arr_key)),
        "qc_key_facets_mu_std": float(np.std(arr_key)),
        "qc_all_facets_mu_mean": float(np.mean(arr_all)),
        "qc_all_facets_mu_std": float(np.std(arr_all)),
        "qc_teichmuller_dist_proxy": float(0.5 * np.log((1 + np.clip(np.mean(arr_key), 0, 0.99)) / (1 - np.clip(np.mean(arr_key), 0, 0.99)))),
    }


def main():
    print("[*] 正在建構全模態完整資料集 (Comprehensive Master Dataset)...")

    # 1. 讀取色彩光影資料集
    color_csv = COLOR_CSV_DIR
    if not color_csv.exists():
        print(f"[-] 找不到色彩資料集: {color_csv}")
        return
    df_color = pd.read_csv(color_csv)

    # 2. 讀取跨模態協同資料集
    if not CROSS_MODAL_CSV.exists():
        print(f"[-] 找不到跨模態資料集: {CROSS_MODAL_CSV}")
        return
    df_cm = pd.read_csv(CROSS_MODAL_CSV)

    # 3. 讀取音訊語意資料庫
    if not DEFAULT_AUDIO_JSON.exists():
        print(f"[-] 找不到音訊彙總檔: {DEFAULT_AUDIO_JSON}")
        return
    with open(DEFAULT_AUDIO_JSON, "r", encoding="utf-8") as f:
        audio_data = json.load(f)

    audio_dict = {}
    for v in audio_data.get("videos", []):
        acc = str(v.get("account_id", "")).lower().strip()
        vid = str(v.get("video_id", "")).lower().strip()
        audio_dict[(acc, vid)] = {
            "audio_hook_first_5s_score": float(v.get("hook_first_5s_score") or 0.0),
            "audio_mean_engagement_score": float(v.get("mean_engagement_score") or 0.0),
            "audio_max_engagement_score": float(v.get("max_engagement_score") or 0.0),
            "audio_top_peak_count": len(v.get("top_peak_seconds", [])),
        }

    # 4. 建立 QC 路徑索引
    print(f"[*] 掃描 D:\\reels 下的所有 *_qc.json ...")
    pattern = str(Path(DEFAULT_REELS_DIR) / "**" / "*_qc.json")
    qc_files = glob.glob(pattern, recursive=True)
    qc_lookup = {}
    for orig_p in qc_files:
        norm_p = orig_p.replace("\\", "/")
        parts = norm_p.split("/")
        fname = parts[-1].lower()
        vid_stem = fname[:-8] if fname.endswith("_qc.json") else os.path.splitext(fname)[0]
        for folder_part in parts[:-1]:
            acc_cand = folder_part.lower().strip()
            if acc_cand and acc_cand not in ["d:", "reels"]:
                qc_lookup[(acc_cand, vid_stem)] = orig_p

    # 5. 標準化鍵值並合併
    df_color["account_id"] = df_color["account_id"].astype(str).str.lower().str.strip()
    df_color["video_id"] = df_color["video_id"].astype(str).str.lower().str.strip()
    df_cm["account_id"] = df_cm["account_id"].astype(str).str.lower().str.strip()
    df_cm["video_id"] = df_cm["video_id"].astype(str).str.lower().str.strip()

    cm_cols = ["account_id", "video_id", "sync_pearson_r", "optimal_lag_max_corr", "optimal_lag_seconds", "peak_coincidence_rate"]
    merged = pd.merge(df_color, df_cm[cm_cols], on=["account_id", "video_id"], how="inner")

    records = []
    print("[*] 正在補齊純 QC 幾何特徵與音訊語意特徵...")
    for idx, row in merged.iterrows():
        acc = row["account_id"]
        vid = row["video_id"]

        row_dict = row.to_dict()

        # 補入音訊特徵
        a_info = audio_dict.get((acc, vid), {})
        row_dict.update(a_info)

        # 補入純 QC 特徵
        qc_p = qc_lookup.get((acc, vid))
        qc_feats = extract_qc_static_features(qc_p) if qc_p else {}
        row_dict.update(qc_feats)

        records.append(row_dict)

    final_df = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "comprehensive_master_dataset.csv"
    final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print(f"[+] 全模態資料集建構完成！")
    print(f"  - 輸出路徑: {out_csv}")
    print(f"  - 總樣本數: {len(final_df)} 支影片")
    print(f"  - 總特徵數: {len(final_df.columns)} 欄位 (包含視覺、光影、純QC、純音訊與跨模態)")
    print("=" * 65)


if __name__ == "__main__":
    main()
