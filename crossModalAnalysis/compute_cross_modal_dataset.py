#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Comprehensive Multi-Modal Master Dataset v2 (Visual + ROI QC + Audio + Cross-Modal + Shares/SRV)
------------------------------------------------------------------------------------------------------
升級重點：
1. 無需重跑耗時的逐片 JSON 解析，直接無縫整合上游已建構的三大標準化表：
   - `DATA_PROC_DIR / "multimodal_dataset" / "multimodal_video_level_summary.csv"`（含 4 大解剖 ROI Pooling 指標：brow_forehead, eyes_duchenne, nasolabial_shear, mouth_zygomaticus, global_face，以及視覺光影與音訊語意指標、shares、SRV、ERV、VFR、CLR、SLR）
   - `CROSS_MODAL_PROC_DIR / "cross_modal_dataset.csv"`（含視聽時空協同指標：sync_pearson_r, optimal_lag_max_corr, peak_coincidence_rate, semantic_facial_max_corr）
   - `AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"`（補齊音訊總表欄位確保 100% 完整）
2. 若單獨執行時缺少 `visual_color_lighting_dataset.csv` 獨立檔，也能自動從 `multimodal_video_level_summary.csv` 提取全部視覺光影欄位，避免報錯。
3. 匯出 `comprehensive_master_dataset.csv`，直接供後續 OLS 計量模型與 XGBoost / LightGBM / Transformer 基準對比使用。
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd

from config import (
    VIDEO_PROC_DIR,
    AUDIO_PROC_DIR,
    CROSS_MODAL_PROC_DIR,
)

DEFAULT_MULTIMODAL_CSV = Path(CROSS_MODAL_PROC_DIR) / "multimodal_dataset" / "multimodal_video_level_summary.csv"
DEFAULT_CROSS_MODAL_CSV = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_dataset.csv"
DEFAULT_AUDIO_JSON = AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"
DEFAULT_COLOR_CSV = VIDEO_PROC_DIR / "color_lighting_analysis" / "visual_color_lighting_dataset.csv"
OUTPUT_DIR = Path(CROSS_MODAL_PROC_DIR) / "master_dataset_results"


def build_comprehensive_master(
    multimodal_csv: Path = DEFAULT_MULTIMODAL_CSV,
    cross_modal_csv: Path = DEFAULT_CROSS_MODAL_CSV,
    audio_json: Path = DEFAULT_AUDIO_JSON,
    color_csv: Path = DEFAULT_COLOR_CSV,
    output_dir: Path = OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 85)
    print("🚀 【正在建構全模態 Master Dataset v2 (Visual + ROI QC + Audio + Cross-Modal + SRV)】")
    print("=" * 85)

    if not multimodal_csv.exists():
        raise FileNotFoundError(f"[-] 找不到 ROI Pooling 彙總表: {multimodal_csv}，請先執行 build_multimodal_roi_dataset.py")
    if not cross_modal_csv.exists():
        raise FileNotFoundError(f"[-] 找不到跨模態協同資料集: {cross_modal_csv}，請先執行 compute_cross_modal_saliency.py")

    df_mm = pd.read_csv(multimodal_csv)
    df_cm = pd.read_csv(cross_modal_csv)

    for d in [df_mm, df_cm]:
        d["account_id"] = d["account_id"].astype(str).str.lower().str.strip()
        d["video_id"] = d["video_id"].astype(str).str.lower().str.strip()

    # 1. 合併 ROI Pooling 多模態主表 與 視聽協同指標表
    cm_keep_cols = [
        "account_id",
        "video_id",
        "sync_pearson_r",
        "sync_pearson_p",
        "sync_spearman_rho",
        "optimal_lag_seconds",
        "optimal_lag_max_corr",
        "peak_coincidence_rate",
        "semantic_facial_sync_r",
        "semantic_facial_max_corr",
        "semantic_facial_peak_coinc",
    ]
    cm_keep_cols = [c for c in cm_keep_cols if c in df_cm.columns]
    merged = pd.merge(df_mm, df_cm[cm_keep_cols], on=["account_id", "video_id"], how="inner")
    print(f"[*] 成功對齊 ROI 多模態主表與跨模態協同表: {len(merged)} 支影片")

    # 2. 若存在獨立的 visual_color_lighting_dataset.csv，補入尚未包含的色彩衍生欄位
    if color_csv.exists():
        df_col = pd.read_csv(color_csv)
        df_col["account_id"] = df_col["account_id"].astype(str).str.lower().str.strip()
        df_col["video_id"] = df_col["video_id"].astype(str).str.lower().str.strip()
        extra_col_fields = [c for c in df_col.columns if c not in merged.columns]
        if extra_col_fields:
            merged = pd.merge(
                merged,
                df_col[["account_id", "video_id"] + extra_col_fields],
                on=["account_id", "video_id"],
                how="left",
            )
            print(f"[*] 額外補入 {len(extra_col_fields)} 個色彩光影衍生欄位")

    # 3. 讀取 dual_stream_audio_saliency_all_records.json 補齊音訊總表欄位（確保 sharesCount/SRV 與 peak 數一致）
    if audio_json.exists():
        with open(audio_json, "r", encoding="utf-8") as f:
            audio_data = json.load(f)
        audio_rows = []
        for v in audio_data.get("videos", []):
            audio_rows.append({
                "account_id": str(v.get("account_id", "")).lower().strip(),
                "video_id": str(v.get("video_id", "")).lower().strip(),
                "audio_top_peak_count": len(v.get("top_peak_seconds", [])),
                "audio_shares_raw": v.get("shares"),
                "audio_srv_raw": v.get("SRV"),
            })
        df_aud_extra = pd.DataFrame(audio_rows).drop_duplicates(subset=["account_id", "video_id"])
        merged = pd.merge(merged, df_aud_extra, on=["account_id", "video_id"], how="left")

        # 若 shares 或 SRV 有缺漏，從 audio_json 補齊
        if "shares" in merged.columns and "audio_shares_raw" in merged.columns:
            merged["shares"] = merged["shares"].fillna(merged["audio_shares_raw"]).fillna(0.0)
        if "SRV" in merged.columns and "audio_srv_raw" in merged.columns:
            merged["SRV"] = merged["SRV"].fillna(merged["audio_srv_raw"]).fillna(0.0)
        merged.drop(columns=[c for c in ["audio_shares_raw", "audio_srv_raw"] if c in merged.columns], inplace=True)

    out_csv = output_dir / "comprehensive_master_dataset.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 85)
    print("[+] 全模態 Master Dataset v2 建構完成！")
    print(f"  - 輸出路徑: {out_csv.resolve()}")
    print(f"  - 總樣本數: {len(merged)} 支影片")
    print(f"  - 總特徵數: {len(merged.columns)} 個欄位 (含 5 大 ROI QC、視覺光影、音訊語意、跨模態協同與 Shares/SRV)")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建構全模態 Master Dataset v2")
    parser.add_argument("--mm_csv", type=str, default=str(DEFAULT_MULTIMODAL_CSV), help="build_multimodal_roi_dataset-v2.py 輸出的 CSV")
    parser.add_argument("--cm_csv", type=str, default=str(DEFAULT_CROSS_MODAL_CSV), help="compute_cross_modal_saliency-v2.py 輸出的 CSV")
    parser.add_argument("--audio_json", type=str, default=str(DEFAULT_AUDIO_JSON), help="dual_stream_audio_saliency_all_records.json 路徑")
    parser.add_argument("--out_dir", type=str, default=str(OUTPUT_DIR), help="輸出目錄")
    args = parser.parse_args()

    build_comprehensive_master(
        multimodal_csv=Path(args.mm_csv),
        cross_modal_csv=Path(args.cm_csv),
        audio_json=Path(args.audio_json),
        output_dir=Path(args.out_dir),
    )
