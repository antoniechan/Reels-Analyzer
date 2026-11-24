#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Audit for the 4 Excluded Videos in Cross-Modal Saliency Analysis
---------------------------------------------------------------------------
功能：
1. 比對 `dual_stream_audio_saliency_all_records.json`（全量 3,000 支）與 `cross_modal_dataset.csv`（成功計算 2,996 支），精準找出被排除的 4 支影片。
2. 逐一檢查這 4 支影片在問題 1 提及的三個維度上的真實數值：
   - 可能性 A：有效人臉過渡幀數 (`qc_valid_mask > 0.5` 的總秒數與比例) 是否 < 3 秒（或 mask-aware 下的 < 4 秒）。
   - 可能性 B：影片總時長 (`duration_seconds` / `len(qc_seq)`) 是否 < 5 秒。
   - 可能性 C：序列標準差 (`std(audio_ts)` 或 `std(face_ts)`) 是否 < 1e-8（常數直線導致相關係數分母為零）。
   - 其他原因：檔案路徑缺失、JSON 解析失敗或別名未匹配。
3. 在終端機印出完整診斷明細表，並匯出 `excluded_4_videos_diagnosis.csv`。
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    AUDIO_PROC_DIR,
    DATA_PROC_DIR,
    CROSS_MODAL_PROC_DIR,
    NEW_REELS_DIR,
)

DEFAULT_AUDIO_JSON = AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"
DEFAULT_CM_CSV = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_dataset.csv"
DEFAULT_CLASS_JSON = DATA_PROC_DIR / "reels_classification_summary.json"
OUTPUT_CSV = Path(CROSS_MODAL_PROC_DIR) / "excluded_4_videos_diagnosis.csv"


def load_alias_map(class_json_path: Path) -> dict:
    alias_map = {}
    if not class_json_path.exists():
        return alias_map
    with open(class_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", data) if isinstance(data, dict) else data
    for item in videos:
        if isinstance(item, dict):
            acc = str(item.get("account_id", "")).lower().strip()
            vid = str(item.get("video_id", "")).lower().strip()
            loc = str(item.get("local_index_name", "")).lower().strip()
            if acc and vid and loc:
                alias_map[(acc, vid)] = loc
    return alias_map


def diagnose_excluded_videos(
    audio_json_path: Path = DEFAULT_AUDIO_JSON,
    cm_csv_path: Path = DEFAULT_CM_CSV,
    class_json_path: Path = DEFAULT_CLASS_JSON,
    reels_dir: Path = Path(NEW_REELS_DIR),
    out_csv: Path = OUTPUT_CSV,
):
    if not audio_json_path.exists():
        raise FileNotFoundError(f"找不到音訊總表: {audio_json_path}")
    if not cm_csv_path.exists():
        raise FileNotFoundError(f"找不到跨模態資料集: {cm_csv_path}")

    with open(audio_json_path, "r", encoding="utf-8") as f:
        audio_data = json.load(f)
    all_videos = audio_data.get("videos", [])

    df_cm = pd.read_csv(cm_csv_path)
    included_keys = set(
        zip(
            df_cm["account_id"].astype(str).str.lower().str.strip(),
            df_cm["video_id"].astype(str).str.lower().str.strip(),
        )
    )

    alias_map = load_alias_map(class_json_path)

    excluded_list = []
    for v in all_videos:
        acc = str(v.get("account_id", "")).strip()
        vid = str(v.get("video_id", "")).strip()
        key = (acc.lower(), vid.lower())
        if key not in included_keys:
            excluded_list.append(v)

    print("=" * 95)
    print(f"🔍 【Cross-Modal 排除樣本深度診斷報告 (總樣本: {len(all_videos)} | 已納入: {len(included_keys)} | 被排除: {len(excluded_list)})】")
    print("=" * 95)

    if not excluded_list:
        print("🎉 沒有任何影片被排除！")
        return

    diag_rows = []
    for idx, v in enumerate(excluded_list, start=1):
        acc = str(v.get("account_id", "")).strip()
        vid = str(v.get("video_id", "")).strip()
        tier = str(v.get("video_classification", "unknown")).lower()
        dur_sec = int(v.get("duration_seconds") or 0)
        alt_stem = alias_map.get((acc.lower(), vid.lower()), vid)

        acc_dir = reels_dir / acc
        npz_path = acc_dir / f"{alt_stem}_multimodal_seq.npz"
        qc_path = acc_dir / f"{alt_stem}_qc.json"

        npz_exists = npz_path.exists()
        qc_exists = qc_path.exists()

        qc_seq_len = 0
        valid_face_secs = 0
        valid_face_ratio = 0.0
        audio_std = np.nan
        face_raw_std = np.nan
        face_interp_std = np.nan
        qc_json_valid_transitions = 0
        qc_json_total_transitions = 0

        if npz_exists:
            try:
                npz = np.load(npz_path)
                qc_seq = npz["qc_roi_seq"]
                aud_seq = npz["audio_seq"]
                qc_mask = npz["qc_valid_mask"]

                qc_seq_len = int(len(qc_seq))
                valid_idx = np.where(qc_mask > 0.5)[0]
                valid_face_secs = int(len(valid_idx))
                valid_face_ratio = round(valid_face_secs / max(1, qc_seq_len), 4)

                if len(aud_seq) > 0:
                    audio_std = float(np.std(aud_seq[:, 0]))

                face_raw = 0.6 * qc_seq[:, 10] + 0.4 * qc_seq[:, 18]
                if valid_face_secs > 0:
                    face_raw_std = float(np.std(face_raw[valid_idx]))
                    x_all = np.arange(qc_seq_len)
                    face_interp = np.interp(x_all, valid_idx, face_raw[valid_idx])
                    face_interp_std = float(np.std(face_interp))
            except Exception as e:
                print(f"  [讀取 .npz 異常] {npz_path}: {e}")

        if qc_exists:
            try:
                with open(qc_path, "r", encoding="utf-8") as qf:
                    q_data = json.load(qf)
                trs = q_data.get("consecutive_qc_metrics", [])
                qc_json_total_transitions = len(trs)
                qc_json_valid_transitions = sum(1 for t in trs if t.get("valid_transition", False))
            except Exception:
                pass

        # 判定觸發了哪一項過濾條件
        reasons = []
        if not npz_exists and not qc_exists:
            reasons.append("檔案缺失 (找不到 .npz 與 _qc.json)")
        if dur_sec < 5 or (npz_exists and qc_seq_len < 5):
            reasons.append(f"可能性 B [時長過短]: duration={dur_sec}s, seq_len={qc_seq_len} (< 5s)")
        if npz_exists and valid_face_secs < 3:
            reasons.append(f"可能性 A [有效人臉秒數不足]: valid_face_secs={valid_face_secs} (< 3s, 僅 {qc_json_valid_transitions}/{qc_json_total_transitions} 有效過渡)")
        if not np.isnan(audio_std) and audio_std < 1e-8:
            reasons.append(f"可能性 C [音訊序列為常數]: audio_std={audio_std:.2e} (< 1e-8)")
        if not np.isnan(face_interp_std) and face_interp_std < 1e-8:
            reasons.append(f"可能性 C [臉部序列為常數]: face_std={face_interp_std:.2e} (< 1e-8)")
        if not reasons:
            reasons.append("其他原因 (請檢查 JSON 欄位結構)")

        reason_str = " ; ".join(reasons)

        print(f"\n[{idx}] 帳號: {acc} | 影片: {vid} (Tier: {tier.upper()})")
        print(f"    • 檔案狀態:   .npz={'存在' if npz_exists else '缺失'} | _qc.json={'存在' if qc_exists else '缺失'}")
        print(f"    • 影片時長:   duration_seconds = {dur_sec}s | npz_seq_len = {qc_seq_len}")
        print(f"    • 有效人臉:   valid_face_secs = {valid_face_secs}s ({valid_face_ratio*100:.1f}%) | QC有效過渡幀 = {qc_json_valid_transitions}/{qc_json_total_transitions}")
        print(f"    • 序列波動:   audio_std = {audio_std:.6f} | face_raw_std = {face_raw_std:.6f} | face_interp_std = {face_interp_std:.6f}")
        print(f"    • 排除主因:   👉 {reason_str}")

        diag_rows.append({
            "account_id": acc,
            "video_id": vid,
            "tier": tier,
            "duration_seconds": dur_sec,
            "npz_exists": npz_exists,
            "qc_exists": qc_exists,
            "npz_seq_len": qc_seq_len,
            "valid_face_secs": valid_face_secs,
            "valid_face_ratio": valid_face_ratio,
            "qc_valid_transitions": qc_json_valid_transitions,
            "qc_total_transitions": qc_json_total_transitions,
            "audio_std": round(audio_std, 6) if not np.isnan(audio_std) else None,
            "face_raw_std": round(face_raw_std, 6) if not np.isnan(face_raw_std) else None,
            "face_interp_std": round(face_interp_std, 6) if not np.isnan(face_interp_std) else None,
            "excluded_reason": reason_str,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diag_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 95)
    print(f"[*] 診斷明細表已匯出至: {out_csv.resolve()}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="診斷 Cross-Modal 分析中被排除的 4 支影片")
    parser.add_argument("--audio_json", type=str, default=str(DEFAULT_AUDIO_JSON))
    parser.add_argument("--cm_csv", type=str, default=str(DEFAULT_CM_CSV))
    parser.add_argument("--class_json", type=str, default=str(DEFAULT_CLASS_JSON))
    parser.add_argument("--reels_dir", type=str, default=str(NEW_REELS_DIR))
    parser.add_argument("--out_csv", type=str, default=str(OUTPUT_CSV))
    args = parser.parse_args()

    diagnose_excluded_videos(
        audio_json_path=Path(args.audio_json),
        cm_csv_path=Path(args.cm_csv),
        class_json_path=Path(args.class_json),
        reels_dir=Path(args.reels_dir),
        out_csv=Path(args.out_csv),
    )
