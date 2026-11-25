#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-label Reels Classification by Viral / Reach Goal (VFR + Shares / SRV) -> *_vfr.json
-----------------------------------------------------------------------------------------
核心功能：
1. 讀取 `DATA_PROC_DIR / "reels_classification_summary.json"`（若不存在則退回 `video_classification_summary.json`），
   並可選融合 `AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"` 的 `VFR`、`SRV`、`shares`。
2. 以「破圈傳播導向（Viral / Reach）」重新計算傳播綜合得分：
   - 優先使用 log 轉換後的觀看粉絲比 `ln(1 + VFR)` 為主軸，並結合分享傳播指標（`ln(1 + SRV * 10000)` 或 `ln(1 + shares)`），
     若原始 JSON 僅有 `video_quality_score` 或 `VFR` 欄位則自動相容計算，避免因單一固定權重混合不同經營目標。
3. 輸出至同層資料夾 (`DATA_PROC_DIR`) 的 `reels_classification_summary_vfr.json`，絕不覆蓋原有檔案。
"""

import os
import sys
import json
import math
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if (CURRENT_DIR.parent / "config.py").exists() else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import DATA_PROC_DIR, AUDIO_PROC_DIR
    D_DIR = Path(DATA_PROC_DIR)
    A_DIR = Path(AUDIO_PROC_DIR)
except ImportError:
    D_DIR = CURRENT_DIR
    A_DIR = PROJECT_ROOT / "audioProcessing"

PRIMARY_IN = D_DIR / "reels_classification_summary.json"
FALLBACK_IN = D_DIR / "video_classification_summary.json"
AUDIO_IN = A_DIR / "dual_stream_audio_saliency_all_records.json"
OUT_JSON = D_DIR / "reels_classification_summary_vfr.json"


def safe_float(val, default=0.0):
    try:
        if val is None:
            return float(default)
        f = float(val)
        return float(default) if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return float(default)


def compute_viral_score(item: dict, aud_item: dict) -> float:
    """計算傳播導向分數：側重 VFR（演算法破圈推流）與分享傳播（SRV / shares）。"""
    vfr = max(0.0, safe_float(item.get("VFR") if item.get("VFR") is not None else aud_item.get("VFR"), 0.0))
    srv = max(0.0, safe_float(item.get("SRV") if item.get("SRV") is not None else aud_item.get("SRV"), 0.0))
    shares = max(0.0, safe_float(item.get("shares") if item.get("shares") is not None else aud_item.get("shares"), 0.0))

    # 權重設計：70% 側重 VFR（非粉絲觸及／破圈），30% 側重分享傳播（SRV bps 與 shares 對數）
    log_vfr = math.log1p(vfr * 100.0) if vfr <= 1.0 else math.log1p(vfr)
    log_srv = math.log1p(srv * 10000.0) if srv <= 1.0 else math.log1p(srv)
    log_shares = math.log1p(shares)
    share_signal = 0.6 * log_srv + 0.4 * log_shares

    raw_score = 0.70 * log_vfr + 0.30 * share_signal
    if raw_score > 0.0:
        return float(raw_score)

    # 若無 VFR/SRV/shares 欄位，退回使用 video_quality_score
    return safe_float(item.get("video_quality_score"), 50.0)


def main():
    in_path = PRIMARY_IN if PRIMARY_IN.exists() else FALLBACK_IN
    if not in_path.exists():
        print(f"[-] 找不到分類來源檔: {PRIMARY_IN} 或 {FALLBACK_IN}")
        return

    print("=" * 84)
    print("【重算影片分級標籤 — 傳播導向 (_vfr)】")
    print(f"[*] 來源總表: {in_path}")
    print(f"[*] 輸出檔案: {OUT_JSON}")
    print("=" * 84)

    aud_lookup = {}
    if AUDIO_IN.exists():
        with open(AUDIO_IN, "r", encoding="utf-8") as f:
            adata = json.load(f)
        for v in adata.get("videos", []):
            acc = str(v.get("account_id", "")).lower().strip()
            vid = str(v.get("video_id", "")).lower().strip()
            stem = str(v.get("local_index_name", "")).lower().strip()
            for k in [vid, stem]:
                if acc and k:
                    aud_lookup[(acc, k)] = v

    with open(in_path, "r", encoding="utf-8") as f:
        raw_obj = json.load(f)

    is_dict = isinstance(raw_obj, dict)
    videos = raw_obj.get("videos", raw_obj) if is_dict else raw_obj

    scores = []
    enriched = []
    for item in videos:
        if not isinstance(item, dict):
            continue
        acc = str(item.get("account_id", "")).lower().strip()
        vid = str(item.get("video_id", "")).lower().strip()
        stem = str(item.get("local_index_name", "")).lower().strip()
        a_info = aud_lookup.get((acc, vid)) or aud_lookup.get((acc, stem), {})
        sc = compute_viral_score(item, a_info)
        scores.append(sc)
        enriched.append((item, a_info, sc))

    if not scores:
        print("[-] 沒有可評分的影片紀錄。")
        return

    arr = np.array(scores, dtype=np.float64)
    # 沿用 25% / 75% 分位數切分：前 25% -> good，中間 50% -> neutral，後 25% -> bad
    q_low = float(np.percentile(arr, 25))
    q_high = float(np.percentile(arr, 75))

    out_videos = []
    counts = {"bad": 0, "neutral": 0, "good": 0}
    for item, a_info, sc in enriched:
        new_item = dict(item)
        if sc >= q_high:
            tier = "good"
        elif sc <= q_low:
            tier = "bad"
        else:
            tier = "neutral"

        new_item["vfr_viral_score"] = round(sc, 6)
        new_item["video_quality_score_vfr"] = round(sc, 6)
        new_item["video_classification_vfr"] = tier
        new_item["video_classification"] = tier
        new_item["tier"] = tier
        if "VFR" not in new_item and "VFR" in a_info:
            new_item["VFR"] = a_info.get("VFR")
        if "SRV" not in new_item and "SRV" in a_info:
            new_item["SRV"] = a_info.get("SRV")
        if "shares" not in new_item and "shares" in a_info:
            new_item["shares"] = a_info.get("shares")

        out_videos.append(new_item)
        counts[tier] += 1

    payload = dict(raw_obj) if is_dict else {}
    payload["criterion"] = "vfr_viral_reach"
    payload["percentile_thresholds"] = {"p25": q_low, "p75": q_high}
    payload["tier_counts"] = counts
    payload["videos"] = out_videos

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[+] 分位數門檻: p25={q_low:.6f}, p75={q_high:.6f}")
    print(f"[+] 分級統計: Good={counts['good']}, Neutral={counts['neutral']}, Bad={counts['bad']} (合計 {len(out_videos)} 支)")
    print(f"[+] 已儲存至: {OUT_JSON}")


if __name__ == "__main__":
    main()
