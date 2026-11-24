#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Modal Saliency: 視聽時空協同對齊分析系統 (Mask-Aware 全量對齊完整版)
---------------------------------------------------------------------------
功能特點：
1. 路徑與檔案對齊最新架構：
   - 預設掃描 `NEW_REELS_DIR`（自動排除 `_quarantined*` 隔離資料夾）。
   - 預設讀取 `AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json"` 與 `DATA_PROC_DIR / "reels_classification_summary.json"`。
2. Mask-Aware 遮罩同步計算（徹底解決 B-roll 假插值稀釋視聽相關性的問題）：
   - 直接從 `*_multimodal_seq.npz` 提取真實逐秒序列，並只保留 `qc_valid_mask > 0.5`（講者實際露臉）的秒數切片計算相關性與峰值重合率。
   - 雙通道臉部動態對齊：
     * 主指標：`mouth_zygomaticus`（口周嘴角發聲動態）× `audio_seq`（音訊吸睛度）的物理發聲同步率。
     * 輔助指標：綜合表情張力（`mouth + nasolabial + eyes`）× `semantic_score`（純語意密度）的重點強調耦合度。
3. 完整整合 `shares`、`SRV`（分享率）、`ERV`、`VFR`、`CLR`、`SLR` 與 `video_quality_score` 欄位，輸出統計檢定報告、CSV 與高解析圖表。
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import seaborn as sns

# 確保無論從根目錄或 crossModalAnalysis 子目錄執行都能正確匯入 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AUDIO_PROC_DIR,
    DATA_PROC_DIR,
    CROSS_MODAL_PROC_DIR,
    NEW_REELS_DIR,
)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_AUDIO_JSON = str(AUDIO_PROC_DIR / "dual_stream_audio_saliency_all_records.json")
DEFAULT_CLASS_JSON = str(DATA_PROC_DIR / "reels_classification_summary.json")
OUTPUT_DIR = str(CROSS_MODAL_PROC_DIR)

# 納入鼻唇溝應力交界帶 (#458, #461, #370) 與口周/眼周關鍵表情點
EXPRESSION_KEY_VERTICES = {
    0, 13, 14, 17, 61, 291, 33, 263, 130, 359,
    458, 461, 370, 238, 241, 141,
}
TIER_ORDER = ["good", "neutral", "bad"]
TIER_COLORS = {"good": "#2ca02c", "neutral": "#ff7f0e", "bad": "#d62728"}


def build_media_lookup_dict(root_dir: str) -> tuple[dict, dict]:
    """
    掃描 NEW_REELS_DIR 下的所有 *_multimodal_seq.npz 與 *_qc.json，建立雙向索引表。
    自動跳過 _quarantined* 隔離目錄。
    """
    print(f"[*] 正在全盤掃描 {root_dir} 下的多模態張量與 QC 檔案...")
    npz_lookup = {}
    qc_lookup = {}

    for dirpath, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_quarantined")]
        acc_folder = os.path.basename(dirpath).lower().strip()

        for fname in filenames:
            f_low = fname.lower()
            full_p = os.path.join(dirpath, fname)
            if f_low.endswith("_multimodal_seq.npz"):
                stem = f_low[:-19]
                npz_lookup[(acc_folder, stem)] = full_p
            elif f_low.endswith("_qc.json"):
                stem = f_low[:-8]
                qc_lookup[(acc_folder, stem)] = full_p

    print(f"[*] 掃描完成：找到 {len(npz_lookup)} 個 .npz 時序張量檔，{len(qc_lookup)} 個 _qc.json 檔。")
    return npz_lookup, qc_lookup


def load_class_metadata_map(class_json_path: str) -> tuple[dict, dict]:
    """
    讀取 video_classification_summary.json，建立別名映射與完整效能指標查找表。
    """
    alias_map = {}
    meta_map = {}
    if not class_json_path or not os.path.exists(class_json_path):
        return alias_map, meta_map

    try:
        with open(class_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        videos = data.get("videos", data) if isinstance(data, dict) else data
        for item in videos:
            if not isinstance(item, dict):
                continue
            acc = str(item.get("account_id", "")).lower().strip()
            vid = str(item.get("video_id", "")).lower().strip()
            loc = str(item.get("local_index_name", "")).lower().strip()
            if acc and vid:
                meta_map[(acc, vid)] = item
            if acc and loc:
                meta_map[(acc, loc)] = item
            if acc and vid and loc:
                alias_map[(acc, vid)] = loc
    except Exception as e:
        print(f"  [提示] 讀取分類總表異常 ({e})")
    return alias_map, meta_map


def extract_timeseries_from_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    從 .npz 提取真實露臉秒數的視聽序列（嚴格過濾 B-roll 無臉遮擋秒數，拒絕假性線性插值）：
    - qc_roi_seq 欄位索引：
        0..3: brow_forehead | 4..7: eyes_duchenne | 8..11: nasolabial_shear | 12..15: mouth_zygomaticus | 16..19: global_face
    回傳: (audio_eng_masked, audio_sem_masked, mouth_face_masked, expr_face_masked)
    """
    try:
        data = np.load(npz_path)
        qc_seq = data["qc_roi_seq"]       # (T, 20)
        aud_seq = data["audio_seq"]       # (T, 5)
        qc_mask = data["qc_valid_mask"]   # (T,)

        valid_idx = np.where(qc_mask > 0.5)[0]
        if len(valid_idx) < 4:
            return None, None, None, None

        # 1. 口周與嘴角發聲動態 (mouth_zygomaticus mu_mean, col 12) —— 與語音能量物理同步
        mouth_ts = qc_seq[valid_idx, 12].astype(np.float64)

        # 2. 綜合表情張力 (0.4*mouth + 0.3*nasolabial + 0.3*eyes, 皆取 mu_mean) —— 與重點強調同步
        expr_ts = (
            0.4 * qc_seq[valid_idx, 12]
            + 0.3 * qc_seq[valid_idx, 8]
            + 0.3 * qc_seq[valid_idx, 4]
        ).astype(np.float64)

        # 僅取對應露臉秒數的音訊吸睛度 (col 0) 與語意密度 (col 2)
        audio_eng_ts = aud_seq[valid_idx, 0].astype(np.float64)
        audio_sem_ts = aud_seq[valid_idx, 2].astype(np.float64)

        # 輕度 3 點加權移動平均消除單幀閃爍雜訊
        if len(valid_idx) >= 6:
            kernel = np.array([0.25, 0.5, 0.25])
            mouth_ts = np.convolve(mouth_ts, kernel, mode="same")
            expr_ts = np.convolve(expr_ts, kernel, mode="same")
            audio_eng_ts = np.convolve(audio_eng_ts, kernel, mode="same")
            audio_sem_ts = np.convolve(audio_sem_ts, kernel, mode="same")

        return audio_eng_ts, audio_sem_ts, mouth_ts, expr_ts
    except Exception:
        return None, None, None, None


def extract_facial_deformation_timeseries_fallback(qc_json_path: str, target_duration_sec: int):
    """備援函式：若尚未執行 build_multimodal_roi_dataset-v2.py，直接從 *_qc.json 提取"""
    try:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = json.load(f)
    except Exception:
        return None

    transitions = qc_data.get("consecutive_qc_metrics", [])
    if not transitions or len(transitions) < 3:
        return None

    raw_frame_series = []
    for trans in transitions:
        if not trans.get("valid_transition", False):
            raw_frame_series.append(np.nan)
            continue
        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            raw_frame_series.append(np.nan)
            continue

        key_mus, all_mus = [], []
        for fm in facet_metrics:
            mu_val = float(fm.get("abs_mu", 0.0))
            if mu_val > 1e-6:
                all_mus.append(mu_val)
                verts = fm.get("vertices", [])
                if any(v in EXPRESSION_KEY_VERTICES for v in verts):
                    key_mus.append(mu_val)

        if key_mus:
            raw_frame_series.append(float(np.mean(key_mus)))
        elif all_mus:
            raw_frame_series.append(float(np.mean(all_mus)))
        else:
            raw_frame_series.append(np.nan)

    arr = np.array(raw_frame_series, dtype=np.float64)
    valid_mask = ~np.isnan(arr)
    if np.sum(valid_mask) < 3:
        return None

    x_all = np.arange(len(arr))
    arr_filled = np.interp(x_all, x_all[valid_mask], arr[valid_mask])
    target_x = np.linspace(0, len(arr_filled) - 1, num=target_duration_sec)
    return np.interp(target_x, x_all, arr_filled)


def compute_cross_modal_metrics(audio_saliency_series: np.ndarray, face_series: np.ndarray):
    """計算單部影片的視聽協同對齊指標 (Pearson r, Spearman rho, 最優延遲 Lag, 峰值重合率)"""
    if audio_saliency_series is None or face_series is None:
        return None

    min_len = min(len(audio_saliency_series), len(face_series))
    if min_len < 4:
        return None

    a = audio_saliency_series[:min_len].astype(np.float64)
    f = face_series[:min_len].astype(np.float64)

    if np.std(a) < 1e-8 or np.std(f) < 1e-8:
        return None

    r0, p0 = stats.pearsonr(a, f)
    rho0, _ = stats.spearmanr(a, f)

    max_lag = min(5, max(1, min_len // 3))
    a_norm = (a - np.mean(a)) / (np.std(a) * min_len)
    f_norm = (f - np.mean(f)) / np.std(f)
    cc = signal.correlate(f_norm, a_norm, mode="full")
    lags = signal.correlation_lags(len(f_norm), len(a_norm), mode="full")

    valid_mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_cc = cc[valid_mask]
    valid_lags = lags[valid_mask]

    best_idx = int(np.argmax(valid_cc))
    max_corr = float(valid_cc[best_idx])
    best_lag = int(valid_lags[best_idx])

    a_th = np.percentile(a, 75)
    f_th = np.percentile(f, 75)
    a_peaks = np.where(a >= a_th)[0]
    f_peaks = set(np.where(f >= f_th)[0])

    coincidence_count = sum(
        1 for ap in a_peaks if any((ap + delta) in f_peaks for delta in [-1, 0, 1])
    )
    peak_coincidence_rate = coincidence_count / len(a_peaks) if len(a_peaks) > 0 else 0.0

    return {
        "sync_pearson_r": float(r0),
        "sync_pearson_p": float(p0),
        "sync_spearman_rho": float(rho0),
        "optimal_lag_seconds": float(best_lag),
        "optimal_lag_max_corr": float(max_corr),
        "peak_coincidence_rate": float(peak_coincidence_rate),
    }


def run_cross_modal_experiment(
    audio_summary_json: str,
    class_json_path: str = DEFAULT_CLASS_JSON,
    root_dir: str = str(NEW_REELS_DIR),
    output_dir: str = OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 88)
    print("【Cross-Modal Saliency v2: 視聽時空協同對齊分析系統 (Mask-Aware 遮罩同步 + SRV)】")
    print(f"  - 音訊彙總檔: {audio_summary_json}")
    print(f"  - 分類指標檔: {class_json_path}")
    print(f"  - Reels 目錄: {root_dir}")
    print("=" * 88 + "\n")

    if not os.path.exists(audio_summary_json):
        raise FileNotFoundError(f"找不到音訊摘要檔案: {audio_summary_json}")

    with open(audio_summary_json, "r", encoding="utf-8") as f:
        audio_data = json.load(f)

    alias_map, meta_map = load_class_metadata_map(class_json_path)
    npz_lookup, qc_lookup = build_media_lookup_dict(root_dir)

    video_items = audio_data.get("videos", [])
    records = []

    cnt_total = len(video_items)
    cnt_npz_used = 0
    cnt_fallback_used = 0
    cnt_metrics_ok = 0

    for v_info in video_items:
        tier = str(v_info.get("video_classification", "")).lower()
        if tier not in TIER_ORDER:
            continue

        acc_id = str(v_info.get("account_id", "")).strip()
        vid_id = str(v_info.get("video_id", "")).strip()
        acc_low, vid_low = acc_id.lower(), vid_id.lower()
        alt_vid = alias_map.get((acc_low, vid_low), vid_low)

        duration = int(v_info.get("duration_seconds") or 0)
        if duration < 5:
            continue

        # 優先讀取 .npz 真實逐秒序列 (Mask-Aware 剔除 B-roll 假插值)
        npz_path = npz_lookup.get((acc_low, vid_low)) or npz_lookup.get((acc_low, alt_vid))
        audio_ts, sem_ts, mouth_ts, expr_ts = (None, None, None, None)

        if npz_path:
            audio_ts, sem_ts, mouth_ts, expr_ts = extract_timeseries_from_npz(npz_path)
            if audio_ts is not None and mouth_ts is not None:
                cnt_npz_used += 1

        # 若無 .npz 則退回讀取單片 *_qc.json + *_audio_saliency.json
        if audio_ts is None or mouth_ts is None:
            qc_path = qc_lookup.get((acc_low, vid_low)) or qc_lookup.get((acc_low, alt_vid))
            if not qc_path:
                continue
            fallback_ts = extract_facial_deformation_timeseries_fallback(qc_path, target_duration_sec=duration)
            if fallback_ts is None:
                continue
            mouth_ts = fallback_ts
            expr_ts = fallback_ts

            aud_json_p = qc_path.replace("_qc.json", "_audio_saliency.json")
            if os.path.exists(aud_json_p):
                with open(aud_json_p, "r", encoding="utf-8") as af:
                    a_detail = json.load(af)
                sbs = a_detail.get("second_by_second_metrics", [])
                audio_ts = np.array([float(x.get("engagement_score", 0.0)) for x in sbs], dtype=np.float64)
                sem_ts = np.array([float(x.get("semantic_score", 0.0)) for x in sbs], dtype=np.float64)
            else:
                base_score = float(v_info.get("mean_engagement_score") or 0.3)
                max_score = float(v_info.get("max_engagement_score") or 0.8)
                audio_ts = np.full(duration, base_score, dtype=np.float64)
                for p_sec in v_info.get("top_peak_seconds", []):
                    p_idx = int(p_sec) - 1 if int(p_sec) >= duration else int(p_sec)
                    if 0 <= p_idx < duration:
                        audio_ts[p_idx] = max_score
                sem_ts = audio_ts.copy()
            cnt_fallback_used += 1

        # 1. 主指標：口周嘴角發聲動態 (mouth_ts) x 音訊吸睛度 (audio_ts)
        cm_metrics = compute_cross_modal_metrics(audio_ts, mouth_ts)
        if cm_metrics is None:
            continue

        # 2. 輔助指標：綜合表情張力 (expr_ts) x 語意密度 (sem_ts)
        sem_cm = compute_cross_modal_metrics(sem_ts, expr_ts) if (sem_ts is not None and expr_ts is not None) else None
        sem_sync_r = sem_cm["sync_pearson_r"] if sem_cm else np.nan
        sem_max_corr = sem_cm["optimal_lag_max_corr"] if sem_cm else np.nan
        sem_peak_coinc = sem_cm["peak_coincidence_rate"] if sem_cm else np.nan

        cnt_metrics_ok += 1
        c_meta = meta_map.get((acc_low, vid_low)) or meta_map.get((acc_low, alt_vid)) or {}

        records.append({
            "account_id": acc_id,
            "video_id": vid_id,
            "tier": tier,
            "video_quality_score": float(v_info.get("video_quality_score") or c_meta.get("video_quality_score") or 0.0),
            "shares": float(v_info.get("shares") if v_info.get("shares") is not None else (c_meta.get("shares") or 0.0)),
            "VFR": float(v_info.get("VFR") if v_info.get("VFR") is not None else (c_meta.get("VFR") or 0.0)),
            "ERV": float(v_info.get("ERV") if v_info.get("ERV") is not None else (c_meta.get("ERV") or 0.0)),
            "CLR": float(v_info.get("CLR") if v_info.get("CLR") is not None else (c_meta.get("CLR") or 0.0)),
            "SRV": float(v_info.get("SRV") if v_info.get("SRV") is not None else (c_meta.get("SRV") or 0.0)),
            "SLR": float(v_info.get("SLR") if v_info.get("SLR") is not None else (c_meta.get("SLR") or 0.0)),
            **cm_metrics,
            "semantic_facial_sync_r": float(sem_sync_r),
            "semantic_facial_max_corr": float(sem_max_corr),
            "semantic_facial_peak_coinc": float(sem_peak_coinc),
        })

    print("-" * 65)
    print("【配對與序列來源診斷統計】")
    print(f"  - JSON 總影片數:                 {cnt_total}")
    print(f"  - 使用 .npz Mask-Aware 序列:     {cnt_npz_used} 支")
    print(f"  - 使用 JSON 備援讀取:            {cnt_fallback_used} 支")
    print(f"  - 成功計算跨模態指標數:          {cnt_metrics_ok} 支")
    print("-" * 65 + "\n")

    df = pd.DataFrame(records)
    if len(df) < 10:
        print(f"[錯誤] 有效樣本數過少 ({len(df)} 支)，無法進行統計檢定。")
        return

    tier_counts = {t: len(df[df["tier"] == t]) for t in TIER_ORDER}
    cm_fields = [
        ("sync_pearson_r", "口周-音訊同步相關性 (Mouth-Audio Correlation r)", "衡量講者嘴部發聲動態與聲音吸睛度的同頻程度"),
        ("peak_coincidence_rate", "顯著性峰值重合率 (Peak Coincidence %)", "衡量聲音高潮時口周表情同步爆發的比例"),
        ("optimal_lag_max_corr", "最佳延遲下的最大相關性 (Max Lagged Corr)", "允許 ±5s 時間平移後的潛在視聽耦合度"),
        ("semantic_facial_max_corr", "語意-綜合表情最大耦合度 (Semantic-Expr Max Corr)", "講者語意密度與全表情肌群的協調度"),
        ("semantic_facial_peak_coinc", "語意-綜合表情峰值重合率 (Semantic-Expr Peak %)", "講者輸出高資訊量金句時的微表情同步爆發比例"),
        ("optimal_lag_seconds", "最優視聽延遲時間 (Optimal Lag s)", "臉部表情落後(+)或超前(-)於聲音的秒數"),
    ]

    report_lines = []
    report_lines.append("=" * 92)
    report_lines.append("【Cross-Modal Saliency v2 跨模態視聽協同統計檢定報告 (Mask-Aware + Shares / SRV)】")
    report_lines.append("=" * 92)
    for t in TIER_ORDER:
        report_lines.append(f"  - {t.upper()} Tier: {tier_counts[t]} 支影片")
    report_lines.append("-" * 92 + "\n")

    for m_key, m_title, m_desc in cm_fields:
        report_lines.append(f"▶ 指標：{m_title} — {m_desc}")
        report_lines.append("-" * 82)

        g_vals = df[df["tier"] == "good"][m_key].dropna().values
        n_vals = df[df["tier"] == "neutral"][m_key].dropna().values
        b_vals = df[df["tier"] == "bad"][m_key].dropna().values

        for t, vals in [("good", g_vals), ("neutral", n_vals), ("bad", b_vals)]:
            m = np.mean(vals) if len(vals) > 0 else 0.0
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            med = np.median(vals) if len(vals) > 0 else 0.0
            report_lines.append(f"  [{t.upper():<7}] Mean = {m:.4f} ± {s:.4f} | Median = {med:.4f}")

        if min(len(g_vals), len(n_vals), len(b_vals)) >= 2:
            f_stat, p_anova = stats.f_oneway(g_vals, n_vals, b_vals)
            t_gb, p_welch = stats.ttest_ind(g_vals, b_vals, equal_var=False)
            u_gb, p_mann = stats.mannwhitneyu(g_vals, b_vals, alternative="two-sided")

            nx, ny = len(g_vals), len(b_vals)
            pooled_std = np.sqrt(((nx - 1) * np.var(g_vals, ddof=1) + (ny - 1) * np.var(b_vals, ddof=1)) / (nx + ny - 2))
            d_val = (np.mean(g_vals) - np.mean(b_vals)) / pooled_std if pooled_std > 0 else 0.0

            report_lines.append(f"  [跨組 ANOVA]     F = {f_stat:.4f}, p = {p_anova:.4e}")
            report_lines.append(f"  [GOOD vs. BAD]   Welch's t = {t_gb:.4f}, p = {p_welch:.4e} | Cohen's d = {d_val:.4f}")
            report_lines.append(f"                   Mann-Whitney U = {u_gb:.1f}, p = {p_mann:.4e}")

            sub_srv = df[["SRV", m_key]].dropna()
            sp_srv, p_srv = stats.spearmanr(sub_srv["SRV"], sub_srv[m_key])
            report_lines.append(f"  [vs SRV 分享率]  Spearman rho = {sp_srv:+.4f}, p = {p_srv:.4e}")
        report_lines.append("")

    report_path = os.path.join(output_dir, "cross_modal_saliency_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    df_csv = os.path.join(output_dir, "cross_modal_dataset.csv")
    df.to_csv(df_csv, index=False, encoding="utf-8-sig")

    print("\n".join(report_lines))
    print(f"[+] 報告已存至: {os.path.abspath(report_path)}")
    print(f"[+] 資料集已存至: {os.path.abspath(df_csv)}")

    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    sns.boxplot(
        data=df,
        x="tier",
        y="optimal_lag_max_corr",
        hue="tier",
        order=TIER_ORDER,
        palette=TIER_COLORS,
        legend=False,
        width=0.45,
        fliersize=0,
        boxprops=dict(alpha=0.75),
        ax=axes[0],
    )
    sns.stripplot(
        data=df,
        x="tier",
        y="optimal_lag_max_corr",
        order=TIER_ORDER,
        color="black",
        alpha=0.18,
        jitter=0.2,
        size=3.5,
        ax=axes[0],
    )
    axes[0].set_title("視聽最佳延遲耦合度 (Max Lagged Correlation)\nacross Video Tiers (Mask-Aware)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Max Lagged Correlation", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    sns.scatterplot(
        data=df,
        x="video_quality_score",
        y="optimal_lag_max_corr",
        hue="tier",
        palette=TIER_COLORS,
        alpha=0.45,
        s=28,
        ax=axes[1],
    )
    sns.regplot(
        data=df,
        x="video_quality_score",
        y="optimal_lag_max_corr",
        scatter=False,
        color="darkblue",
        line_kws={"linewidth": 2.0, "linestyle": "--"},
        ax=axes[1],
    )
    r_val, p_val = stats.pearsonr(df["video_quality_score"], df["optimal_lag_max_corr"])
    axes[1].set_title(f"影片品質得分 vs 視聽最大耦合度\nPearson r = {r_val:.4f} (p = {p_val:.4e})", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Max Lagged Correlation", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_png = os.path.join(output_dir, "cross_modal_alignment_comparison.png")
    plt.savefig(plot_png, bbox_inches="tight")
    plt.close()
    print(f"[+] 對比圖表已生成至: {os.path.abspath(plot_png)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 Cross-Modal Saliency v2 視聽協同對齊分析 (Mask-Aware)")
    parser.add_argument("--audio_json", type=str, default=DEFAULT_AUDIO_JSON, help="音訊彙總 JSON 檔路徑")
    parser.add_argument("--class_json", type=str, default=DEFAULT_CLASS_JSON, help="分類與指標總表 JSON 路徑")
    parser.add_argument("--dir", type=str, default=str(NEW_REELS_DIR), help="Reels 根目錄路徑 (預設 NEW_REELS_DIR)")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR, help="結果輸出目錄")
    args = parser.parse_args()

    run_cross_modal_experiment(
        audio_summary_json=args.audio_json,
        class_json_path=args.class_json,
        root_dir=args.dir,
        output_dir=args.out_dir,
    )
