#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 1: Independent Visual Color & Lighting Statistical Analysis (Fixed Indexing & Plotting)
--------------------------------------------------------------------------------------------
修正重點：
1. 採用 (account_id, video_stem) 雙鍵值索引，徹底解決不同帳號同名檔案 (如 video_001) 互相覆蓋導致 Good=0 的問題。
2. 支援雙重讀取 video_classification_summary.json 與 local_transcript_audio_saliency_summary_all_tiers.json。
3. 修正 seaborn.boxplot 的 boxprops=dict(alpha=0.75) 與 hue="tier" 語法。
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from config import VIDEO_PROC_DIR, DATA_PROC_DIR, AUDIO_PROC_DIR, DEFAULT_REELS_DIR

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CLASS_JSON = DATA_PROC_DIR / "video_classification_summary.json"
DEFAULT_AUDIO_JSON = AUDIO_PROC_DIR / "local_transcript_audio_saliency_summary_all_tiers.json"
OUTPUT_DIR = VIDEO_PROC_DIR / "color_lighting_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIER_ORDER = ["good", "neutral", "bad"]
TIER_COLORS = {"good": "#2ca02c", "neutral": "#ff7f0e", "bad": "#d62728"}


def build_color_lighting_lookup(root_dir: Path) -> dict:
    """
    掃描 D:\reels 下所有的 *_color_lighting.json，建立快速索引字典：
    Key: (account_id_lower, video_stem_lower) -> Value: 完整檔案路徑
    """
    print(f"[*] 正在全盤掃描 {root_dir} 下的 *_color_lighting.json 檔案...")
    pattern = str(root_dir / "**" / "*_color_lighting.json")
    all_files = glob.glob(pattern, recursive=True)
    print(f"[*] 磁碟中實際找到 {len(all_files)} 個色彩光影分析檔案。正在建立雙向索引表...")

    lookup = {}
    for orig_p in all_files:
        norm_p = orig_p.replace("\\", "/")
        parts = norm_p.split("/")
        fname = parts[-1].lower()
        vid_stem = fname[:-20] if fname.endswith("_color_lighting.json") else os.path.splitext(fname)[0]

        # 將路徑上所有上層資料夾名稱都註冊為潛在的 account_id
        for folder_part in parts[:-1]:
            acc_cand = folder_part.lower().strip()
            if acc_cand and acc_cand not in ["d:", "reels"]:
                lookup[(acc_cand, vid_stem)] = orig_p
                for sfx in ["_processed", "_face", "_aligned"]:
                    if sfx in vid_stem:
                        lookup[(acc_cand, vid_stem.replace(sfx, ""))] = orig_p
    return lookup


def load_video_metadata() -> list:
    """
    載入影片分類清單 (整合 video_classification_summary.json 與 audio_summary 的別名映射)
    """
    alias_map = {}
    items_list = []

    if DEFAULT_CLASS_JSON.exists():
        with open(DEFAULT_CLASS_JSON, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        videos = c_data.get("videos", c_data) if isinstance(c_data, dict) else c_data
        for item in videos:
            if not isinstance(item, dict):
                continue
            acc = str(item.get("account_id", "")).lower().strip()
            vid = str(item.get("video_id", "")).lower().strip()
            loc = str(item.get("local_index_name", "")).lower().strip()
            if acc and vid and loc:
                alias_map[(acc, vid)] = loc
            items_list.append(item)

    # 若 audio_summary 存在且包含完整品質分數，優先用來補齊清單
    if DEFAULT_AUDIO_JSON.exists():
        with open(DEFAULT_AUDIO_JSON, "r", encoding="utf-8") as f:
            a_data = json.load(f)
        a_videos = a_data.get("videos", [])
        if len(a_videos) >= len(items_list):
            items_list = a_videos

    return items_list, alias_map


def collect_matched_dataset(root_dir: Path) -> pd.DataFrame:
    lookup = build_color_lighting_lookup(root_dir)
    video_items, alias_map = load_video_metadata()

    records = []
    used_paths = set()

    for v_info in video_items:
        tier = str(v_info.get("video_classification", v_info.get("tier", ""))).lower().strip()
        if tier not in TIER_ORDER:
            continue

        acc_id = str(v_info.get("account_id", "")).strip()
        vid_id = str(v_info.get("video_id", "")).strip()
        loc_id = str(v_info.get("local_index_name", "")).strip()
        acc_low, vid_low, loc_low = acc_id.lower(), vid_id.lower(), loc_id.lower()

        # 查找對應的 *_color_lighting.json
        cl_path = lookup.get((acc_low, vid_low))
        if not cl_path and loc_low:
            cl_path = lookup.get((acc_low, loc_low))
        if not cl_path and (acc_low, vid_low) in alias_map:
            alt_vid = alias_map[(acc_low, vid_low)]
            cl_path = lookup.get((acc_low, alt_vid))

        if not cl_path or cl_path in used_paths:
            continue

        try:
            with open(cl_path, "r", encoding="utf-8") as f:
                cl_data = json.load(f)
        except Exception:
            continue

        used_paths.add(cl_path)

        row = {
            "account_id": acc_id,
            "video_id": vid_id,
            "tier": tier,
            "video_quality_score": float(v_info.get("video_quality_score") or 0.0),
            "duration_sec": float(cl_data.get("duration_sec") or 0.0),
        }

        for k, v in cl_data.get("whole_video_summary", {}).items():
            row[f"whole_{k}"] = float(v)
        for k, v in cl_data.get("hook_window_summary", {}).items():
            row[k] = float(v)

        records.append(row)

    df = pd.DataFrame(records)
    print(f"[*] 成功透過 (account_id, video_id) 雙鍵精準配對: {len(df)} 支影片。")
    return df


def main():
    df = collect_matched_dataset(Path(DEFAULT_REELS_DIR))
    if len(df) < 10:
        print("[-] 配對成功的有效樣本數過少，請檢查路徑。")
        return

    tier_counts = {t: len(df[df["tier"] == t]) for t in TIER_ORDER}
    print(f"[*] 各 Tier 樣本數量: {tier_counts}\n")

    features = [
        ("hook_figure_ground_lum_contrast_mean", "前 3 秒前景背景明暗反差", "衡量開頭主體是否從背景跳脫"),
        ("hook_colorfulness_mean", "前 3 秒畫面色彩豐富度", "衡量開頭色彩鮮豔度 vs 單調灰暗"),
        ("hook_face_rms_contrast_mean", "前 3 秒面部光影 RMS 對比度", "衡量臉部打光立體感 vs 死亡平光"),
        ("hook_skin_warmth_cbcr_mean", "前 3 秒膚色血色感 (Cr-Cb)", "衡量膚色紅潤通透 vs 暗沉偏冷"),
        ("whole_figure_ground_lum_contrast_mean", "全片前景背景明暗反差", "全片人物與環境的明度分離度"),
        ("whole_colorfulness_mean", "全片畫面色彩豐富度", "全片平均視覺色彩飽和豐富感"),
        ("whole_face_rms_contrast_mean", "全片面部光影 RMS 對比度", "全片臉部光影立體層次"),
        ("whole_skin_warmth_cbcr_mean", "全片膚色血色感 (Cr-Cb)", "全片講者氣色與暖度"),
    ]

    report = []
    report.append("=" * 85)
    report.append("【獨立分析項目：短影音色彩與光影特徵統計檢定報告】")
    report.append("=" * 85)
    for t in TIER_ORDER:
        report.append(f"  - {t.upper()} Tier: {tier_counts[t]} 支影片")
    report.append("-" * 85 + "\n")

    for f_key, f_name, f_desc in features:
        if f_key not in df.columns:
            continue
        report.append(f"▶ 指標：{f_name} ({f_key}) — {f_desc}")
        report.append("-" * 75)

        g_vals = df[df["tier"] == "good"][f_key].dropna().values
        n_vals = df[df["tier"] == "neutral"][f_key].dropna().values
        b_vals = df[df["tier"] == "bad"][f_key].dropna().values

        for t, vals in [("good", g_vals), ("neutral", n_vals), ("bad", b_vals)]:
            m = np.mean(vals) if len(vals) > 0 else 0.0
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            med = np.median(vals) if len(vals) > 0 else 0.0
            report.append(f"  [{t.upper():<7}] Mean = {m:.4f} ± {s:.4f} | Median = {med:.4f}")

        if min(len(g_vals), len(n_vals), len(b_vals)) >= 2:
            f_stat, p_anova = stats.f_oneway(g_vals, n_vals, b_vals)
            t_gb, p_welch = stats.ttest_ind(g_vals, b_vals, equal_var=False)
            u_gb, p_mann = stats.mannwhitneyu(g_vals, b_vals, alternative="two-sided")

            nx, ny = len(g_vals), len(b_vals)
            pooled_std = np.sqrt(((nx - 1) * np.var(g_vals, ddof=1) + (ny - 1) * np.var(b_vals, ddof=1)) / (nx + ny - 2))
            d_val = (np.mean(g_vals) - np.mean(b_vals)) / pooled_std if pooled_std > 0 else 0.0

            sig = "***" if p_welch < 0.001 else "**" if p_welch < 0.01 else "*" if p_welch < 0.05 else ""
            report.append(f"  [跨組 ANOVA]     F = {f_stat:.4f}, p = {p_anova:.4e}")
            report.append(f"  [GOOD vs. BAD]   Welch's t = {t_gb:.4f}, p = {p_welch:.4e} {sig} | Cohen's d = {d_val:.4f}")
            report.append(f"                   Mann-Whitney U = {u_gb:.1f}, p = {p_mann:.4e}\n")
        else:
            report.append("  [提示] 某組樣本數不足，跳過檢定。\n")

    report_text = "\n".join(report)
    print(report_text)

    report_path = OUTPUT_DIR / "visual_color_lighting_independent_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    csv_path = OUTPUT_DIR / "visual_color_lighting_dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 繪製箱形圖 (修正 Seaborn 新版語法)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor("white")
    plots = [
        ("hook_figure_ground_lum_contrast_mean", "前 3 秒前景背景明暗反差 (Figure-Ground Contrast)", axes[0, 0]),
        ("hook_colorfulness_mean", "前 3 秒畫面色彩豐富度 (Hook Colorfulness)", axes[0, 1]),
        ("hook_face_rms_contrast_mean", "前 3 秒面部光影立體感 (Facial RMS Contrast)", axes[1, 0]),
        ("hook_skin_warmth_cbcr_mean", "前 3 秒膚色血色感 (Skin Warmth Cr-Cb)", axes[1, 1]),
    ]

    for col, title, ax in plots:
        if col not in df.columns:
            continue
        sns.boxplot(
            data=df,
            x="tier",
            y=col,
            hue="tier",
            order=TIER_ORDER,
            palette=TIER_COLORS,
            legend=False,
            width=0.45,
            fliersize=0,
            boxprops=dict(alpha=0.75),
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="tier",
            y=col,
            order=TIER_ORDER,
            color="black",
            alpha=0.2,
            jitter=0.2,
            size=3,
            ax=ax,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Tier", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "visual_color_lighting_distributions.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"[+] 報告已儲存至: {report_path}")
    print(f"[+] 資料集已存至: {csv_path}")
    print(f"[+] 分佈圖已存至: {plot_path}")


if __name__ == "__main__":
    main()