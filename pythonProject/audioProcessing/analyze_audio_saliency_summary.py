import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文字型與負號顯示
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TARGET_DIR = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\audioProcessing"
DEFAULT_JSON_NAME = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProjectlocal_transcript_audio_saliency_summary_all_tiers.json"

METRICS = [
    ("mean_engagement_score", "全片平均吸睛度 (Mean Engagement)"),
    ("hook_first_5s_score", "前5秒開頭吸睛度 (Hook Score)"),
    ("mean_acoustic_score", "純聲學表現得分 (Acoustic Score)"),
    ("mean_semantic_score", "純語意密度得分 (Semantic Score)"),
    ("max_engagement_score", "單片峰值高潮度 (Max Score)"),
    ("std_engagement_score", "單片內部起伏標準差 (Score Std)"),
    ("duration_seconds", "影片時長 (Duration s)"),
]

TIER_ORDER = ["good", "neutral", "bad"]
TIER_COLORS = {"good": "#2ca02c", "neutral": "#ff7f0e", "bad": "#d62728"}


def cohen_d(x, y):
    """計算獨立雙樣本 Cohen's d 效應量"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def locate_input_json(folder_path=TARGET_DIR, file_name=DEFAULT_JSON_NAME):
    """確認輸入檔案路徑，若無精確名稱則自動搜尋替代的 summary json"""
    exact_path = os.path.join(folder_path, file_name)
    if os.path.exists(exact_path):
        return exact_path

    # Fallback 搜尋
    candidates = glob.glob(os.path.join(folder_path, "*audio_saliency_summary*.json"))
    if candidates:
        print(f"[提示] 未找到 {file_name}，自動選用候選檔案: {candidates[0]}")
        return candidates[0]

    # 若在當前目錄下有
    if os.path.exists(file_name):
        return file_name

    raise FileNotFoundError(f"在 {folder_path} 目錄下找不到可供分析的 JSON 檔案。")


def load_and_preprocess_data(json_path):
    """讀取 JSON 並將每支影片的資料扁平化為 DataFrame"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_videos = data.get("videos", [])
    records = []
    for v in raw_videos:
        tier = str(v.get("video_classification", "")).lower()
        if tier not in TIER_ORDER:
            # 略過未分類或邊界異常值
            continue

        rec = {
            "account_id": v.get("account_id"),
            "video_id": v.get("video_id"),
            "tier": tier,
            "video_quality_score": float(v.get("video_quality_score") or 0.0),
            "raw_residual": float(v.get("raw_residual") or 0.0),
            "duration_seconds": float(v.get("duration_seconds") or 0.0),
            "mean_engagement_score": float(v.get("mean_engagement_score")) if v.get(
                "mean_engagement_score") is not None else np.nan,
            "hook_first_5s_score": float(v.get("hook_first_5s_score")) if v.get(
                "hook_first_5s_score") is not None else np.nan,
            "mean_acoustic_score": float(v.get("mean_acoustic_score")) if v.get(
                "mean_acoustic_score") is not None else np.nan,
            "mean_semantic_score": float(v.get("mean_semantic_score")) if v.get(
                "mean_semantic_score") is not None else np.nan,
            "max_engagement_score": float(v.get("max_engagement_score")) if v.get(
                "max_engagement_score") is not None else np.nan,
            "std_engagement_score": float(v.get("std_engagement_score")) if v.get(
                "std_engagement_score") is not None else np.nan,
            "mean_saliency_gradient": float(v.get("mean_saliency_gradient")) if v.get(
                "mean_saliency_gradient") is not None else np.nan,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    return df, data.get("tier_group_statistics", {})


def run_statistical_analysis(df, output_folder):
    """執行 ANOVA、Kruskal-Wallis、Welch's t-test 與相關性檢定，匯出報表與 CSV"""
    report_lines = []
    report_lines.append("=" * 88)
    report_lines.append("【多模態短影音音訊吸睛度 (Audio Saliency) 統計檢定與深度分析報告】")
    report_lines.append("=" * 88)
    report_lines.append(f"總分析有效影片數: {len(df)} 支")
    for t in TIER_ORDER:
        cnt = len(df[df["tier"] == t])
        report_lines.append(f"  - {t.upper()} 組: {cnt} 支 ({cnt / len(df) * 100:.1f}%)")
    report_lines.append("-" * 88 + "\n")

    summary_table_rows = []

    for metric_key, metric_title in METRICS:
        report_lines.append(f"▶ 指標：{metric_title} [{metric_key}]")
        report_lines.append("-" * 75)

        sub_df = df[["tier", metric_key]].dropna()
        g_vals = sub_df[sub_df["tier"] == "good"][metric_key].values
        n_vals = sub_df[sub_df["tier"] == "neutral"][metric_key].values
        b_vals = sub_df[sub_df["tier"] == "bad"][metric_key].values

        # 描述性統計
        stats_dict = {}
        for t, vals in [("good", g_vals), ("neutral", n_vals), ("bad", b_vals)]:
            m = np.mean(vals)
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            med = np.median(vals)
            stats_dict[t] = (m, s, med)
            report_lines.append(f"  [{t.upper():<7}] (N={len(vals):>3}): Mean = {m:.4f} ± {s:.4f} | Median = {med:.4f}")

        # 跨三組整體檢定 (ANOVA & Kruskal-Wallis)
        f_stat, p_anova = stats.f_oneway(g_vals, n_vals, b_vals)
        kw_stat, p_kw = stats.kruskal(g_vals, n_vals, b_vals)

        # 效應量 Eta-squared
        all_vals = np.concatenate([g_vals, n_vals, b_vals])
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(v) * ((np.mean(v) - grand_mean) ** 2) for v in [g_vals, n_vals, b_vals])
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0

        report_lines.append(f"  [跨組檢定] One-way ANOVA:      F = {f_stat:.4f}, p = {p_anova:.4e} (η² = {eta_sq:.4f})")
        report_lines.append(f"             Kruskal-Wallis H:   H = {kw_stat:.4f}, p = {p_kw:.4e}")

        # 核心兩兩對比 (Good vs Bad)
        t_gb, p_gb_welch = stats.ttest_ind(g_vals, b_vals, equal_var=False)
        u_gb, p_gb_mann = stats.mannwhitneyu(g_vals, b_vals, alternative="two-sided")
        d_val = cohen_d(g_vals, b_vals)

        report_lines.append(f"  [兩兩檢定: GOOD vs. BAD]")
        report_lines.append(f"    - Welch's t-test: t = {t_gb:.4f}, p = {p_gb_welch:.4e}")
        report_lines.append(f"    - Mann-Whitney U: U = {u_gb:.1f}, p = {p_gb_mann:.4e}")
        report_lines.append(
            f"    - 效應量 (Cohen's d): d = {d_val:.4f} ({'顯著差異' if p_gb_welch < 0.05 else '不顯著'})")
        report_lines.append("")

        summary_table_rows.append({
            "metric": metric_key,
            "metric_title": metric_title,
            "good_mean": stats_dict["good"][0],
            "good_std": stats_dict["good"][1],
            "neutral_mean": stats_dict["neutral"][0],
            "neutral_std": stats_dict["neutral"][1],
            "bad_mean": stats_dict["bad"][0],
            "bad_std": stats_dict["bad"][1],
            "anova_F": f_stat,
            "anova_p": p_anova,
            "eta_squared": eta_sq,
            "kruskal_p": p_kw,
            "good_vs_bad_t": t_gb,
            "good_vs_bad_p_welch": p_gb_welch,
            "good_vs_bad_cohen_d": d_val,
        })

    # 相關性分析
    report_lines.append("=" * 88)
    report_lines.append("【連續變數相關性分析 (對齊 video_quality_score 殘差分數)】")
    report_lines.append("=" * 88)
    report_lines.append(
        f"{'指標名稱':<25} | {'Pearson r':<12} | {'Pearson p':<14} | {'Spearman rho':<14} | {'Spearman p':<14}")
    report_lines.append("-" * 88)

    corr_df = df.dropna(subset=["video_quality_score"])
    for metric_key, metric_title in METRICS:
        clean_sub = corr_df[["video_quality_score", metric_key]].dropna()
        pr_r, pr_p = stats.pearsonr(clean_sub["video_quality_score"], clean_sub[metric_key])
        sp_r, sp_p = stats.spearmanr(clean_sub["video_quality_score"], clean_sub[metric_key])
        report_lines.append(
            f"{metric_key:<25} | {pr_r:<12.4f} | {pr_p:<14.4e} | {sp_r:<14.4f} | {sp_p:<14.4e}"
        )

    # 寫入文字報表
    report_file = os.path.join(output_folder, "audio_saliency_statistical_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  -> 已產出統計報表: {os.path.abspath(report_file)}")

    # 寫入 CSV 摘要
    summary_df = pd.DataFrame(summary_table_rows)
    summary_csv = os.path.join(output_folder, "tier_comparison_stats.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"  -> 已產出統計對比表格: {os.path.abspath(summary_csv)}")

    # 寫入單片資料集 CSV
    dataset_csv = os.path.join(output_folder, "audio_metrics_dataset.csv")
    df.to_csv(dataset_csv, index=False, encoding="utf-8-sig")
    print(f"  -> 已產出單片特徵 CSV 檔: {os.path.abspath(dataset_csv)}")


def generate_visualizations(df, output_folder):
    """繪製並輸出三大高解析圖表 (Boxplot, Scatter, Correlation Heatmap)"""
    # --------------------------------------------------------------------------
    # 圖一：核心指標的組別對比箱型圖 (Boxplot with Jittered Strip)
    # --------------------------------------------------------------------------
    plot_metrics = [
        ("mean_engagement_score", "全片平均吸睛度", "[0.0 ~ 0.8]"),
        ("hook_first_5s_score", "前5秒開頭 Hook 得分", "[0.0 ~ 0.8]"),
        ("mean_acoustic_score", "純聲學表現得分", "[0.1 ~ 0.5]"),
        ("mean_semantic_score", "純語意密度得分", "[0.0 ~ 1.0]"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    for i, (m_key, m_name, y_lim) in enumerate(plot_metrics):
        ax = axes[i]
        sub = df[["tier", m_key]].dropna()

        # 箱型圖
        sns.boxplot(
            data=sub,
            x="tier",
            y=m_key,
            order=TIER_ORDER,
            palette=TIER_COLORS,
            width=0.45,
            fliersize=0,
            boxprops=dict(alpha=0.75),
            ax=ax,
        )
        # 散點抖動
        sns.stripplot(
            data=sub,
            x="tier",
            y=m_key,
            order=TIER_ORDER,
            color="black",
            alpha=0.18,
            jitter=0.2,
            size=3.5,
            ax=ax,
        )

        ax.set_title(f"{m_name}\n({m_key})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Video Tier", fontsize=10)
        ax.set_ylabel("Score [0, 1]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Multimodal Audio Engagement Metrics across Video Tiers (Good vs. Neutral vs. Bad)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    boxplot_png = os.path.join(output_folder, "audio_metrics_tier_boxplots.png")
    plt.savefig(boxplot_png, bbox_inches="tight")
    plt.close()
    print(f"  -> 已輸出箱型分佈圖: {os.path.abspath(boxplot_png)}")

    # --------------------------------------------------------------------------
    # 圖二：影片品質分數 vs 語意密度 & 吸睛度的迴歸散點圖
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("white")

    scatter_pairs = [
        (axes[0], "mean_semantic_score", "純語意密度得分 (Semantic Density)", "#1f77b4"),
        (axes[1], "mean_engagement_score", "全片平均吸睛度 (Engagement Score)", "#ff7f0e"),
    ]

    for ax, m_key, m_title, dot_color in scatter_pairs:
        clean_sub = df[["video_quality_score", m_key, "tier"]].dropna()
        sns.scatterplot(
            data=clean_sub,
            x="video_quality_score",
            y=m_key,
            hue="tier",
            palette=TIER_COLORS,
            alpha=0.45,
            s=28,
            ax=ax,
        )
        sns.regplot(
            data=clean_sub,
            x="video_quality_score",
            y=m_key,
            scatter=False,
            color="darkblue",
            line_kws={"linewidth": 2.0, "linestyle": "--"},
            ax=ax,
        )
        r_val, p_val = stats.pearsonr(clean_sub["video_quality_score"], clean_sub[m_key])
        ax.set_title(f"{m_title}\nPearson r = {r_val:.4f} (p = {p_val:.4e})", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Video Quality Score (Residualized 0~100)", fontsize=10)
        ax.set_ylabel(m_key, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Tier", loc="upper left")

    plt.tight_layout()
    scatter_png = os.path.join(output_folder, "quality_vs_audio_metrics_scatter.png")
    plt.savefig(scatter_png, bbox_inches="tight")
    plt.close()
    print(f"  -> 已輸出品質相關散點圖: {os.path.abspath(scatter_png)}")

    # --------------------------------------------------------------------------
    # 圖三：音訊特徵相關性矩陣熱力圖 (Correlation Heatmap)
    # --------------------------------------------------------------------------
    feature_cols = [
        "video_quality_score",
        "mean_engagement_score",
        "hook_first_5s_score",
        "mean_acoustic_score",
        "mean_semantic_score",
        "max_engagement_score",
        "duration_seconds",
    ]
    corr_matrix = df[feature_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    fig.patch.set_facecolor("white")
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Spearman Correlation Matrix: Video Quality vs. Audio Metrics", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_png = os.path.join(output_folder, "audio_metrics_correlation_heatmap.png")
    plt.savefig(corr_png, bbox_inches="tight")
    plt.close()
    print(f"  -> 已輸出特徵相關性熱力圖: {os.path.abspath(corr_png)}")


def main():
    parser = argparse.ArgumentParser(description="深度分析音訊顯著性與吸睛度彙總 JSON 檔案")
    parser.add_argument("--dir", type=str, default=TARGET_DIR, help=f"目標工作目錄 (預設: {TARGET_DIR})")
    parser.add_argument("--file", type=str, default=DEFAULT_JSON_NAME, help="指定 JSON 檔名")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    json_path = locate_input_json(folder_path=args.dir, file_name=args.file)

    print("=" * 75)
    print("【正在執行音訊顯著性統計分析任務】")
    print(f"  - 讀取來源: {os.path.abspath(json_path)}")
    print(f"  - 輸出資料夾: {os.path.abspath(args.dir)}")
    print("=" * 75)

    df, tier_stats = load_and_preprocess_data(json_path)
    if df.empty:
        print("[錯誤] 解析後未獲取到任何有效影片記錄。")
        return

    print(f"\n1. 執行統計檢定並匯出報表與數據表...")
    run_statistical_analysis(df, args.dir)

    print(f"\n2. 生成出版級視覺化圖表...")
    generate_visualizations(df, args.dir)

    print("\n" + "=" * 75)
    print("【全部分析已完成，所有報告與圖表皆已存於該目錄中！】")
    print("=" * 75)


if __name__ == "__main__":
    main()