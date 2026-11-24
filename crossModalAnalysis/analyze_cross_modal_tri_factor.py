#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 (v2): Tri-Modal Cross-Dimension Econometric & Feature Importance Analysis
--------------------------------------------------------------------------------
功能升級：
1. 全面涵蓋 6 大影片效能與受眾互動應變數（皆採用 HC3 異質變異數穩健標準誤）：
   - Model 1: video_quality_score  (去偏誤綜合品質殘差得分 0~100)
   - Model 2: log_vfr              (觀看粉絲比 ln(1 + VFR) — 衡量演算法破圈推流力)
   - Model 3: log_erv_bps          (觀看互動率 ln(1 + ERV * 10000) — 衡量觀看後互動轉換)
   - Model 4: log_clr_permille     (留言點讚比 ln(1 + CLR * 1000) — 衡量引發討論與提問深度)
   - Model 5: log_srv_bps          (觀看分享率 ln(1 + SRV * 10000) — 衡量實用價值與社交轉發)
   - Model 6: log_slr_permille     (分享點讚比 ln(1 + SLR * 1000) — 衡量實用工具型 vs 純按讚型)
2. 計算標準化迴歸係數 (Standardized Beta) 與 |z| 統計量顯著性排名。
3. 完整輸出 5 份成果檔案：
   - tri_modal_econometric_regression_report.txt
   - tri_modal_feature_significance_matrix.csv
   - tri_modal_master_dataset.csv
   - tri_modal_correlation_heatmap.png
   - tri_modal_beta_significance_heatmap.png
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CROSS_MODAL_PROC_DIR

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MASTER_CSV = Path(CROSS_MODAL_PROC_DIR) / "master_dataset_results" / "comprehensive_master_dataset.csv"
FALLBACK_MM_CSV = Path(CROSS_MODAL_PROC_DIR) / "multimodal_dataset" / "multimodal_video_level_summary.csv"
FALLBACK_CM_CSV = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_dataset.csv"
OUTPUT_DIR = Path(CROSS_MODAL_PROC_DIR) / "tri_factor_results"


def load_master_dataset(master_csv: Path = MASTER_CSV) -> pd.DataFrame:
    if master_csv.exists():
        print(f"[*] 正在讀取全模態 Master Dataset: {master_csv} ...")
        df = pd.read_csv(master_csv)
    elif FALLBACK_MM_CSV.exists() and FALLBACK_CM_CSV.exists():
        print("[*] 自動合併 ROI 多模態表與跨模態協同表...")
        df_mm = pd.read_csv(FALLBACK_MM_CSV)
        df_cm = pd.read_csv(FALLBACK_CM_CSV)
        for d in [df_mm, df_cm]:
            d["account_id"] = d["account_id"].astype(str).str.lower().str.strip()
            d["video_id"] = d["video_id"].astype(str).str.lower().str.strip()
        cm_cols = [c for c in df_cm.columns if c not in df_mm.columns]
        df = pd.merge(df_mm, df_cm[["account_id", "video_id"] + cm_cols], on=["account_id", "video_id"], how="inner")
    else:
        raise FileNotFoundError(
            f"找不到輸入資料集 ({master_csv})，請先執行 build_multimodal_roi_dataset-v2.py -> compute_cross_modal_saliency.py -> compute_cross_modal_dataset-v2.py。"
        )

    # 兼容視覺欄位命名
    col_aliases = {
        "hook_colorfulness_mean": "vis_hook_colorfulness_mean",
        "hook_face_rms_contrast_mean": "vis_hook_face_rms_contrast_mean",
        "whole_skin_warmth_cbcr_mean": "vis_whole_skin_warmth_cbcr_mean",
        "hook_figure_ground_lum_contrast_mean": "vis_hook_figure_ground_lum_contrast_mean",
    }
    for std_name, alt_name in col_aliases.items():
        if std_name not in df.columns and alt_name in df.columns:
            df[std_name] = df[alt_name]

    # 建構對數化受眾互動應變數（緩解右偏長尾分佈，保留 0 值）
    ratio_transforms = [
        ("VFR", "log_vfr", 1.0),
        ("ERV", "log_erv_bps", 10000.0),
        ("CLR", "log_clr_permille", 1000.0),
        ("SRV", "log_srv_bps", 10000.0),
        ("SLR", "log_slr_permille", 1000.0),
    ]
    for raw_col, log_col, scale in ratio_transforms:
        if raw_col in df.columns:
            val = pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            df[log_col] = np.log1p(val * scale)

    return df


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return ""


def run_single_target_ols(
    df: pd.DataFrame,
    target_col: str,
    target_title: str,
    feature_cols: list,
    feature_labels: dict,
) -> Tuple[str, list]:
    """對單一應變數執行 HC3 穩健 OLS，並計算標準化係數 (Std Beta) 與統計顯著性排名"""
    sub = df[[target_col] + feature_cols].dropna().copy()
    X_raw = sub[feature_cols].astype(float)
    y_raw = sub[target_col].astype(float)

    X_const = sm.add_constant(X_raw)
    model_raw = sm.OLS(y_raw, X_const).fit(cov_type="HC3")

    X_std = (X_raw - X_raw.mean()) / X_raw.std(ddof=1)
    y_std = (y_raw - y_raw.mean()) / y_raw.std(ddof=1)
    model_std = sm.OLS(y_std, sm.add_constant(X_std)).fit(cov_type="HC3")

    rank_rows = []
    for c in feature_cols:
        coef_raw = float(model_raw.params[c])
        se_raw = float(model_raw.bse[c])
        z_val = float(model_raw.tvalues[c])
        p_val = float(model_raw.pvalues[c])
        std_beta = float(model_std.params[c])
        rank_rows.append({
            "target": target_col,
            "target_title": target_title,
            "feature": c,
            "feature_label": feature_labels.get(c, c),
            "coef_raw": coef_raw,
            "std_err_hc3": se_raw,
            "std_beta": std_beta,
            "z_stat": z_val,
            "abs_z": abs(z_val),
            "p_value": p_val,
            "sig": p_to_stars(p_val),
            "r_squared": float(model_raw.rsquared),
            "f_stat": float(model_raw.fvalue),
            "f_pvalue": float(model_raw.f_pvalue),
            "n_obs": int(len(sub)),
        })

    rank_df = pd.DataFrame(rank_rows).sort_values(by="abs_z", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    lines = []
    lines.append("=" * 105)
    lines.append(
        f"【{target_title} [{target_col}] | N = {len(sub)} | R² = {model_raw.rsquared:.4f} | F = {model_raw.fvalue:.2f} (p = {model_raw.f_pvalue:.2e})】"
    )
    lines.append("=" * 105)
    lines.append("▶ 特徵統計顯著性與標準化影響力排行榜 (按 |z| 降序排列，*** p<0.001, ** p<0.01, * p<0.05)：")
    lines.append("-" * 105)
    lines.append(
        f"{'排名':<4} | {'跨模態特徵名稱':<44} | {'Std Beta':>9} | {'原始 Coef':>10} | {'z 統計量':>8} | {'p-value':>10} | {'顯著'}"
    )
    lines.append("-" * 105)

    for _, r in rank_df.iterrows():
        lines.append(
            f"#{int(r['rank']):<3} | {r['feature_label']:<44} | {r['std_beta']:+9.4f} | {r['coef_raw']:+10.4f} | "
            f"{r['z_stat']:+8.2f} | {r['p_value']:>10.4e} | {r['sig']}"
        )
    lines.append("-" * 105 + "\n")

    return "\n".join(lines), rank_df.to_dict(orient="records")


def run_tri_modal_econometrics(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    candidate_features = [
        ("semantic_facial_peak_coinc", "跨模態：語意-綜合表情峰值重合率 (Semantic-Expr Peak %)"),
        ("optimal_lag_max_corr", "跨模態：視聽最佳延遲最大相關性 (Max Lagged Corr)"),
        ("qc_nasolabial_shear_dt_mean", "幾何微表情：鼻唇溝應力交界帶形變 (Nasolabial d_T)"),
        ("qc_eyes_duchenne_dt_mean", "幾何微表情：眼周區域微形變 (Duchenne d_T)"),
        ("qc_mouth_zygomaticus_dt_mean", "幾何微表情：口周與嘴角發聲動態 (Mouth d_T)"),
        ("aud_std_engagement_score", "音訊語意：單片節奏波動標準差 (Audio Pacing Std)"),
        ("aud_hook_first_5s_score", "音訊語意：前 5 秒開頭吸睛度 (Hook 5s Score)"),
        ("aud_mean_semantic_score", "音訊語意：全片純語意資訊密度 (Semantic Density)"),
        ("hook_colorfulness_mean", "視覺光影：前 3 秒色彩豐富度 (Hook Colorfulness)"),
        ("hook_face_rms_contrast_mean", "視覺光影：前 3 秒臉部 RMS 對比度 (Facial Contrast)"),
        ("whole_skin_warmth_cbcr_mean", "視覺光影：全片膚色血色暖度 (Skin Warmth Cr-Cb)"),
        ("hook_figure_ground_lum_contrast_mean", "視覺光影：前 3 秒主體背景明暗反差 (Fg-Bg Contrast)"),
    ]

    feature_cols = [c for c, _ in candidate_features if c in df.columns]
    feature_labels = dict(candidate_features)

    clean_df = df.dropna(subset=["video_quality_score"] + feature_cols).copy()

    report_sections = []
    report_sections.append("=" * 105)
    report_sections.append("【三模態跨維度聯合計量經濟學分析報告 v2 — 特徵有效性篩選與多維度互動指標對比】")
    report_sections.append("=" * 105)
    report_sections.append(f"有效全模態樣本總數: {len(clean_df)} 支短影音")
    report_sections.append(f"納入聯合迴歸變數數: {len(feature_cols)} 個跨模態指標")
    report_sections.append("-" * 105)

    X_vif = clean_df[feature_cols].astype(float)
    report_sections.append("▶ 多重共線性診斷 (VIF < 5 代表變數間高度獨立，無共線性威脅)：")
    for i, c in enumerate(feature_cols):
        vif_val = variance_inflation_factor(X_vif.values, i)
        report_sections.append(f"  - {feature_labels[c]:<48}: VIF = {vif_val:.2f}")
    report_sections.append("=" * 105 + "\n")

    target_models = [
        ("video_quality_score", "Model 1 — 綜合品質殘差得分 (Quality Score 0~100)"),
        ("log_vfr", "Model 2 — 觀看粉絲比 ln(1 + VFR) [演算法破圈推流]"),
        ("log_erv_bps", "Model 3 — 觀看互動率 ln(1 + ERV×10000) [點讚共鳴轉換]"),
        ("log_clr_permille", "Model 4 — 留言點讚比 ln(1 + CLR×1000) [深度討論提問]"),
        ("log_srv_bps", "Model 5 — 觀看分享率 ln(1 + SRV×10000) [實用轉發傳播]"),
        ("log_slr_permille", "Model 6 — 分享點讚比 ln(1 + SLR×1000) [收藏工具性 vs 純按讚]"),
    ]

    all_rank_records = []
    for t_col, t_title in target_models:
        if t_col not in clean_df.columns:
            continue
        sec_text, records = run_single_target_ols(clean_df, t_col, t_title, feature_cols, feature_labels)
        report_sections.append(sec_text)
        all_rank_records.extend(records)

    matrix_df = pd.DataFrame(all_rank_records)
    return "\n".join(report_sections), clean_df, matrix_df


def plot_tri_modal_correlation_heatmap(df: pd.DataFrame, out_path: Path):
    """圖表 1：雙變量無條件相關性矩陣熱力圖 (Spearman Correlation Matrix)"""
    desired_fields = [
        ("video_quality_score", "綜合品質得分 (Y1)"),
        ("log_vfr", "觀看粉絲比 (log VFR)"),
        ("log_erv_bps", "觀看互動率 (log ERV)"),
        ("log_clr_permille", "留言點讚比 (log CLR)"),
        ("log_srv_bps", "觀看分享率 (log SRV)"),
        ("log_slr_permille", "分享點讚比 (log SLR)"),
        ("semantic_facial_peak_coinc", "語意-表情峰值重合率"),
        ("optimal_lag_max_corr", "視聽最大諧振"),
        ("qc_nasolabial_shear_dt_mean", "鼻唇溝微應變"),
        ("qc_eyes_duchenne_dt_mean", "眼周微應變"),
        ("qc_mouth_zygomaticus_dt_mean", "口周發聲動態"),
        ("aud_std_engagement_score", "音訊節奏波動Std"),
        ("aud_hook_first_5s_score", "前5秒開頭吸睛度"),
        ("hook_colorfulness_mean", "前3秒色彩豐富度"),
        ("hook_face_rms_contrast_mean", "前3秒臉部對比度"),
        ("whole_skin_warmth_cbcr_mean", "膚色血色暖度"),
        ("hook_figure_ground_lum_contrast_mean", "前3秒主體背景反差"),
    ]

    valid_pairs = [(c, lbl) for c, lbl in desired_fields if c in df.columns]
    cols = [c for c, _ in valid_pairs]
    labels = [lbl for _, lbl in valid_pairs]

    corr = df[cols].corr(method="spearman")

    plt.figure(figsize=(13.5, 11.5), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-0.30,
        vmax=0.40,
        square=True,
        linewidths=0.6,
        annot_kws={"size": 8.5},
        cbar_kws={"shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(
        "Tri-Modal Cross-Dimension Spearman Correlation Matrix\n"
        "(6 Engagement Metrics × Facial ROI QC × Audio-Text × Color & Lighting)",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(rotation=40, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[+] 雙變量相關性熱圖已生成至: {out_path}")


def plot_multi_target_significance_heatmap(matrix_df: pd.DataFrame, out_path: Path):
    """圖表 2：多模型控制後標準化係數 (Std Beta) 與顯著性星號熱力圖"""
    target_order = [
        "video_quality_score",
        "log_vfr",
        "log_erv_bps",
        "log_clr_permille",
        "log_srv_bps",
        "log_slr_permille",
    ]
    target_short_names = {
        "video_quality_score": "綜合品質\n(Quality Score)",
        "log_vfr": "觀看粉絲比\n(log VFR)",
        "log_erv_bps": "觀看互動率\n(log ERV)",
        "log_clr_permille": "留言點讚比\n(log CLR)",
        "log_srv_bps": "觀看分享率\n(log SRV)",
        "log_slr_permille": "分享點讚比\n(log SLR)",
    }

    valid_targets = [t for t in target_order if t in matrix_df["target"].unique()]
    pivot_beta = matrix_df.pivot(index="feature_label", columns="target", values="std_beta")[valid_targets]
    pivot_sig = matrix_df.pivot(index="feature_label", columns="target", values="sig")[valid_targets]

    if "video_quality_score" in pivot_beta.columns:
        sort_idx = pivot_beta["video_quality_score"].abs().sort_values(ascending=False).index
        pivot_beta = pivot_beta.loc[sort_idx]
        pivot_sig = pivot_sig.loc[sort_idx]

    annot_matrix = np.empty(pivot_beta.shape, dtype=object)
    for i in range(pivot_beta.shape[0]):
        for j in range(pivot_beta.shape[1]):
            b_val = pivot_beta.iloc[i, j]
            s_val = pivot_sig.iloc[i, j]
            annot_matrix[i, j] = f"{b_val:+.3f}{s_val}"

    plt.figure(figsize=(13, 8.5), dpi=300)
    sns.heatmap(
        pivot_beta,
        annot=annot_matrix,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-0.16,
        vmax=0.16,
        linewidths=0.8,
        cbar_kws={"label": "Standardized Beta (HC3 Robust OLS)", "shrink": 0.85},
        xticklabels=[target_short_names[t] for t in valid_targets],
        yticklabels=pivot_beta.index,
    )
    plt.title(
        "Tri-Modal Feature Significance & Standardized Effect Size Across Engagement Metrics\n"
        "(*** p < 0.001, ** p < 0.01, * p < 0.05, . p < 0.10 | N = 2,996)",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontsize=10.5, fontweight="bold")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[+] 多模型顯著性 Beta 熱圖已生成至: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="執行三模態特徵有效性篩選與多維度互動指標迴歸分析 v2")
    parser.add_argument("--master_csv", type=str, default=str(MASTER_CSV), help="comprehensive_master_dataset.csv 路徑")
    parser.add_argument("--out_dir", type=str, default=str(OUTPUT_DIR), help="報告與圖表輸出目錄")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        merged_df = load_master_dataset(Path(args.master_csv))
    except Exception as e:
        print(f"[-] 資料載入失敗: {e}")
        return

    report_text, final_df, matrix_df = run_tri_modal_econometrics(merged_df)
    print(report_text)

    # 1. 儲存計量迴歸報告
    report_path = out_dir / "tri_modal_econometric_regression_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[+] 完整計量迴歸報告已存至: {report_path}")

    # 2. 儲存特徵顯著性總矩陣 CSV
    matrix_path = out_dir / "tri_modal_feature_significance_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    print(f"[+] 特徵顯著性總矩陣 CSV 已存至: {matrix_path}")

    # 3. 儲存三模態計量分析 Master Dataset CSV
    master_out = out_dir / "tri_modal_master_dataset.csv"
    final_df.to_csv(master_out, index=False, encoding="utf-8-sig")
    print(f"[+] 三模態分析資料集已匯出至: {master_out}")

    # 4. 生成雙變量相關性熱圖
    corr_heatmap_path = out_dir / "tri_modal_correlation_heatmap.png"
    plot_tri_modal_correlation_heatmap(final_df, corr_heatmap_path)

    # 5. 生成多模型控制後 Beta 顯著性熱圖
    beta_heatmap_path = out_dir / "tri_modal_beta_significance_heatmap.png"
    plot_multi_target_significance_heatmap(matrix_df, beta_heatmap_path)

    print("\n" + "=" * 75)
    print("[+] 三模態特徵篩選與 6 大互動指標聯合分析全部順利完成！")
    print("=" * 75)


if __name__ == "__main__":
    main()