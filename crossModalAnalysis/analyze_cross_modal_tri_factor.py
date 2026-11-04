#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2: Tri-Modal Cross-Dimension Econometric & Interaction Analysis
---------------------------------------------------------------------
三模態跨維度聯合分析系統：
- 模態一 (環境與光影視覺): 色彩豐富度、面部柔光對比 (RMS)、膚色血色感 (Cr-Cb)
- 模態二 (臉部微表情幾何): Teichmüller 擬共形變形與關鍵表情點
- 模態三 (聲學波形與文本語意): Dual-Stream Audio-Text Saliency

研究核心：
1. 控制了畫面色彩豐富度與臉部打光均勻度後，視聽時空動態諧振 (Max Lagged Corr) 是否依然維持統計顯著？
2. 評估三模態特徵之間的相互關聯與整體多重共線性 (VIF / Correlation)。
3. 輸出計量經濟學 OLS 報表、整合 Master Dataset 與高解析度相關性熱圖。
"""

import sys
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from config import VIDEO_PROC_DIR, CROSS_MODAL_PROC_DIR

# 設定高解析度與中文字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 1. 動態定位路徑設定
# ---------------------------------------------------------------------------
COLOR_CSV = VIDEO_PROC_DIR / "color_lighting_analysis" / "visual_color_lighting_dataset.csv"
CROSS_MODAL_CSV = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_dataset.csv"
OUTPUT_DIR = CROSS_MODAL_CSV / "cross_modal_tri_factor_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. 資料載入與雙鍵精準合併 (Account ID + Video ID)
# ---------------------------------------------------------------------------
def load_and_merge_datasets() -> pd.DataFrame:
    if not COLOR_CSV.exists():
        raise FileNotFoundError(f"[-] 找不到第一部分的色彩光影資料集: {COLOR_CSV}，請先執行第一部分分析。")
    if not CROSS_MODAL_CSV.exists():
        raise FileNotFoundError(f"[-] 找不到視聽跨模態資料集: {CROSS_MODAL_CSV}")

    print(f"[*] 正在讀取視覺光影資料集: {COLOR_CSV} ...")
    df_color = pd.read_csv(COLOR_CSV)

    print(f"[*] 正在讀取視聽跨模態資料集: {CROSS_MODAL_CSV} ...")
    df_cm = pd.read_csv(CROSS_MODAL_CSV)

    # 標準化雙鍵值，徹底免疫跨帳號同名 (例如皆有 video_001) 覆蓋問題
    for df in [df_color, df_cm]:
        df["account_id"] = df["account_id"].astype(str).str.lower().str.strip()
        df["video_id"] = df["video_id"].astype(str).str.lower().str.strip()

    # 挑選跨模態的核心指標欄位
    cm_cols = [
        "account_id",
        "video_id",
        "sync_pearson_r",
        "optimal_lag_max_corr",
        "optimal_lag_seconds",
        "peak_coincidence_rate"
    ]

    # 執行內部合併 (Inner Join)
    merged = pd.merge(df_color, df_cm[cm_cols], on=["account_id", "video_id"], how="inner")
    print(f"[*] 成功整合三模態有效對齊樣本數: {len(merged)} 支影片。")

    # 清理品質得分與核心自變數缺失值
    clean_df = merged.dropna(subset=[
        "video_quality_score",
        "optimal_lag_max_corr",
        "hook_colorfulness_mean",
        "hook_face_rms_contrast_mean",
        "hook_skin_warmth_cbcr_mean"
    ]).copy()

    print(f"[*] 剔除缺失值後的最終計量樣本數: {len(clean_df)} 支影片。")
    return clean_df


# ---------------------------------------------------------------------------
# 3. 計量多元迴歸與統計檢驗核心
# ---------------------------------------------------------------------------
def run_tri_modal_econometrics(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("【三模態跨維度聯合計量經濟學迴歸分析報告】")
    report_lines.append("=" * 90)
    report_lines.append("模態構成說明：")
    report_lines.append("  1. 視聽動態諧振 (Audio-Text + Facial QC): 最佳延遲下最大相關性 (optimal_lag_max_corr)")
    report_lines.append("  2. 開頭視覺衝擊 (Visual Colorfulness):   前 3 秒色彩豐富度 (hook_colorfulness_mean)")
    report_lines.append("  3. 面部光影立體/柔和度 (Lighting Contrast):  前 3 秒臉部 RMS 對比度 (hook_face_rms_contrast_mean)")
    report_lines.append("  4. 講者好氣色通透度 (Skin Health Tone):     全片膚色血色感 (whole_skin_warmth_cbcr_mean)")
    report_lines.append("  5. 主體分離度 (Figure-Ground Separation):  前 3 秒人物與背景反差 (hook_figure_ground_lum_contrast_mean)")
    report_lines.append("-" * 90)
    report_lines.append(f"應變數 (Dependent Variable):  video_quality_score (計量品質殘差得分)")
    report_lines.append(f"總觀察樣本數 (N):              {len(df)} 支短影音")
    report_lines.append("-" * 90 + "\n")

    # 定義自變數特徵
    feature_cols = [
        "optimal_lag_max_corr",
        "hook_colorfulness_mean",
        "hook_face_rms_contrast_mean",
        "whole_skin_warmth_cbcr_mean",
        "hook_figure_ground_lum_contrast_mean"
    ]

    feature_labels = {
        "optimal_lag_max_corr": "視聽最大諧振 (Max Lagged Corr)",
        "hook_colorfulness_mean": "前 3 秒色彩豐富度 (Hook Colorfulness)",
        "hook_face_rms_contrast_mean": "前 3 秒臉部 RMS 對比度 (Facial Contrast)",
        "whole_skin_warmth_cbcr_mean": "全片膚色血色感 (Skin Warmth Cr-Cb)",
        "hook_figure_ground_lum_contrast_mean": "前 3 秒主體背景明暗反差 (Fg-Bg Contrast)"
    }

    X = df[feature_cols].astype(float)
    y = df["video_quality_score"].astype(float)

    # 檢測多重共線性 (Variance Inflation Factor, VIF)
    vif_data = pd.DataFrame()
    vif_data["特徵變數"] = [feature_labels[c] for c in feature_cols]
    vif_data["VIF 指數"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    report_lines.append("▶ 多重共線性診斷 (VIF 檢定，小於 5 代表無共線性威脅)：")
    report_lines.append("-" * 60)
    for _, r in vif_data.iterrows():
        report_lines.append(f"  - {r['特徵變數']:<40}: VIF = {r['VIF 指數']:.2f}")
    report_lines.append("\n" + "=" * 90)

    # 建立多元 OLS 模型 (採用 HC3 異質變異數穩健標準誤)
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit(cov_type="HC3")

    report_lines.append("▶ 多元 OLS 計量模型估計結果 (HC3 Robust Standard Errors)：")
    report_lines.append("-" * 90)
    report_lines.append(str(model.summary()))
    report_lines.append("=" * 90 + "\n")

    report_text = "\n".join(report_lines)
    return report_text, df


# ---------------------------------------------------------------------------
# 4. 繪製三模態跨維度特徵相關矩陣熱圖
# ---------------------------------------------------------------------------
def plot_tri_modal_correlation_heatmap(df: pd.DataFrame, out_path: Path):
    target_fields = [
        "video_quality_score",
        "optimal_lag_max_corr",
        "peak_coincidence_rate",
        "hook_colorfulness_mean",
        "hook_face_rms_contrast_mean",
        "whole_skin_warmth_cbcr_mean",
        "hook_figure_ground_lum_contrast_mean"
    ]

    target_labels = [
        "品質得分 (Y)",
        "視聽最大諧振 (Tri)",
        "峰值重合率",
        "色彩豐富度 (Hook)",
        "面部 RMS (柔光)",
        "膚色血色感 (Warmth)",
        "主體明暗反差"
    ]

    corr = df[target_fields].corr(method="pearson")

    plt.figure(figsize=(9.5, 8), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-0.2,
        vmax=0.3,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        xticklabels=target_labels,
        yticklabels=target_labels
    )
    plt.title("三模態跨維度特徵與品質分數相關性熱圖\n(Tri-Modal Saliency × Color & Lighting Correlation)", fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[+] 相關性熱圖已生成至: {out_path}")


# ---------------------------------------------------------------------------
# 5. 主執行入口
# ---------------------------------------------------------------------------
def main():
    try:
        merged_df = load_and_merge_datasets()
    except Exception as e:
        print(f"[-] 資料整合失敗: {e}")
        return

    # 1. 執行多元迴歸分析
    report_text, final_df = run_tri_modal_econometrics(merged_df)
    print(report_text)

    report_path = OUTPUT_DIR / "tri_modal_econometric_regression_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[+] 完整迴歸報告已存至: {report_path}")

    # 2. 儲存三模態 Master Dataset
    master_csv_path = OUTPUT_DIR / "tri_modal_master_dataset.csv"
    final_df.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] 三模態完整資料集已匯出至: {master_csv_path}")

    # 3. 生成相關性熱圖
    heatmap_path = OUTPUT_DIR / "tri_modal_correlation_heatmap.png"
    plot_tri_modal_correlation_heatmap(final_df, heatmap_path)

    print("\n" + "=" * 60)
    print("[+] 三模態聯合分析全部順利完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()