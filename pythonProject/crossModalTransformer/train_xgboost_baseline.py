#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Baseline Model for Reels Multimodal Quality Prediction
--------------------------------------------------------------
功能：
1. 讀取三模態整合資料集 (tri_modal_master_dataset.csv)。
2. 任務一：分類模型 (Good vs. Bad)，輸出 Accuracy、F1-Score 與 ROC-AUC。
3. 任務二：迴歸模型 (預測連續品質得分)，輸出 RMSE、MAE 與 R^2。
4. 計算特徵重要性 (Gain / Weight)，並自動繪製高解析度特徵貢獻長條圖。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import xgboost as xgb
from config import CROSS_MODAL_PROC_DIR, TRANSFORMER_DIR

# 設定繪圖字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 路徑定位
CSV_PATH = CROSS_MODAL_PROC_DIR / "master_dataset_results" / "comprehensive_master_dataset.csv"
OUTPUT_DIR = TRANSFORMER_DIR / "xgboost_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"[-] 找不到資料集: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[*] 成功讀取資料集，總樣本數: {len(df)} 筆。")

    # 挑選所有我們提取過的跨模態特徵 (自變數 X)
    candidate_features = [
        # 1. 視聽動態諧振指標
        "optimal_lag_max_corr",
        "optimal_lag_seconds",
        "peak_coincidence_rate",
        "sync_pearson_r",
        # 2. 黃金前 3 秒視覺與光影
        "hook_colorfulness_mean",
        "hook_face_rms_contrast_mean",
        "hook_skin_warmth_cbcr_mean",
        "hook_figure_ground_lum_contrast_mean",
        # 3. 全片視覺與光影基調
        "whole_colorfulness_mean",
        "whole_face_rms_contrast_mean",
        "whole_skin_warmth_cbcr_mean",
        "whole_figure_ground_lum_contrast_mean",
    ]

    # 保留存在於 CSV 中的欄位
    features = [f for f in candidate_features if f in df.columns]
    print(f"[*] 納入評估的特徵數量: {len(features)} 個。")
    return df, features


def evaluate_classification(df: pd.DataFrame, features: list):
    """任務一：預測極端組 (Good = 1 vs. Bad = 0)"""
    print("\n" + "=" * 65)
    print("【任務一：極端組分類評估 (Good vs. Bad 分類器)】")
    print("=" * 65)

    sub_df = df[df["tier"].isin(["good", "bad"])].copy()
    sub_df["target"] = (sub_df["tier"] == "good").astype(int)

    X = sub_df[features].values
    y = sub_df["target"].values

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    y_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, y_proba)
    print(classification_report(y, y_pred, target_names=["Bad", "Good"], digits=4))
    print(f"[*] 5-Fold 交叉驗證 ROC-AUC 得分: {auc:.4f}")

    # 用全量極端組訓練模型以提取特徵重要性
    clf.fit(X, y)
    importances = clf.feature_importances_

    return clf, features, importances


def evaluate_regression(df: pd.DataFrame, features: list):
    """任務二：連續品質分數預測 (全量樣本 1,200+ 支)"""
    print("\n" + "=" * 65)
    print("【任務二：全量影片連續品質得分預測 (Regression)】")
    print("=" * 65)

    valid_df = df.dropna(subset=["video_quality_score"]).copy()
    X = valid_df[features].values
    y = valid_df["video_quality_score"].values

    reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(reg, X, y, cv=kf)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"[*] 5-Fold 交叉驗證評估指標：")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE:  {mae:.4f}")
    print(f"  - R^2:   {r2:.4f}")

    return reg


def plot_feature_importance(features: list, importances: np.ndarray, out_path: Path):
    """繪製特徵重要性長條圖"""
    label_map = {
        "hook_face_rms_contrast_mean": "前 3 秒面部 RMS (柔光程度)",
        "whole_skin_warmth_cbcr_mean": "全片膚色血色感 (Cr-Cb)",
        "hook_colorfulness_mean": "前 3 秒畫面色彩豐富度",
        "optimal_lag_max_corr": "視聽最大諧振 (Max Lag Corr)",
        "whole_face_rms_contrast_mean": "全片面部 RMS 對比度",
        "whole_colorfulness_mean": "全片色彩豐富度",
        "hook_skin_warmth_cbcr_mean": "前 3 秒膚色血色感",
        "peak_coincidence_rate": "視聽峰值重合率",
        "optimal_lag_seconds": "最優時滯秒數 (Optimal Lag)",
        "hook_figure_ground_lum_contrast_mean": "前 3 秒主體背景明暗反差",
        "whole_figure_ground_lum_contrast_mean": "全片主體背景明暗反差",
        "sync_pearson_r": "零時滯即時同步率 (Pearson r)",
    }

    readable_names = [label_map.get(f, f) for f in features]
    feat_df = pd.DataFrame({"Feature": readable_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 6.5), dpi=300)
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#1f77b4", alpha=0.85, edgecolor="black")
    plt.xlabel("特徵重要性 (Gain / Relative Importance)", fontsize=11, fontweight="bold")
    plt.title("XGBoost 跨模態特徵貢獻度排名 (Good vs. Bad)", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\n[+] 特徵重要性排名圖已儲存至: {out_path}")


def main():
    df, features = load_dataset(CSV_PATH)

    # 1. 執行分類模型評估
    clf, feat_list, importances = evaluate_classification(df, features)

    # 2. 執行迴歸模型評估
    _ = evaluate_regression(df, features)

    # 3. 繪製特徵重要性圖
    plot_path = OUTPUT_DIR / "xgboost_feature_importance.png"
    plot_feature_importance(feat_list, importances, plot_path)


if __name__ == "__main__":
    main()
