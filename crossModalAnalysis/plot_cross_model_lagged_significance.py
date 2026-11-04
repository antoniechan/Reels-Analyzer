import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import CROSS_MODAL_PROC_DIR

CSV_PATH = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_dataset.csv"
OUT_IMG = Path(CROSS_MODAL_PROC_DIR) / "cross_modal_lagged_significance.png"

# 設定字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

if not CSV_PATH.exists():
    print(f"[-] 找不到資料集: {CSV_PATH}")
    exit()

df = pd.read_csv(CSV_PATH)
TIER_ORDER = ["good", "neutral", "bad"]
TIER_COLORS = {"good": "#2ca02c", "neutral": "#ff7f0e", "bad": "#d62728"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.patch.set_facecolor("white")

# 1. 左圖：核心顯著指標 Max Lagged Correlation (Good vs Bad p=0.0262*)
sns.boxplot(
    data=df,
    x="tier",
    y="optimal_lag_max_corr",
    order=TIER_ORDER,
    palette=TIER_COLORS,
    width=0.45,
    fliersize=0,
    boxprops=dict(alpha=0.75),
    ax=axes[0]
)
sns.stripplot(
    data=df,
    x="tier",
    y="optimal_lag_max_corr",
    order=TIER_ORDER,
    color="black",
    alpha=0.2,
    jitter=0.2,
    size=3.5,
    ax=axes[0]
)

# 標註顯著性 bracket 與 p-value (Good vs Bad)
y_max = df["optimal_lag_max_corr"].quantile(0.98)
axes[0].plot([0, 0, 2, 2], [y_max, y_max * 1.05, y_max * 1.05, y_max], color="black", lw=1.2)
axes[0].text(1, y_max * 1.07, "p = 0.0262* (Welch's t)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="darkred")

axes[0].set_title("最佳延遲下最大視聽相關性\n(Max Lagged Correlation: Good vs Bad p < 0.05)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Max Lagged Correlation", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

# 2. 右圖：最優延遲時間 (Optimal Lag) 的分佈 KDE / 箱形圖
sns.kdeplot(
    data=df,
    x="optimal_lag_seconds",
    hue="tier",
    hue_order=TIER_ORDER,
    palette=TIER_COLORS,
    common_norm=False,
    linewidth=2,
    fill=True,
    alpha=0.15,
    ax=axes[1]
)
axes[1].axvline(0, color="gray", linestyle="--", alpha=0.7, label="零延遲基準 (Lag=0)")
axes[1].set_xlim(-4, 4)
axes[1].set_title("最優視聽延遲時間分佈\n(Optimal Lag Distribution: Seconds)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Optimal Lag (s) [負值: 視覺先行 | 正值: 聲音先行]", fontsize=10)
axes[1].set_ylabel("密度 (Density)", fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_IMG, bbox_inches="tight")
plt.close()

print(f"[+] 顯著性對比圖表已生成至: {OUT_IMG}")