import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from config import VIDEO_PROC_DIR, DATA_PROC_DIR, DEFAULT_REELS_DIR

# 中文字型設定
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CLASS_JSON = str(DATA_PROC_DIR / "video_classification_summary.json")
OUTPUT_DIR = str(VIDEO_PROC_DIR)

TIER_ORDER = ["good", "neutral", "bad"]
TIER_COLORS = {"good": "#2ca02c", "neutral": "#ff7f0e", "bad": "#d62728"}


def load_video_tier_lookup(summary_json_path: str) -> dict:
    """讀取 video_classification_summary.json 建立檢索字典"""
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(f"找不到影片分類檔: {summary_json_path}")
    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    videos_list = data.get("videos", data) if isinstance(data, dict) else data
    lookup = {}
    for item in videos_list:
        if not isinstance(item, dict):
            continue
        acc_id = str(item.get("account_id", "")).strip()
        vid_id = str(item.get("video_id", "")).strip()
        local_name = str(item.get("local_index_name", "")).strip()
        if acc_id and vid_id:
            lookup[(acc_id, vid_id)] = item
        if acc_id and local_name:
            lookup[(acc_id, local_name)] = item
    return lookup


def compute_frame_yaw_angle(landmarks: list) -> float:
    """
    依據 #33 (右眼內眥) 與 #263 (左眼內眥) 的 3D 座標 (x, y, z)
    計算該影格的頭部水平偏轉角 (Yaw Angle, 單位: 度)
    """
    if not landmarks or len(landmarks) < 468:
        return np.nan

    # MediaPipe 索引：33 = 右眼內眥, 263 = 左眼內眥
    lm_r = landmarks[33]
    lm_l = landmarks[263]

    dx = float(lm_l["x"]) - float(lm_r["x"])
    dz = float(lm_l["z"]) - float(lm_r["z"])

    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0

    # 弧度轉角度
    yaw_deg = np.arctan2(dz, dx) * (180.0 / np.pi)
    return float(yaw_deg)


def process_video_face_json(face_json_path: str):
    """
    讀取單一 *_face.json，提取所有有效幀的 Yaw Angle，
    計算該影片的：
    1. 平均絕對偏轉角 mean_abs_yaw (代表側臉傾斜程度)
    2. 有符號平均偏轉角 mean_signed_yaw (代表固定偏向左側或右側)
    3. 頭部晃動穩定度 std_yaw (代表片中轉頭/晃動幅度)
    4. 最大偏轉角 max_abs_yaw
    """
    try:
        with open(face_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    frames = data.get("frames", [])
    if not frames:
        return None

    yaw_series = []
    for fr in frames:
        if not fr.get("face_detected", False):
            continue
        lms = fr.get("landmarks")
        if not lms:
            continue
        yaw = compute_frame_yaw_angle(lms)
        if not np.isnan(yaw):
            yaw_series.append(yaw)

    if not yaw_series:
        return None

    yaws = np.array(yaw_series, dtype=np.float64)
    abs_yaws = np.abs(yaws)

    return {
        "valid_face_frames": len(yaws),
        "mean_abs_yaw": float(np.mean(abs_yaws)),
        "median_abs_yaw": float(np.median(abs_yaws)),
        "max_abs_yaw": float(np.max(abs_yaws)),
        "mean_signed_yaw": float(np.mean(yaws)),
        "std_yaw": float(np.std(yaws, ddof=1)) if len(yaws) > 1 else 0.0,
    }


def analyze_all_tiers_yaw(root_dir=DEFAULT_REELS_DIR, class_json=DEFAULT_CLASS_JSON, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    lookup = load_video_tier_lookup(class_json)

    # 搜尋所有 *_face.json 檔案
    face_files = sorted(glob.glob(os.path.join(root_dir, "**", "*_face.json"), recursive=True))
    print("=" * 80)
    print("【短影音頭部水平偏轉角 (Head Yaw Angle) 跨組分析系統】")
    print(f"  - 影片目錄:   {root_dir}")
    print(f"  - 分類來源:   {class_json}")
    print(f"  - 找到 face.json 總數: {len(face_files)}")
    print("=" * 80 + "\n")

    records = []
    matched_count = 0

    for idx, fpath in enumerate(face_files, 1):
        folder_dir, fname = os.path.split(fpath)
        account_id = os.path.basename(folder_dir)
        video_stem = fname[:-10] if fname.endswith("_face.json") else os.path.splitext(fname)[0]

        v_info = lookup.get((account_id, video_stem))
        if v_info is None:
            continue

        tier = str(v_info.get("video_classification", "neutral")).lower()
        if tier not in TIER_ORDER:
            continue

        yaw_metrics = process_video_face_json(fpath)
        if yaw_metrics is None:
            continue

        matched_count += 1
        records.append({
            "account_id": account_id,
            "video_id": video_stem,
            "tier": tier,
            "video_quality_score": float(v_info.get("video_quality_score") or 0.0),
            **yaw_metrics
        })

    df = pd.DataFrame(records)
    print(f"成功配對並提取 {len(df)} 支影片的有效人臉姿態資料。\n")

    if df.empty:
        print("未提取到足夠的資料，程式結束。")
        return

    # --------------------------------------------------------------------------
    # 統計檢定與文字報告
    # --------------------------------------------------------------------------
    metrics_to_test = [
        ("mean_abs_yaw", "平均絕對偏轉角 (Mean |Yaw| °)", "衡量側臉/未正對鏡頭的程度"),
        ("std_yaw", "偏轉角標準差 (Std of Yaw °)", "衡量影片中頭部轉動/晃動劇烈程度"),
        ("max_abs_yaw", "最大偏轉角 (Max |Yaw| °)", "衡量影片中最大側臉極端角度"),
        ("mean_signed_yaw", "有符號平均偏轉角 (Signed Yaw °)", "衡量偏向左側(+)或右側(-)的系統性偏好"),
    ]

    report_lines = []
    report_lines.append("=" * 85)
    report_lines.append("【各 Video Tier 頭部水平偏轉角 (Head Yaw Angle) 統計檢定報告】")
    report_lines.append("=" * 85)
    for t in TIER_ORDER:
        cnt = len(df[df["tier"] == t])
        report_lines.append(f"  - {t.upper()} Tier: {cnt} 支影片")
    report_lines.append("-" * 85 + "\n")

    summary_rows = []

    for m_key, m_title, m_desc in metrics_to_test:
        report_lines.append(f"▶ 指標：{m_title} — {m_desc}")
        report_lines.append("-" * 75)

        g_vals = df[df["tier"] == "good"][m_key].dropna().values
        n_vals = df[df["tier"] == "neutral"][m_key].dropna().values
        b_vals = df[df["tier"] == "bad"][m_key].dropna().values

        stats_dict = {}
        for t, vals in [("good", g_vals), ("neutral", n_vals), ("bad", b_vals)]:
            m = np.mean(vals)
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            med = np.median(vals)
            stats_dict[t] = (m, s, med)
            report_lines.append(f"  [{t.upper():<7}] Mean = {m:.2f}° ± {s:.2f}° | Median = {med:.2f}°")

        # 檢定
        f_stat, p_anova = stats.f_oneway(g_vals, n_vals, b_vals)
        t_gb, p_welch = stats.ttest_ind(g_vals, b_vals, equal_var=False)
        u_gb, p_mann = stats.mannwhitneyu(g_vals, b_vals, alternative="two-sided")

        # 效應量
        nx, ny = len(g_vals), len(b_vals)
        pooled_std = np.sqrt(((nx - 1) * np.var(g_vals, ddof=1) + (ny - 1) * np.var(b_vals, ddof=1)) / (nx + ny - 2))
        d_val = (np.mean(g_vals) - np.mean(b_vals)) / pooled_std if pooled_std > 0 else 0.0

        report_lines.append(f"  [跨組 ANOVA]     F = {f_stat:.4f}, p = {p_anova:.4e}")
        report_lines.append(f"  [GOOD vs. BAD]   Welch's t = {t_gb:.4f}, p = {p_welch:.4e} | Cohen's d = {d_val:.4f}")
        report_lines.append(f"                   Mann-Whitney U = {u_gb:.1f}, p = {p_mann:.4e}")
        report_lines.append("")

        summary_rows.append({
            "metric": m_key,
            "title": m_title,
            "good_mean": stats_dict["good"][0],
            "good_std": stats_dict["good"][1],
            "bad_mean": stats_dict["bad"][0],
            "bad_std": stats_dict["bad"][1],
            "anova_p": p_anova,
            "welch_t": t_gb,
            "welch_p": p_welch,
            "cohen_d": d_val,
        })

    # 輸出文字與 CSV 報表
    report_txt = os.path.join(output_dir, "head_yaw_analysis/head_yaw_statistical_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    df_csv = os.path.join(output_dir, "head_yaw_analysis/head_yaw_video_dataset.csv")
    df.to_csv(df_csv, index=False, encoding="utf-8-sig")

    print("\n".join(report_lines))
    print(f"\n報告已儲存至: {os.path.abspath(report_txt)}")
    print(f"數據集已儲存至: {os.path.abspath(df_csv)}")

    # --------------------------------------------------------------------------
    # 繪製出版級圖表 (箱型圖與核密度估計圖)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    plot_configs = [
        (axes[0], "mean_abs_yaw", "平均絕對偏轉角 (|Yaw| °)\n(越小越正面，越大越側臉)", (0, 35)),
        (axes[1], "std_yaw", "偏轉角標準差 (Std °)\n(越小越沉穩，越大越晃動)", (0, 25)),
        (axes[2], "max_abs_yaw", "最大極端偏轉角 (Max |Yaw| °)\n(轉頭動作的最大極限)", (0, 60)),
    ]

    for ax, m_key, m_title, y_range in plot_configs:
        sns.boxplot(
            data=df,
            x="tier",
            y=m_key,
            order=TIER_ORDER,
            palette=TIER_COLORS,
            width=0.45,
            fliersize=0,
            boxprops=dict(alpha=0.8),
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="tier",
            y=m_key,
            order=TIER_ORDER,
            color="black",
            alpha=0.18,
            jitter=0.2,
            size=3.5,
            ax=ax,
        )
        ax.set_title(m_title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Video Tier", fontsize=10)
        ax.set_ylabel("Degrees (°)", fontsize=10)
        ax.set_ylim(y_range)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Head Pose Stability & Orientation Across Video Performance Tiers", fontsize=13, fontweight="bold",
                 y=1.03)
    plt.tight_layout()
    plot_png = os.path.join(output_dir, "head_yaw_analysis/head_yaw_tier_comparison.png")
    plt.savefig(plot_png, bbox_inches="tight")
    plt.close()
    print(f"對比圖表已生成至: {os.path.abspath(plot_png)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析不同 Video Tier 的頭部水平偏轉角 (Head Yaw Angle)")
    parser.add_argument("--dir", type=str, default=DEFAULT_REELS_DIR, help="Reels 根目錄路徑")
    parser.add_argument("--class_json", type=str, default=DEFAULT_CLASS_JSON, help="影片分類 JSON 路徑")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR, help="分析結果輸出目錄")
    args = parser.parse_args()

    analyze_all_tiers_yaw(root_dir=args.dir, class_json=args.class_json, output_dir=args.out_dir)
