import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DIR = r"D:\reels"
VIDEO_SUMMARY_JSON = "video_classification_summary.json"


def load_qc_data(qc_json_path):
    """讀取單一 _qc.json 檔案"""
    try:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"無法讀取檔案 {qc_json_path}: {e}")
        return None


def extract_vertex_measurements(qc_data, vertex_id, metric="abs_mu", zero_eps=1e-8):
    """
    從一段影片的 QC 資料中，提取特定頂點在所有連續幀轉換中的指定指標數值。
    嚴格剔除未計算出 mu (即 abs_mu <= zero_eps) 的無效面片與影格。
    """
    values = []
    zero_dropped_count = 0
    transitions = qc_data.get("consecutive_qc_metrics", [])

    for trans in transitions:
        if not trans.get("valid_transition", False):
            zero_dropped_count += 1
            continue

        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            zero_dropped_count += 1
            continue

        # 找出包含此 vertex_id、且 abs_mu 非 0 的有效相鄰三角形面片
        neighbor_facets = [
            fm for fm in facet_metrics
            if (vertex_id in fm.get("vertices", [])) and (float(fm.get("abs_mu", 0.0)) > zero_eps)
        ]

        if not neighbor_facets:
            zero_dropped_count += 1
            continue

        if metric == "abs_mu":
            val = float(np.mean([fm["abs_mu"] for fm in neighbor_facets]))
            if val <= zero_eps:
                zero_dropped_count += 1
                continue
            values.append(val)

        elif metric == "dilation":
            val = float(np.mean([fm["dilation"] for fm in neighbor_facets]))
            values.append(val)

        elif metric == "teichmuller_distance":
            max_local_mu = float(np.max([fm["abs_mu"] for fm in neighbor_facets]))
            if max_local_mu <= zero_eps:
                zero_dropped_count += 1
                continue
            max_local_mu = min(max_local_mu, 0.999)
            local_dt = 0.5 * np.log((1.0 + max_local_mu) / (1.0 - max_local_mu))
            values.append(float(local_dt))

    return values, zero_dropped_count


def load_video_tier_lookup(summary_json_path=VIDEO_SUMMARY_JSON):
    """
    從 video_classification_summary.json 讀取每支影片的分類資訊。
    建立 key = (account_id, video_id) 或 (account_id, local_index_name) 的對照字典
    """
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(
            f"找不到 {summary_json_path}，請先確認檔案路徑或執行 video_residual_classification.py。"
        )

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


def get_qc_files_by_video_tier(root_dir=DEFAULT_DIR, target_tier="all", summary_json_path=VIDEO_SUMMARY_JSON):
    """依據 video_classification_summary.json 篩選對應 Tier 的 *_qc.json 檔案"""
    lookup = load_video_tier_lookup(summary_json_path)
    target_tier = target_tier.lower()

    pattern = os.path.join(root_dir, "**", "*_qc.json")
    all_qc_files = sorted(glob.glob(pattern, recursive=True))

    matched_qc = []
    for qc_path in all_qc_files:
        folder_dir, fname = os.path.split(qc_path)
        account_id = os.path.basename(folder_dir)
        video_stem = fname[:-8] if fname.endswith("_qc.json") else os.path.splitext(fname)[0]

        v_info = lookup.get((account_id, video_stem))
        if v_info is None:
            continue

        v_tier = str(v_info.get("video_classification", "neutral")).lower()
        if target_tier == "all" or v_tier == target_tier:
            matched_qc.append((qc_path, v_info))

    # 依 video_quality_score 由高至低排序
    matched_qc.sort(key=lambda x: float(x[1].get("video_quality_score", 0.0)), reverse=True)
    return matched_qc


def plot_video_deformation(
    root_dir=DEFAULT_DIR,
    tier="all",
    vertex_id=1,
    metric="abs_mu",
    max_videos=None,
    summary_json_path=VIDEO_SUMMARY_JSON,
):
    metric_labels = {
        "abs_mu": r"Shear Strain $|\mu|$ (Beltrami Coefficient)",
        "dilation": r"Maximal Dilation $K = \frac{1+|\mu|}{1-|\mu|}$",
        "teichmuller_distance": r"Teichmüller Distance $d_T$",
    }

    if metric not in metric_labels:
        raise ValueError(f"不支援的測量指標: {metric}")

    matched_qc_tuples = get_qc_files_by_video_tier(
        root_dir=root_dir, target_tier=tier, summary_json_path=summary_json_path
    )

    scope_description = f"Video Tier: [{tier.upper()}]"

    print("=" * 78)
    print("【執行參數與兩階段 Video-Level 統計設定 (無 Outlier 剔除 + 零值幀過濾)】")
    print(f"  - 分類依據檔案:             {summary_json_path}")
    print(f"  - 影片分級篩選 (Video Tier): {tier.upper()} (配對檔案數: {len(matched_qc_tuples)})")
    print(f"  - 特徵點編號 (Vertex ID):    #{vertex_id}")
    print(f"  - 測量指標 (Measurement):    {metric} ({metric_labels[metric]})")
    print(f"  - 零值過濾規則:             自動刪除 abs_mu == 0 的無效影格")
    print("=" * 78 + "\n")

    if not matched_qc_tuples:
        print("錯誤：未找到任何符合該 Video Tier 的 *_qc.json 檔案，程式結束。")
        return

    if max_videos is not None:
        matched_qc_tuples = matched_qc_tuples[:max_videos]

    # -------------------------------------------------------------------------
    # Stage 1: 在每條影片內過濾 mu == 0 的影格，不剔除 Outliers，直接計算 Max, Min, Mean, Std
    # -------------------------------------------------------------------------
    per_video_records = []
    total_valid_frames = 0
    total_zero_dropped_frames = 0

    for idx, (qc_path, v_info) in enumerate(matched_qc_tuples, start=1):
        qc_data = load_qc_data(qc_path)
        if qc_data is None:
            continue

        valid_vals, dropped_zeros = extract_vertex_measurements(qc_data, vertex_id, metric=metric)
        total_zero_dropped_frames += dropped_zeros

        if not valid_vals:
            continue

        arr = np.asarray(valid_vals, dtype=np.float64)
        total_valid_frames += len(arr)

        vid_max = float(np.max(arr))
        vid_min = float(np.min(arr))
        vid_mean = float(np.mean(arr))
        vid_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        per_video_records.append({
            "seq_idx": idx,
            "account_id": v_info.get("account_id"),
            "video_id": v_info.get("video_id"),
            "video_quality_score": v_info.get("video_quality_score"),
            "valid_frame_count": len(arr),
            "dropped_zero_frames": dropped_zeros,
            "frames": arr,
            "max": vid_max,
            "min": vid_min,
            "mean": vid_mean,
            "std": vid_std,
        })

    if not per_video_records:
        print("未提取到任何非零有效頂點數據，請確認頂點 ID 是否正確。")
        return

    # -------------------------------------------------------------------------
    # Stage 2: Across-Video 針對 4 個單片統計量 (Max, Min, Mean, Std) 再求 Mean 與 Std
    # -------------------------------------------------------------------------
    n_vids = len(per_video_records)
    stat_keys = ["max", "min", "mean", "std"]
    stat_names = {
        "max": "Video Max  (單片最大值)",
        "min": "Video Min  (單片最小值)",
        "mean": "Video Mean (單片平均值)",
        "std": "Video Std  (單片標準差)",
    }

    across_video_summary = {}
    for k in stat_keys:
        vec = np.array([r[k] for r in per_video_records], dtype=np.float64)
        across_video_summary[k] = {
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec, ddof=1)) if len(vec) > 1 else 0.0,
            "median": float(np.median(vec)),
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
        }

    # Console 輸出對齊報表
    print("=" * 78)
    print(f"【{scope_description} — Across-Video 四大統計量彙總結果 (Vertex #{vertex_id})】")
    print(f"  - 有效分析影片數 (Valid Videos):           {n_vids} 支")
    print(f"  - 納入計算之非零總影格數 (Valid Frames):   {total_valid_frames} 幀")
    print(f"  - 已刪除之 mu=0 無效影格數 (Dropped 0s):   {total_zero_dropped_frames} 幀")
    print("-" * 78)
    print(f"  {'單片指標 (Within-Video Stat)':<26} | {'Across-Video Mean':<18} | {'Across-Video Std':<18}")
    print("-" * 78)
    for k in stat_keys:
        s_info = across_video_summary[k]
        print(f"  {stat_names[k]:<24} | {s_info['mean']:<18.6f} | {s_info['std']:<18.6f}")
    print("=" * 78 + "\n")

    # -------------------------------------------------------------------------
    # 繪製圖表：呈現每條影片的非零資料點、單片 Max/Min/Mean/Std 與跨影片基準線
    # -------------------------------------------------------------------------
    fig_width = max(12, min(28, n_vids * 0.15))
    plt.figure(figsize=(fig_width, 7.5), dpi=150)

    plot_x_positions = list(range(1, n_vids + 1))

    for x_pos, rec in zip(plot_x_positions, per_video_records):
        vals = rec["frames"]
        x_coords = np.full(len(vals), x_pos)
        jitter = np.random.normal(0, 0.04, size=len(vals))

        # 1. 該影片所有非零影格散點
        plt.scatter(
            x_coords + jitter,
            vals,
            color="#1f77b4",
            alpha=0.30,
            s=16,
            edgecolors="none",
        )

        # 2. 單片 Mean ± 1 Std (紅色菱形 + 誤差線)
        plt.errorbar(
            x_pos,
            rec["mean"],
            yerr=rec["std"],
            fmt="D",
            color="#d62728",
            ecolor="#d62728",
            elinewidth=1.0,
            capsize=2.5,
            markersize=5.0,
            alpha=0.85,
            zorder=3,
        )

        # 3. 單片 Max (深橘色上三角) 與 Min (深藍色下三角)
        plt.scatter(x_pos, rec["max"], color="#ff7f0e", marker="^", s=28, alpha=0.85, zorder=4)
        plt.scatter(x_pos, rec["min"], color="#17becf", marker="v", s=28, alpha=0.85, zorder=4)

    # 繪製 Across-Video 的 Mean of Means 與 Mean of Max 基準線
    mean_of_means = across_video_summary["mean"]["mean"]
    std_of_means = across_video_summary["mean"]["std"]
    mean_of_max = across_video_summary["max"]["mean"]
    std_of_max = across_video_summary["max"]["std"]
    mean_of_min = across_video_summary["min"]["mean"]

    plt.axhline(
        mean_of_means,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        label=f"Across-Video Mean(Mean): {mean_of_means:.4f} ± {std_of_means:.4f}",
        zorder=5,
    )
    plt.axhline(
        mean_of_max,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.8,
        label=f"Across-Video Mean(Max):  {mean_of_max:.4f} ± {std_of_max:.4f}",
        zorder=5,
    )
    plt.axhline(
        mean_of_min,
        color="#17becf",
        linestyle=":",
        linewidth=1.5,
        label=f"Across-Video Mean(Min):  {mean_of_min:.4f}",
        zorder=5,
    )

    tier_tag = f"Video Tier: {tier.upper()}" if tier.lower() != "all" else "All Videos"
    title_text = (
        f"[{tier_tag}] Across-Video Summary of Within-Video (Max, Min, Mean, Std) — Zero-Frames Removed\n"
        f"Vertex #{vertex_id}  |  Mean(Mean): {mean_of_means:.4f}±{std_of_means:.4f}, "
        f"Mean(Max): {mean_of_max:.4f}±{std_of_max:.4f}, "
        f"Mean(Std): {across_video_summary['std']['mean']:.4f}±{across_video_summary['std']['std']:.4f}"
    )
    plt.title(title_text, fontsize=11.5, fontweight="bold", pad=14)
    plt.xlabel(f"Video Rank in [{tier_tag}] (N = {n_vids} Videos)", fontsize=11, labelpad=10)
    plt.ylabel(metric_labels[metric], fontsize=11, labelpad=10)
    plt.legend(loc="upper right", frameon=True, fontsize=9.5)

    if len(plot_x_positions) <= 50:
        plt.xticks(plot_x_positions, labels=[r["seq_idx"] for r in per_video_records])
    else:
        step = max(1, len(plot_x_positions) // 25)
        plt.xticks(
            plot_x_positions[::step],
            labels=[r["seq_idx"] for r in per_video_records[::step]],
        )

    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    output_img = f"vertex_{vertex_id}_{metric}_videolevel_4stats_{tier.lower()}.png"
    plt.savefig(output_img)
    print(f"圖表已成功生成並儲存至: {os.path.abspath(output_img)}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="過濾 mu=0 影格後，計算每支影片的 (Max, Min, Mean, Std)，再於 Across-Video 計算這 4 個統計量的 Mean 與 Std"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "good", "neutral", "bad"],
        help="選擇欲分析的影片品質級別 (預設: all，可選: good, neutral, bad)",
    )
    parser.add_argument(
        "--vertex",
        type=int,
        default=1,
        help="欲觀察的 Face Vertex ID (預設為 1，鼻尖點；範圍 0~467)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="abs_mu",
        choices=["abs_mu", "dilation", "teichmuller_distance"],
        help="測量指標: abs_mu, dilation, teichmuller_distance",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help=f"Reels 根目錄路徑 (預設: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--class_json",
        type=str,
        default=VIDEO_SUMMARY_JSON,
        help=f"影片分類 JSON 路徑 (預設: {VIDEO_SUMMARY_JSON})",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="限制分析的最大影片數量",
    )
    args = parser.parse_args()

    plot_video_deformation(
        root_dir=args.dir,
        tier=args.tier,
        vertex_id=args.vertex,
        metric=args.metric,
        max_videos=args.max_videos,
        summary_json_path=args.class_json,
    )