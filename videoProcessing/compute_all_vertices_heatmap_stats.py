import os
import glob
import json
import time
import urllib.request
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from config import DATA_PROC_DIR, VIDEO_PROC_DIR, DEFAULT_REELS_DIR

# 設定中文字型支援
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

VIDEO_SUMMARY_JSON = str(DATA_PROC_DIR / "video_classification_summary.json")
OUTPUT_STATS_JSON = str(VIDEO_PROC_DIR / "all_vertices_deformation_stats.json")

NUM_VERTICES = 468
METRICS = ["abs_mu", "dilation", "teichmuller_distance"]
WITHIN_STATS = ["max", "min", "mean", "std"]
TIERS = ["all", "good", "neutral", "bad"]

# MediaPipe 官方標準 468 點人臉網格模型
CANONICAL_OBJ_URL = (
    "https://raw.githubusercontent.com/google/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)
LOCAL_OBJ_FILE = str(VIDEO_PROC_DIR / "canonical_face_model.obj")


# ==============================================================================
# 1. 載入影片分組對照表與搜尋 *_qc.json
# ==============================================================================

def load_video_tier_lookup(summary_json_path: str) -> dict:
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(
            f"找不到 {summary_json_path}，請確認檔案路徑或先執行 video_residual_classification.py。"
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


def collect_matched_qc_files(root_dir: str, summary_json_path: str) -> list:
    lookup = load_video_tier_lookup(summary_json_path)
    pattern = os.path.join(root_dir, "**", "*_qc.json")
    all_qc_files = sorted(glob.glob(pattern, recursive=True))

    matched = []
    for qc_path in all_qc_files:
        folder_dir, fname = os.path.split(qc_path)
        account_id = os.path.basename(folder_dir)
        video_stem = fname[:-8] if fname.endswith("_qc.json") else os.path.splitext(fname)[0]

        v_info = lookup.get((account_id, video_stem))
        if v_info is None:
            continue

        matched.append((qc_path, v_info))

    return matched


# ==============================================================================
# 2. 單次讀取全頂點同步提取 (Single-Pass 468-Vertex Extraction)
# ==============================================================================

def extract_all_vertices_within_video_stats(qc_json_path: str, num_vertices: int = NUM_VERTICES, zero_eps: float = 1e-8):
    """
    只讀取一次 _qc.json，同時計算全部 468 個頂點在三個指標 (abs_mu, dilation, teichmuller_distance)
    下的單片內 (Max, Min, Mean, Std)。
    嚴格剔除 abs_mu <= zero_eps 的無效面片與影格。
    回傳: dict[metric][stat] -> shape (468,) 的 numpy array (若某頂點無有效幀則為 np.nan)
    """
    try:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = json.load(f)
    except Exception as e:
        print(f"  [錯誤] 無法讀取 {qc_json_path}: {e}")
        return None

    transitions = qc_data.get("consecutive_qc_metrics", [])
    if not transitions:
        return None

    # 為每個頂點收集所有有效影格的序列
    v_abs_mu_series = [[] for _ in range(num_vertices)]
    v_dilation_series = [[] for _ in range(num_vertices)]
    v_teich_series = [[] for _ in range(num_vertices)]

    for trans in transitions:
        if not trans.get("valid_transition", False):
            continue
        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            continue

        # 當前幀轉換的暫存累加器
        sum_mu = np.zeros(num_vertices, dtype=np.float64)
        sum_dil = np.zeros(num_vertices, dtype=np.float64)
        max_mu = np.zeros(num_vertices, dtype=np.float64)
        counts = np.zeros(num_vertices, dtype=np.int32)

        for fm in facet_metrics:
            mu_val = float(fm.get("abs_mu", 0.0))
            # 嚴格過濾 mu == 0 的無效面片
            if mu_val <= zero_eps:
                continue
            dil_val = float(fm.get("dilation", 1.0))
            verts = fm.get("vertices", [])

            for v in verts:
                if 0 <= v < num_vertices:
                    sum_mu[v] += mu_val
                    sum_dil[v] += dil_val
                    if mu_val > max_mu[v]:
                        max_mu[v] = mu_val
                    counts[v] += 1

        # 找出在此幀轉換中有非零有效面片的頂點
        active_verts = np.where(counts > 0)[0]
        for v in active_verts:
            c = counts[v]
            mean_local_mu = sum_mu[v] / c
            if mean_local_mu <= zero_eps:
                continue

            mean_local_dil = sum_dil[v] / c
            clamped_max_mu = min(max_mu[v], 0.999)
            local_dt = 0.5 * np.log((1.0 + clamped_max_mu) / (1.0 - clamped_max_mu))

            v_abs_mu_series[v].append(float(mean_local_mu))
            v_dilation_series[v].append(float(mean_local_dil))
            v_teich_series[v].append(float(local_dt))

    # 計算單片內 (Max, Min, Mean, Std)
    video_stats = {
        m: {s: np.full(num_vertices, np.nan, dtype=np.float64) for s in WITHIN_STATS}
        for m in METRICS
    }

    series_map = {
        "abs_mu": v_abs_mu_series,
        "dilation": v_dilation_series,
        "teichmuller_distance": v_teich_series,
    }

    has_any_valid = False
    for m_name, v_lists in series_map.items():
        for v in range(num_vertices):
            vals = v_lists[v]
            if not vals:
                continue
            has_any_valid = True
            arr = np.asarray(vals, dtype=np.float64)
            video_stats[m_name]["max"][v] = float(np.max(arr))
            video_stats[m_name]["min"][v] = float(np.min(arr))
            video_stats[m_name]["mean"][v] = float(np.mean(arr))
            video_stats[m_name]["std"][v] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return video_stats if has_any_valid else None


# ==============================================================================
# 3. Across-Video 跨影片彙總統計 (針對 4 個單片指標各自計算 Mean 與 Std)
# ==============================================================================

def compute_across_video_all_vertices(matched_qc_tuples: list, max_videos: int = None) -> dict:
    if max_videos is not None:
        matched_qc_tuples = matched_qc_tuples[:max_videos]

    # 容器：tier -> metric -> within_stat -> list of (468,) arrays
    tier_storage = {
        t: {m: {s: [] for s in WITHIN_STATS} for m in METRICS}
        for t in TIERS
    }
    tier_video_counts = {t: 0 for t in TIERS}

    total_files = len(matched_qc_tuples)
    start_time = time.time()

    for idx, (qc_path, v_info) in enumerate(matched_qc_tuples, start=1):
        vid_tier = str(v_info.get("video_classification", "neutral")).lower()
        rel_name = f"{v_info.get('account_id')}/{v_info.get('video_id')}"

        if idx % 50 == 0 or idx == 1 or idx == total_files:
            elapsed = time.time() - start_time
            print(f"  [{idx}/{total_files}] ({elapsed:.1f}s) 正在同步提取 468 頂點特徵: {rel_name} [{vid_tier.upper()}]")

        vid_stats = extract_all_vertices_within_video_stats(qc_path)
        if vid_stats is None:
            continue

        # 存入 "all" 以及該影片所屬的 tier ('good' / 'neutral' / 'bad')
        target_buckets = ["all"]
        if vid_tier in tier_storage:
            target_buckets.append(vid_tier)

        for b in target_buckets:
            tier_video_counts[b] += 1
            for m in METRICS:
                for s in WITHIN_STATS:
                    tier_storage[b][m][s].append(vid_stats[m][s])

    # 計算 Across-Video 的 Mean 與 Std
    final_results = {
        "metadata": {
            "num_vertices": NUM_VERTICES,
            "metrics": METRICS,
            "within_video_stats": WITHIN_STATS,
            "across_video_aggregations": ["mean", "std"],
            "tier_video_counts": tier_video_counts,
        },
        "tiers": {},
    }

    for t in TIERS:
        n_vids = tier_video_counts[t]
        if n_vids == 0:
            continue

        final_results["tiers"][t] = {
            "video_count": n_vids,
            " heatmap_arrays": {},   # 長度 468 的陣列，專供直接載入繪製 Face Mesh Heatmap
            "per_vertex_details": {}, # 按頂點 ID "0"~"467" 索引的詳細字典
        }

        # 初始化每個頂點的詳細字典
        for v in range(NUM_VERTICES):
            final_results["tiers"][t]["per_vertex_details"][str(v)] = {}

        for m in METRICS:
            final_results["tiers"][t]["heatmap_arrays"][m] = {}

            for s in WITHIN_STATS:
                # shape: (N_videos, 468)
                mat = np.vstack(tier_storage[t][m][s])

                # 忽略個別影片可能缺失的 nan，計算跨影片 Mean 與 Std (ddof=1)
                with np.errstate(invalid="ignore"):
                    across_mean = np.nanmean(mat, axis=0)
                    across_std = np.nanstd(mat, axis=0, ddof=1) if n_vids > 1 else np.zeros(NUM_VERTICES)

                across_mean = np.nan_to_num(across_mean, nan=0.0)
                across_std = np.nan_to_num(across_std, nan=0.0)

                # 儲存為一維長度 468 的陣列 (例如 mean_of_max, std_of_max, mean_of_mean, std_of_mean...)
                final_results["tiers"][t]["heatmap_arrays"][m][f"mean_of_{s}"] = [
                    round(float(x), 6) for x in across_mean
                ]
                final_results["tiers"][t]["heatmap_arrays"][m][f"std_of_{s}"] = [
                    round(float(x), 6) for x in across_std
                ]

                # 同步寫入個別頂點字典
                for v in range(NUM_VERTICES):
                    if m not in final_results["tiers"][t]["per_vertex_details"][str(v)]:
                        final_results["tiers"][t]["per_vertex_details"][str(v)][m] = {}
                    final_results["tiers"][t]["per_vertex_details"][str(v)][m][s] = {
                        "mean": round(float(across_mean[v]), 6),
                        "std": round(float(across_std[v]), 6),
                    }

    return final_results


# ==============================================================================
# 4. 在標準 468 點人臉網格模板上繪製統計熱力圖 (Face Mesh Heatmap Visualizer)
# ==============================================================================

def load_canonical_face_mesh():
    """下載或讀取 MediaPipe 官方標準 468 點人臉 OBJ 模型"""
    if not os.path.exists(LOCAL_OBJ_FILE):
        print(f"正在下載官方 MediaPipe 標準 468 點模型 ({LOCAL_OBJ_FILE})...")
        urllib.request.urlretrieve(CANONICAL_OBJ_URL, LOCAL_OBJ_FILE)

    vertices = []
    faces = []
    with open(LOCAL_OBJ_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face_indices = [int(p.split("/")[0]) - 1 for p in parts]
                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def plot_mesh_heatmap_comparison(
    stats_data: dict,
    metric: str = "teichmuller_distance",
    stat_field: str = "mean_of_mean",
    output_dir: str = "face_heatmaps",
):
    """
    將指定的 metric 與 stat_field (例如 'mean_of_max', 'std_of_max', 'mean_of_mean', 'std_of_mean' 等)
    投影在標準 468 點人臉三角網格上，一次繪出：
    [Panel 1: GOOD Tier] | [Panel 2: NEUTRAL Tier] | [Panel 3: BAD Tier] | [Panel 4: GOOD - BAD 差異圖]
    """
    os.makedirs(output_dir, exist_ok=True)
    vertices, faces = load_canonical_face_mesh()
    x, y = vertices[:, 0], vertices[:, 1]
    triang = mtri.Triangulation(x, y, faces)

    tiers_Data = stats_data.get("tiers", {})
    if not all(k in tiers_Data for k in ["good", "neutral", "bad"]):
        print("  [提示] JSON 中尚未集齊 good / neutral / bad 三組資料，僅繪製現有組別。")
        return

    arr_good = np.array(tiers_Data["good"]["heatmap_arrays"][metric][stat_field])
    arr_neutral = np.array(tiers_Data["neutral"]["heatmap_arrays"][metric][stat_field])
    arr_bad = np.array(tiers_Data["bad"]["heatmap_arrays"][metric][stat_field])
    arr_diff = arr_good - arr_bad

    # 統一前三張圖的色標範圍 (使用 2% ~ 98% 分位數避免單一邊界點主導顏色)
    combined_vals = np.concatenate([arr_good, arr_neutral, arr_bad])
    vmin = float(np.percentile(combined_vals, 2))
    vmax = float(np.percentile(combined_vals, 98))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=200)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], arr_good, f"GOOD Tier (N={tiers_Data['good']['video_count']})", "inferno", vmin, vmax),
        (axes[1], arr_neutral, f"NEUTRAL Tier (N={tiers_Data['neutral']['video_count']})", "inferno", vmin, vmax),
        (axes[2], arr_bad, f"BAD Tier (N={tiers_Data['bad']['video_count']})", "inferno", vmin, vmax),
    ]

    for ax, vals, title, cmap_name, v_min, v_max in panels:
        tpc = ax.tripcolor(triang, vals, shading="gouraud", cmap=cmap_name, vmin=v_min, vmax=v_max)
        ax.triplot(triang, color="white", linewidth=0.18, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.03)

    # Panel 4: GOOD - BAD 差異熱力圖 (紅正藍負，0 居中)
    ax_diff = axes[3]
    max_abs_diff = float(np.percentile(np.abs(arr_diff), 98)) or 1e-4
    tpc_diff = ax_diff.tripcolor(
        triang, arr_diff, shading="gouraud", cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff
    )
    ax_diff.triplot(triang, color="black", linewidth=0.15, alpha=0.2)
    ax_diff.set_title("Difference: (GOOD - BAD)\n[Red: Good > Bad | Blue: Good < Bad]", fontsize=11.5, fontweight="bold", pad=10)
    ax_diff.set_aspect("equal")
    ax_diff.axis("off")
    plt.colorbar(tpc_diff, ax=ax_diff, fraction=0.046, pad=0.03)

    plt.suptitle(
        f"468-Vertex Face Mesh Heatmap  |  Metric: [{metric}]  |  Statistic: [{stat_field}]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = os.path.join(output_dir, f"heatmap_{metric}_{stat_field}.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 已生成人臉網格熱力圖: {os.path.abspath(out_png)}")


# ==============================================================================
# 5. 主程序入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="全頂點 (0~467) 幾何形變統計預算與 Face Mesh 熱力圖生成系統"
    )
    parser.add_argument("--dir", type=str, default=DEFAULT_REELS_DIR, help="Reels 根目錄路徑")
    parser.add_argument("--class_json", type=str, default=VIDEO_SUMMARY_JSON, help="影片分類 JSON 路徑")
    parser.add_argument("--output_json", type=str, default=OUTPUT_STATS_JSON, help="輸出的全頂點統計 JSON 路徑")
    parser.add_argument("--max_videos", type=int, default=None, help="限制處理影片數 (測試用)")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="若已算好 all_vertices_deformation_stats.json，加上此參數可直接跳過運算並繪製熱力圖",
    )
    parser.add_argument(
        "--plot_all_combinations",
        action="store_true",
        help="自動將 3 個指標 x 8 種統計量 = 24 張全臉熱力圖一次全部輸出至 face_heatmaps 資料夾",
    )
    args = parser.parse_args()

    if args.plot_only and os.path.exists(args.output_json):
        print(f"直接載入已存在的統計結果: {args.output_json}")
        with open(args.output_json, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
    else:
        print("=" * 78)
        print("【全臉 468 頂點 x 3 指標 x 4 單片統計量 — Across-Video 批次運算系統】")
        print(f"  - 掃描目錄: {args.dir}")
        print(f"  - 分類依據: {args.class_json}")
        print("=" * 78)

        matched_qc = collect_matched_qc_files(args.dir, args.class_json)
        print(f"共配對到 {len(matched_qc)} 個有效的 *_qc.json 檔案，開始單次遍歷全頂點提取...\n")

        stats_data = compute_across_video_all_vertices(matched_qc, max_videos=args.max_videos)

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 78)
        print(f"全臉 468 頂點統計數據已完整儲存至: {os.path.abspath(args.output_json)}")
        print("=" * 78)

    # 自動生成熱力圖
    print("\n開始繪製 Canonical Face Mesh 熱力圖...")
    all_stat_fields = [
        "mean_of_max", "std_of_max",
        "mean_of_min", "std_of_min",
        "mean_of_mean", "std_of_mean",
        "mean_of_std", "std_of_std",
    ]

    if args.plot_all_combinations:
        for m in METRICS:
            for sf_name in all_stat_fields:
                plot_mesh_heatmap_comparison(stats_data, metric=m, stat_field=sf_name)
    else:
        # 預設輸出最具代表性的 6 張核心熱力圖 (3 指標 x [mean_of_mean, mean_of_max, std_of_mean])
        for m in ["abs_mu", "teichmuller_distance"]:
            for sf_name in ["mean_of_mean", "mean_of_max", "std_of_mean", "mean_of_std"]:
                plot_mesh_heatmap_comparison(stats_data, metric=m, stat_field=sf_name)

    print("\n所有處理已完成！")


if __name__ == "__main__":
    main()