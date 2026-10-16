import os
import glob
import json
import time
import argparse
import numpy as np

DEFAULT_REELS_DIR = r"D:\reels"
VIDEO_SUMMARY_JSON = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\dataProcessing\video_classification_summary.json"
OUTPUT_STATS_JSON = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\videoProcessing\all_vertices_deformation_stats.json"

NUM_VERTICES = 468
METRICS = ["abs_mu", "dilation", "teichmuller_distance"]
WITHIN_STATS = ["max", "min", "mean", "std"]
TIERS = ["all", "good", "neutral", "bad"]


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


def extract_all_vertices_within_video_stats(qc_json_path: str, num_vertices: int = NUM_VERTICES, zero_eps: float = 1e-8):
    """
    單次掃描單一 _qc.json，提取全部 468 頂點的單片 (Max, Min, Mean, Std)。
    嚴格剔除 abs_mu <= zero_eps 的無效面片與影格。
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

    v_abs_mu_series = [[] for _ in range(num_vertices)]
    v_dilation_series = [[] for _ in range(num_vertices)]
    v_teich_series = [[] for _ in range(num_vertices)]

    for trans in transitions:
        if not trans.get("valid_transition", False):
            continue
        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            continue

        sum_mu = np.zeros(num_vertices, dtype=np.float64)
        sum_dil = np.zeros(num_vertices, dtype=np.float64)
        max_mu = np.zeros(num_vertices, dtype=np.float64)
        counts = np.zeros(num_vertices, dtype=np.int32)

        for fm in facet_metrics:
            mu_val = float(fm.get("abs_mu", 0.0))
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


def compute_and_save_all_stats(root_dir: str, class_json: str, output_json: str, max_videos: int = None):
    print("=" * 78)
    print("【全臉 468 頂點 x 3 指標 x 4 單片統計量 — Across-Video 計算腳本】")
    print(f"  - 影片根目錄: {root_dir}")
    print(f"  - 分類檔案:   {class_json}")
    print(f"  - 輸出目標:   {output_json}")
    print("=" * 78)

    matched_qc = collect_matched_qc_files(root_dir, class_json)
    if not matched_qc:
        print("未找到任何符合條件的 *_qc.json 檔案。")
        return

    if max_videos is not None:
        matched_qc = matched_qc[:max_videos]

    print(f"共配對到 {len(matched_qc)} 個有效的 *_qc.json 檔案，開始單次遍歷全頂點提取...\n")

    tier_storage = {
        t: {m: {s: [] for s in WITHIN_STATS} for m in METRICS}
        for t in TIERS
    }
    tier_video_counts = {t: 0 for t in TIERS}

    total_files = len(matched_qc)
    start_time = time.time()

    for idx, (qc_path, v_info) in enumerate(matched_qc, start=1):
        vid_tier = str(v_info.get("video_classification", "neutral")).lower()
        rel_name = f"{v_info.get('account_id')}/{v_info.get('video_id')}"

        if idx % 50 == 0 or idx == 1 or idx == total_files:
            elapsed = time.time() - start_time
            print(f"  [{idx}/{total_files}] ({elapsed:.1f}s) 處理中: {rel_name} [{vid_tier.upper()}]")

        vid_stats = extract_all_vertices_within_video_stats(qc_path)
        if vid_stats is None:
            continue

        target_buckets = ["all"]
        if vid_tier in tier_storage:
            target_buckets.append(vid_tier)

        for b in target_buckets:
            tier_video_counts[b] += 1
            for m in METRICS:
                for s in WITHIN_STATS:
                    tier_storage[b][m][s].append(vid_stats[m][s])

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
            "heatmap_arrays": {},
            "per_vertex_details": {str(v): {} for v in range(NUM_VERTICES)},
        }

        for m in METRICS:
            final_results["tiers"][t]["heatmap_arrays"][m] = {}

            for s in WITHIN_STATS:
                mat = np.vstack(tier_storage[t][m][s])

                with np.errstate(invalid="ignore"):
                    across_mean = np.nanmean(mat, axis=0)
                    across_std = np.nanstd(mat, axis=0, ddof=1) if n_vids > 1 else np.zeros(NUM_VERTICES)

                across_mean = np.nan_to_num(across_mean, nan=0.0)
                across_std = np.nan_to_num(across_std, nan=0.0)

                final_results["tiers"][t]["heatmap_arrays"][m][f"mean_of_{s}"] = [
                    round(float(x), 6) for x in across_mean
                ]
                final_results["tiers"][t]["heatmap_arrays"][m][f"std_of_{s}"] = [
                    round(float(x), 6) for x in across_std
                ]

                for v in range(NUM_VERTICES):
                    if m not in final_results["tiers"][t]["per_vertex_details"][str(v)]:
                        final_results["tiers"][t]["per_vertex_details"][str(v)][m] = {}
                    final_results["tiers"][t]["per_vertex_details"][str(v)][m][s] = {
                        "mean": round(float(across_mean[v]), 6),
                        "std": round(float(across_std[v]), 6),
                    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("【計算完成並已儲存！】")
    print(f"  - 輸出檔案: {os.path.abspath(output_json)}")
    print(f"  - 涵蓋 Tier 樣本數: {tier_video_counts}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全頂點形變兩階段統計計算並輸出 JSON 檔")
    parser.add_argument("--dir", type=str, default=DEFAULT_REELS_DIR, help="Reels 根目錄路徑")
    parser.add_argument("--class_json", type=str, default=VIDEO_SUMMARY_JSON, help="影片分類 JSON 路徑")
    parser.add_argument("--output_json", type=str, default=OUTPUT_STATS_JSON, help="輸出的全頂點統計 JSON 路徑")
    parser.add_argument("--max_videos", type=int, default=None, help="限制處理檔案數 (測試用)")
    args = parser.parse_args()

    compute_and_save_all_stats(
        root_dir=args.dir,
        class_json=args.class_json,
        output_json=args.output_json,
        max_videos=args.max_videos,
    )