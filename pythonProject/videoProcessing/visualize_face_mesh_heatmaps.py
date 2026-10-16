import os
import json
import urllib.request
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 設定中文字型支援
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_STATS_JSON = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\videoProcessing\all_vertices_deformation_stats.json"
OUTPUT_DIR = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\videoProcessing\face_heatmaps"

CANONICAL_OBJ_URL = (
    "https://raw.githubusercontent.com/google/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)
LOCAL_OBJ_FILE = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\videoProcessing\canonical_face_model.obj"

METRICS = ["abs_mu", "dilation", "teichmuller_distance"]
STAT_FIELDS = [
    "mean_of_max", "std_of_max",
    "mean_of_min", "std_of_min",
    "mean_of_mean", "std_of_mean",
    "mean_of_std", "std_of_std",
]


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


def plot_single_mesh_heatmap(
    stats_data: dict,
    triang: mtri.Triangulation,
    metric: str = "teichmuller_distance",
    stat_field: str = "mean_of_mean",
    output_dir: str = OUTPUT_DIR,
):
    """
    將指定的指標 (metric) 與統計欄位 (stat_field) 投影至標準人臉網格上，產生四聯圖：
    [GOOD] | [NEUTRAL] | [BAD] | [GOOD - BAD 差異圖]
    """
    tiers_data = stats_data.get("tiers", {})
    if not all(k in tiers_data for k in ["good", "neutral", "bad"]):
        print("  [錯誤] JSON 中未包含完整的 good, neutral, bad 資料。")
        return

    arr_good = np.array(tiers_data["good"]["heatmap_arrays"][metric][stat_field])
    arr_neutral = np.array(tiers_data["neutral"]["heatmap_arrays"][metric][stat_field])
    arr_bad = np.array(tiers_data["bad"]["heatmap_arrays"][metric][stat_field])
    arr_diff = arr_good - arr_bad

    combined_vals = np.concatenate([arr_good, arr_neutral, arr_bad])
    vmin = float(np.percentile(combined_vals, 2))
    vmax = float(np.percentile(combined_vals, 98))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=200)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], arr_good, f"GOOD Tier (N={tiers_data['good']['video_count']})", "inferno", vmin, vmax),
        (axes[1], arr_neutral, f"NEUTRAL Tier (N={tiers_data['neutral']['video_count']})", "inferno", vmin, vmax),
        (axes[2], arr_bad, f"BAD Tier (N={tiers_data['bad']['video_count']})", "inferno", vmin, vmax),
    ]

    for ax, vals, title, cmap_name, v_min, v_max in panels:
        tpc = ax.tripcolor(triang, vals, shading="gouraud", cmap=cmap_name, vmin=v_min, vmax=v_max)
        ax.triplot(triang, color="white", linewidth=0.18, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.03)

    ax_diff = axes[3]
    max_abs_diff = float(np.percentile(np.abs(arr_diff), 98)) or 1e-4
    tpc_diff = ax_diff.tripcolor(
        triang, arr_diff, shading="gouraud", cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff
    )
    ax_diff.triplot(triang, color="black", linewidth=0.15, alpha=0.2)
    ax_diff.set_title("Difference: (GOOD - BAD)\n[紅: Good > Bad | 藍: Good < Bad]", fontsize=11.5, fontweight="bold", pad=10)
    ax_diff.set_aspect("equal")
    ax_diff.axis("off")
    plt.colorbar(tpc_diff, ax=ax_diff, fraction=0.046, pad=0.03)

    plt.suptitle(
        f"468-Vertex Face Mesh Heatmap  |  Metric: [{metric}]  |  Field: [{stat_field}]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = os.path.join(output_dir, f"heatmap_{metric}_{stat_field}.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 已輸出: {os.path.abspath(out_png)}")


def main():
    parser = argparse.ArgumentParser(description="讀取統計 JSON 檔案並在標準 Face Mesh 上展示熱力圖")
    parser.add_argument("--json", type=str, default=DEFAULT_STATS_JSON, help="統計 JSON 檔案路徑")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="熱力圖圖片輸出目錄")
    parser.add_argument(
        "--metric",
        type=str,
        default="teichmuller_distance",
        choices=METRICS,
        help="欲繪製的幾何指標 (預設: teichmuller_distance)",
    )
    parser.add_argument(
        "--stat_field",
        type=str,
        default="mean_of_mean",
        choices=STAT_FIELDS,
        help="欲繪製的統計欄位 (預設: mean_of_mean)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="批次輸出全部 3 指標 x 8 統計量 = 24 張熱力圖",
    )
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"錯誤：找不到統計檔案 {args.json}。請先執行 compute_all_vertices_stats.py。")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("【Face Mesh 熱力圖視覺化系統】")
    print(f"  - 讀取資料: {args.json}")
    print(f"  - 輸出目錄: {args.output_dir}")
    print("=" * 70)

    with open(args.json, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    vertices, faces = load_canonical_face_mesh()
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], faces)

    if args.all:
        print("開始批次生成全部 24 張熱力圖組合...\n")
        for m in METRICS:
            for sf in STAT_FIELDS:
                plot_single_mesh_heatmap(stats_data, triang, metric=m, stat_field=sf, output_dir=args.output_dir)
    else:
        print(f"生成指定熱力圖: [{args.metric}] - [{args.stat_field}] ...\n")
        plot_single_mesh_heatmap(stats_data, triang, metric=args.metric, stat_field=args.stat_field, output_dir=args.output_dir)

    print("\n繪圖作業已完成！")


if __name__ == "__main__":
    main()