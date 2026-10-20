import os
import json
import urllib.request
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 中文字型支援
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_INPUT_STATS = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\videoProcessing\all_vertices_deformation_stats.json"
DEFAULT_OUTPUT_STATS = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\videoProcessing\all_vertices_deformation_stats_symmetric.json"
DEFAULT_HEATMAP_DIR = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\\videoProcessing\face_heatmaps_symmetric"

CANONICAL_OBJ_URL = (
    "https://raw.githubusercontent.com/google/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)
LOCAL_OBJ_FILE = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\videoProcessing\canonical_face_model.obj"

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


def build_bilateral_symmetry_mapping(vertices: np.ndarray, mid_line_tol: float = 0.8) -> np.ndarray:
    """
    依據 MediaPipe 標準 3D 人臉頂點的空間座標 (X 軸鏡像: x -> -x)，
    自動構建 468 點左右精確配對表。
    中軸線上的頂點對應自身。
    """
    num_v = len(vertices)
    paired_index = np.arange(num_v, dtype=np.int32)

    for i in range(num_v):
        x, y, z = vertices[i]
        # 若非常接近中軸線 X=0，則視為中央矢狀面點 (Sagittal Midline)
        if abs(x) < mid_line_tol:
            paired_index[i] = i
            continue

        target_x = -x
        target_y = y
        target_z = z

        dists = (vertices[:, 0] - target_x) ** 2 + \
                (vertices[:, 1] - target_y) ** 2 + \
                (vertices[:, 2] - target_z) ** 2
        best_match = int(np.argmin(dists))
        paired_index[i] = best_match

    return paired_index


def apply_bilateral_folding_to_stats(raw_stats: dict, pair_map: np.ndarray) -> dict:
    """
    將所有 Tier 的 heatmap_arrays 進行左右鏡像對稱平均 (Bilateral Folding)。
    """
    sym_stats = json.loads(json.dumps(raw_stats))  # 深拷貝
    tiers = sym_stats.get("tiers", {})

    for t_name, t_data in tiers.items():
        heatmap_arrays = t_data.get("heatmap_arrays", {})
        for metric, fields in heatmap_arrays.items():
            for sf_name, val_list in fields.items():
                arr = np.array(val_list, dtype=np.float64)
                # 執行對稱平均折疊
                folded_arr = 0.5 * (arr + arr[pair_map])
                fields[sf_name] = [round(float(x), 6) for x in folded_arr]

        # 更新 per_vertex_details
        per_v = t_data.get("per_vertex_details", {})
        for v_str in per_v:
            v_idx = int(v_str)
            paired_v_idx = int(pair_map[v_idx])
            for metric in per_v[v_str]:
                for sf in per_v[v_str][metric]:
                    mean_val = 0.5 * (
                        raw_stats["tiers"][t_name]["per_vertex_details"][str(v_idx)][metric][sf]["mean"] +
                        raw_stats["tiers"][t_name]["per_vertex_details"][str(paired_v_idx)][metric][sf]["mean"]
                    )
                    std_val = 0.5 * (
                        raw_stats["tiers"][t_name]["per_vertex_details"][str(v_idx)][metric][sf]["std"] +
                        raw_stats["tiers"][t_name]["per_vertex_details"][str(paired_v_idx)][metric][sf]["std"]
                    )
                    per_v[v_str][metric][sf] = {
                        "mean": round(float(mean_val), 6),
                        "std": round(float(std_val), 6),
                    }

    sym_stats["metadata"]["bilateral_symmetry_applied"] = True
    return sym_stats


def plot_single_mesh_heatmap(
    stats_data: dict,
    triang: mtri.Triangulation,
    metric: str = "abs_mu",
    stat_field: str = "mean_of_max",
    output_dir: str = DEFAULT_HEATMAP_DIR,
):
    """
    繪製對稱化後的四聯人臉熱力圖：
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

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=300)
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
    max_abs_diff = float(np.percentile(np.abs(arr_diff), 98))
    if max_abs_diff < 1e-5:
        max_abs_diff = float(np.max(np.abs(arr_diff))) or 1e-4

    tpc_diff = ax_diff.tripcolor(
        triang, arr_diff, shading="gouraud", cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff
    )
    ax_diff.triplot(triang, color="black", linewidth=0.15, alpha=0.2)
    ax_diff.set_title("Symmetric Difference (GOOD - BAD)\n[紅: Good > Bad | 藍: Good < Bad]", fontsize=11.5, fontweight="bold", pad=10)
    ax_diff.set_aspect("equal")
    ax_diff.axis("off")
    plt.colorbar(tpc_diff, ax=ax_diff, fraction=0.046, pad=0.03)

    plt.suptitle(
        f"Bilateral Folded Face Mesh Heatmap  |  Metric: [{metric}]  |  Field: [{stat_field}]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = os.path.join(output_dir, f"symmetric_heatmap_{metric}_{stat_field}.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 已輸出對稱熱力圖: {os.path.abspath(out_png)}")


def main():
    parser = argparse.ArgumentParser(description="執行雙側對稱折疊 (Bilateral Symmetry Folding) 並展示人臉熱力圖")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_STATS, help="原始統計 JSON 路徑")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_STATS, help="對稱折疊後輸出的 JSON 路徑")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_HEATMAP_DIR, help="熱力圖圖片輸出目錄")
    parser.add_argument("--metric", type=str, default="abs_mu", choices=METRICS, help="欲展示的幾何指標")
    parser.add_argument("--stat_field", type=str, default="mean_of_max", choices=STAT_FIELDS, help="欲展示的統計欄位")
    parser.add_argument("--all", action="store_true", help="批次繪製全部 24 種指標組合熱力圖")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"錯誤：找不到輸入檔案 {args.input_json}，請確認路徑或先執行 compute_all_vertices_stats.py。")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("【人臉網格雙側對稱折疊 (Bilateral Symmetry Folding) 與視覺化系統】")
    print(f"  - 讀取來源: {args.input_json}")
    print(f"  - 對稱 JSON 輸出: {args.output_json}")
    print(f"  - 圖片輸出目錄: {args.output_dir}")
    print("=" * 80)

    # 1. 讀取 Canonical 模型並建立對稱映射
    vertices, faces = load_canonical_face_mesh()
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    pair_map = build_bilateral_symmetry_mapping(vertices)
    print(f"已成功構建 468 點空間雙側對稱鏡像映射表。")

    # 2. 讀取原始統計並折疊
    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_stats = json.load(f)

    sym_stats = apply_bilateral_folding_to_stats(raw_stats, pair_map)

    # 3. 儲存對稱折疊後的 JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(sym_stats, f, ensure_ascii=False, indent=2)
    print(f"雙側折疊數據已儲存至: {os.path.abspath(args.output_json)}\n")

    # 4. 繪製並輸出熱力圖
    if args.all:
        print("開始批次輸出全部 24 張對稱人臉熱力圖...")
        for m in METRICS:
            for sf in STAT_FIELDS:
                plot_single_mesh_heatmap(sym_stats, triang, metric=m, stat_field=sf, output_dir=args.output_dir)
    else:
        print(f"生成指定對稱熱力圖: [{args.metric}] - [{args.stat_field}] ...")
        plot_single_mesh_heatmap(sym_stats, triang, metric=args.metric, stat_field=args.stat_field, output_dir=args.output_dir)

    print("\n繪圖作業已完成！")


if __name__ == "__main__":
    main()