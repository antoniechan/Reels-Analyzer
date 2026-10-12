import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DIR = r"D:\reels"


def load_qc_data(qc_json_path):
    """讀取單一 _qc.json 檔案"""
    try:
        with open(qc_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"無法讀取檔案 {qc_json_path}: {e}")
        return None


def extract_vertex_measurements(qc_data, vertex_id, metric="abs_mu"):
    """
    從一段影片的 QC 資料中，提取特定頂點在所有連續幀轉換中的指定指標數值。
    """
    values = []
    transitions = qc_data.get("consecutive_qc_metrics", [])

    for trans in transitions:
        if not trans.get("valid_transition", False):
            continue

        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            continue

        # 找出所有包含此 vertex_id 的相鄰三角形面片 (1-ring neighbor faces)
        neighbor_facets = [
            fm for fm in facet_metrics
            if vertex_id in fm.get("vertices", [])
        ]

        if not neighbor_facets:
            continue

        if metric == "abs_mu":
            val = np.mean([fm["abs_mu"] for fm in neighbor_facets])
            values.append(float(val))

        elif metric == "dilation":
            val = np.mean([fm["dilation"] for fm in neighbor_facets])
            values.append(float(val))

        elif metric == "teichmuller_distance":
            max_local_mu = np.max([fm["abs_mu"] for fm in neighbor_facets])
            max_local_mu = min(max_local_mu, 0.999)
            local_dt = 0.5 * np.log((1.0 + max_local_mu) / (1.0 - max_local_mu))
            values.append(float(local_dt))

    return values


def parse_target_folders(folder_args):
    """
    解析命令列輸入的 folder 參數（支援多個名稱或逗號分隔）
    """
    if not folder_args:
        return ["all"]

    parsed = []
    for item in folder_args:
        for sub in item.split(","):
            sub_clean = sub.strip()
            if sub_clean:
                parsed.append(sub_clean)

    if not parsed or any(f.lower() == "all" for f in parsed):
        return ["all"]
    return parsed


def plot_video_deformation(root_dir=DEFAULT_DIR, target_folders=None, vertex_id=1, metric="abs_mu", max_videos=None):
    """
    依據選用的資料夾與頂點提取變形指標並繪圖
    """
    metric_labels = {
        "abs_mu": r"Shear Strain $|\mu|$ (Beltrami Coefficient)",
        "dilation": r"Maximal Dilation $K = \frac{1+|\mu|}{1-|\mu|}$",
        "teichmuller_distance": r"Teichmüller Distance $d_T$"
    }

    if metric not in metric_labels:
        raise ValueError(f"不支援的測量指標: {metric}。請選擇: {list(metric_labels.keys())}")

    folders_list = parse_target_folders(target_folders)
    is_all = (folders_list == ["all"])

    qc_files = []
    folder_summary = {}

    # 1. 檢索符合條件的檔案
    if is_all:
        pattern = os.path.join(root_dir, "**", "*_qc.json")
        qc_files = sorted(glob.glob(pattern, recursive=True))
    else:
        for fld in folders_list:
            pattern = os.path.join(root_dir, fld, "**", "*_qc.json")
            matched = sorted(glob.glob(pattern, recursive=True))
            qc_files.extend(matched)
            folder_summary[fld] = len(matched)
        qc_files = sorted(list(set(qc_files)))

    # 2. Console 終端機顯示詳細清單
    print("=" * 65)
    print("【執行參數與選定目錄資訊】")
    if is_all:
        print("  - 選用範圍 (Folders): 全部資料夾 (All Folders)")
    else:
        print(f"  - 選用資料夾數量: {len(folders_list)} 個")
        print("  - 詳細選用 Folder 清單:")
        for idx, fld in enumerate(folders_list, 1):
            count = folder_summary.get(fld, 0)
            status = f"{count} 個檔案" if count > 0 else "警告: 未找到任何 _qc.json"
            print(f"      [{idx}] {fld} ({status})")

    print(f"  - 特徵點編號 (Vertex ID): #{vertex_id}")
    print(f"  - 測量指標 (Measurement): {metric} ({metric_labels[metric]})")
    print(f"  - 總掃描到符合條件的 QC 檔案數: {len(qc_files)}")
    print("=" * 65 + "\n")

    if not qc_files:
        print("錯誤：未找到任何可讀取的 *_qc.json 檔案，程式結束。")
        return

    if max_videos is not None:
        qc_files = qc_files[:max_videos]

    video_indices = []
    all_values_per_video = []

    for idx, path in enumerate(qc_files, start=1):
        qc_data = load_qc_data(path)
        if qc_data is None:
            continue

        vals = extract_vertex_measurements(qc_data, vertex_id, metric=metric)
        if vals:
            video_indices.append(idx)
            all_values_per_video.append(vals)

    if not all_values_per_video:
        print("未提取到任何有效頂點數據，請確認頂點 ID 是否存在於網格面片中。")
        return

    # ---------------------------------------------------------
    # 3. 繪製圖表
    # ---------------------------------------------------------
    fig_width = max(12, len(video_indices) * 0.4)
    plt.figure(figsize=(fig_width, 7), dpi=150)

    for x_pos, vals in zip(video_indices, all_values_per_video):
        x_coords = np.full(len(vals), x_pos)
        jitter = np.random.normal(0, 0.04, size=len(vals))

        plt.scatter(
            x_coords + jitter,
            vals,
            color="#1f77b4",
            alpha=0.45,
            s=22,
            edgecolors="none"
        )

        mean_val = np.mean(vals)
        plt.scatter(x_pos, mean_val, color="#d62728", s=32, marker="D", zorder=3)

    if is_all:
        display_name = "All Folders"
    elif len(folders_list) <= 3:
        display_name = ", ".join(folders_list)
    else:
        display_name = f"{folders_list[0]} 等 {len(folders_list)} 個資料夾"

    title_text = (
        f"Folder: [{display_name}]  |  Vertex #{vertex_id}\n"
        f"Measurement: {metric_labels[metric]}"
    )
    plt.title(title_text, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel(f"Video Sequence in [{display_name}]", fontsize=11, labelpad=10)
    plt.ylabel(metric_labels[metric], fontsize=11, labelpad=10)

    if len(video_indices) <= 50:
        plt.xticks(video_indices, labels=video_indices)
    else:
        step = max(1, len(video_indices) // 25)
        plt.xticks(video_indices[::step], labels=video_indices[::step])

    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    # 檔名標籤處理
    clean_tag = "All" if is_all else "_".join(folders_list[:2])
    if len(folders_list) > 2:
        clean_tag += f"_plus_{len(folders_list) - 2}"
    output_img = f"vertex_{vertex_id}_{metric}_{clean_tag}.png"
    plt.savefig(output_img)
    print(f"\n圖表已成功生成並儲存至: {os.path.abspath(output_img)}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="統計並繪製指定資料夾中人臉頂點的幾何變形指標分佈圖")
    parser.add_argument(
        "--folder",
        nargs="+",
        default=["all"],
        help="指定要觀察的資料夾名稱，可指定多個（例如: --folder chang_ricky_life 或 --folder fld1 fld2），預設為 all"
    )
    parser.add_argument(
        "--vertex",
        type=int,
        default=1,
        help="欲觀察的 Face Vertex ID (預設為 1，鼻尖點；範圍通常在 0~467)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="abs_mu",
        choices=["abs_mu", "dilation", "teichmuller_distance"],
        help="測量指標: abs_mu, dilation, teichmuller_distance"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help=f"Reels 根目錄路徑 (預設: {DEFAULT_DIR})"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="限制繪製的最大影片數量"
    )
    args = parser.parse_args()

    plot_video_deformation(
        root_dir=args.dir,
        target_folders=args.folder,
        vertex_id=args.vertex,
        metric=args.metric,
        max_videos=args.max_videos
    )