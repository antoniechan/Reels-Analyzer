import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DIR = r"D:\reels"
ACCOUNTS_DATA_DIR = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\accountsData"
SUMMARY_JSON = "account_classification_summary.json"


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

        # 找出包含此 vertex_id 的相鄰三角形面片 (1-ring neighbor faces)
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


def get_accounts_by_tier(target_tier="all"):
    """
    依據指定的 tier ('good', 'neutral', 'bad', 'all') 取得符合條件的 account_id 列表
    """
    target_tier = target_tier.lower()
    if target_tier == "all":
        return None

    matched_accounts = []

    # 1. 優先嘗試讀取 account_classification_summary.json
    if os.path.exists(SUMMARY_JSON):
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            for item in summary_data:
                if str(item.get("classification", "")).lower() == target_tier:
                    matched_accounts.append(item["account_id"])
            return sorted(list(set(matched_accounts)))
        except Exception as e:
            print(f"讀取 {SUMMARY_JSON} 失敗 ({e})，改由 accountsData 直接檢索。")

    # 2. 回退方案：直接讀取 accountsData 目錄下的所有 JSON 檔案
    if os.path.exists(ACCOUNTS_DATA_DIR):
        pattern = os.path.join(ACCOUNTS_DATA_DIR, "*.json")
        for fpath in glob.glob(pattern):
            if os.path.basename(fpath) in [SUMMARY_JSON, "accounts_followers.json"]:
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    acc = json.load(f)
                clf = acc.get("quality_evaluation", {}).get("classification")
                if clf and str(clf).lower() == target_tier:
                    acc_id = acc.get("account_id", os.path.splitext(os.path.basename(fpath))[0])
                    matched_accounts.append(acc_id)
            except Exception:
                continue

    return sorted(list(set(matched_accounts)))


def plot_video_deformation(root_dir=DEFAULT_DIR, tier="all", target_folders=None, vertex_id=1, metric="abs_mu",
                           max_videos=None, trim_percent=5.0):
    """
    依據選用的 tier/folder 與頂點提取幾何指標、剔除最高與最低 5% 的 Outliers 後計算統計並繪圖
    """
    metric_labels = {
        "abs_mu": r"Shear Strain $|\mu|$ (Beltrami Coefficient)",
        "dilation": r"Maximal Dilation $K = \frac{1+|\mu|}{1-|\mu|}$",
        "teichmuller_distance": r"Teichmüller Distance $d_T$"
    }

    if metric not in metric_labels:
        raise ValueError(f"不支援的測量指標: {metric}。請選擇: {list(metric_labels.keys())}")

    tier_accounts = get_accounts_by_tier(tier)
    selected_folders = []

    if tier_accounts is not None:
        selected_folders = tier_accounts
        scope_description = f"Tier: [{tier.upper()}] (共 {len(selected_folders)} 個帳號)"
    elif target_folders and "all" not in [f.lower() for f in target_folders]:
        selected_folders = target_folders
        scope_description = f"指定的資料夾 (共 {len(selected_folders)} 個)"
    else:
        scope_description = "全部資料夾 (All Folders)"

    qc_files = []
    folder_summary = {}

    if selected_folders:
        for fld in selected_folders:
            pattern = os.path.join(root_dir, fld, "**", "*_qc.json")
            matched = sorted(glob.glob(pattern, recursive=True))
            qc_files.extend(matched)
            folder_summary[fld] = len(matched)
        qc_files = sorted(list(set(qc_files)))
    else:
        pattern = os.path.join(root_dir, "**", "*_qc.json")
        qc_files = sorted(glob.glob(pattern, recursive=True))

    print("=" * 65)
    print("【執行參數與資料集資訊】")
    print(f"  - 帳號分級篩選 (Tier):       {tier.upper()}")
    print(f"  - 統計範圍描述:             {scope_description}")
    print(f"  - 離群值修剪比例:           最高與最低各剔除 {trim_percent}%")

    if selected_folders:
        print("  - 包含的帳號資料夾與檔案數:")
        for idx, fld in enumerate(selected_folders, 1):
            cnt = folder_summary.get(fld, 0)
            status = f"{cnt} 個檔案" if cnt > 0 else "未找到任何 _qc.json"
            print(f"      [{idx}] {fld} ({status})")

    print(f"  - 特徵點編號 (Vertex ID):    #{vertex_id}")
    print(f"  - 測量指標 (Measurement):    {metric} ({metric_labels[metric]})")
    print(f"  - 符合條件的 QC 檔案總數:    {len(qc_files)}")
    print("=" * 65 + "\n")

    if not qc_files:
        print("錯誤：未找到任何可讀取的 *_qc.json 檔案，程式結束。")
        return

    if max_videos is not None:
        qc_files = qc_files[:max_videos]

    raw_video_indices = []
    raw_values_per_video = []
    all_raw_values = []

    for idx, path in enumerate(qc_files, start=1):
        qc_data = load_qc_data(path)
        if qc_data is None:
            continue

        vals = extract_vertex_measurements(qc_data, vertex_id, metric=metric)
        if vals:
            raw_video_indices.append(idx)
            raw_values_per_video.append(vals)
            all_raw_values.extend(vals)

    if not all_raw_values:
        print("未提取到任何有效頂點數據，請確認頂點 ID 是否存在於網格面片中。")
        return

    # ---------------------------------------------------------
    # 計算 5% 與 95% 分位數閾值，剔除 Outliers
    # ---------------------------------------------------------
    lower_pct = trim_percent
    upper_pct = 100.0 - trim_percent

    lower_bound = float(np.percentile(all_raw_values, lower_pct))
    upper_bound = float(np.percentile(all_raw_values, upper_pct))

    # 逐影片過濾掉超出範圍的資料點
    filtered_video_indices = []
    filtered_values_per_video = []
    all_trimmed_values = []

    for idx, vals in zip(raw_video_indices, raw_values_per_video):
        filtered_vals = [v for v in vals if lower_bound <= v <= upper_bound]
        if filtered_vals:
            filtered_video_indices.append(idx)
            filtered_values_per_video.append(filtered_vals)
            all_trimmed_values.extend(filtered_vals)

    trimmed_arr = np.array(all_trimmed_values)
    raw_arr = np.array(all_raw_values)

    # ---------------------------------------------------------
    # Console 輸出修剪前後統計
    # ---------------------------------------------------------
    group_mean = float(np.mean(trimmed_arr))
    group_std = float(np.std(trimmed_arr, ddof=1)) if len(trimmed_arr) > 1 else 0.0
    group_median = float(np.median(trimmed_arr))
    group_min = float(np.min(trimmed_arr))
    group_max = float(np.max(trimmed_arr))

    removed_count = len(raw_arr) - len(trimmed_arr)
    removed_ratio = (removed_count / len(raw_arr)) * 100

    print("=" * 65)
    print(f"【{scope_description} - 離群值剔除與統計結果 (Vertex #{vertex_id})】")
    print(f"  - 原始資料點數:               {len(raw_arr)}")
    print(f"  - 截斷門檻 (5% ~ 95%):        [{lower_bound:.6f}, {upper_bound:.6f}]")
    print(f"  - 剔除離群值數量:             {removed_count} 點 ({removed_ratio:.2f}%)")
    print(f"  - 修剪後資料點數 (Trimmed):   {len(trimmed_arr)}")
    print("  " + "-" * 60)
    print(f"  - 修剪後平均值 (Trimmed Mean): {group_mean:.6f}")
    print(f"  - 修剪後標準差 (Trimmed Std):  {group_std:.6f}")
    print(f"  - 修剪後中位數 (Median):       {group_median:.6f}")
    print(f"  - 修剪後數值區間 (Min ~ Max):  {group_min:.6f} ~ {group_max:.6f}")
    print("=" * 65 + "\n")

    # ---------------------------------------------------------
    # 繪製圖表 (使用剔除 Outliers 後的乾淨資料)
    # ---------------------------------------------------------
    fig_width = max(12, len(filtered_video_indices) * 0.4)
    plt.figure(figsize=(fig_width, 7), dpi=150)

    for x_pos, vals in zip(filtered_video_indices, filtered_values_per_video):
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

    # 繪製 Trimmed Mean 基準線
    plt.axhline(
        group_mean,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label=f"Trimmed Mean (5%-95%): {group_mean:.4f} (Std: {group_std:.4f})"
    )

    tier_tag = f"Tier: {tier.upper()}" if tier.lower() != "all" else "All Accounts"
    title_text = (
        f"[{tier_tag}] (Outliers Removed: 5% Each Tail)  |  Vertex #{vertex_id}\n"
        f"Trimmed Mean: {group_mean:.4f}, Trimmed Std: {group_std:.4f}  |  Measurement: {metric_labels[metric]}"
    )
    plt.title(title_text, fontsize=12, fontweight="bold", pad=15)
    plt.xlabel(f"Video Sequence in [{tier_tag}]", fontsize=11, labelpad=10)
    plt.ylabel(metric_labels[metric], fontsize=11, labelpad=10)
    plt.legend(loc="upper right", frameon=True)

    if len(filtered_video_indices) <= 50:
        plt.xticks(filtered_video_indices, labels=filtered_video_indices)
    else:
        step = max(1, len(filtered_video_indices) // 25)
        plt.xticks(filtered_video_indices[::step], labels=filtered_video_indices[::step])

    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    output_img = f"vertex_{vertex_id}_{metric}_tier_{tier.lower()}_trimmed.png"
    plt.savefig(output_img)
    print(f"圖表已成功生成並儲存至: {os.path.abspath(output_img)}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="統計人臉幾何形變指標（剔除最高/最低各 5% 的極端離群值）")
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "good", "neutral", "bad"],
        help="選擇欲分析的帳號品質級別 (預設: all，可選: good, neutral, bad)"
    )
    parser.add_argument(
        "--folder",
        nargs="+",
        default=None,
        help="手動指定特定資料夾名稱 (若有設定 --tier，則以 --tier 優先)"
    )
    parser.add_argument(
        "--vertex",
        type=int,
        default=1,
        help="欲觀察的 Face Vertex ID (預設為 1，鼻尖點；範圍 0~467)"
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
    parser.add_argument(
        "--trim",
        type=float,
        default=5.0,
        help="兩端各刪去的百分比 (預設 5.0，即剔除最低 5% 與最高 5%)"
    )
    args = parser.parse_args()

    plot_video_deformation(
        root_dir=args.dir,
        tier=args.tier,
        target_folders=args.folder,
        vertex_id=args.vertex,
        metric=args.metric,
        max_videos=args.max_videos,
        trim_percent=args.trim
    )