import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

DEFAULT_REELS_DIR = r"D:\reels"
OUTPUT_REPORT_DIR = r".\videoProcessing"

# 常見影片副檔名
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def check_single_qc_json(qc_path: str, zero_eps: float = 1e-8):
    """
    檢查單一 _qc.json 的內容完整性與數值健康度。
    回傳: (is_valid: bool, issues: list, metrics_summary: dict)
    """
    issues = []

    # 1. 檔案大小檢查 (避免 0 KB 空檔案)
    if os.path.getsize(qc_path) == 0:
        return False, ["檔案大小為 0 位元組 (空檔案)"], {}

    # 2. JSON 解析檢查
    try:
        with open(qc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"JSON 格式損毀無法解析: {e}"], {}

    if not isinstance(data, dict):
        return False, ["根節點非 JSON 物件 (Dictionary)"], {}

    # 3. 欄位存在性檢查
    transitions = data.get("consecutive_qc_metrics", [])
    if not transitions:
        return False, ["缺少 consecutive_qc_metrics 欄位或為空陣列"], {}

    total_transitions = len(transitions)
    valid_transitions_count = 0
    all_mu_values = []
    all_dil_values = []
    has_facet_data = False

    # 4. 影格級與面片級數值檢查
    for t_idx, trans in enumerate(transitions):
        if not isinstance(trans, dict):
            issues.append(f"第 {t_idx} 幀過渡不是物件格式")
            continue

        if trans.get("valid_transition", False):
            valid_transitions_count += 1

        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            continue

        has_facet_data = True
        for fm in facet_metrics:
            mu = fm.get("abs_mu")
            dil = fm.get("dilation")
            verts = fm.get("vertices", [])

            # 頂點檢查
            if len(verts) != 3:
                issues.append(f"第 {t_idx} 幀存在非三角形面片 (vertices != 3)")

            # 數值型態與異常值檢查
            if mu is not None:
                if np.isnan(mu) or np.isinf(mu):
                    issues.append(f"第 {t_idx} 幀發現 NaN 或 Inf abs_mu")
                else:
                    all_mu_values.append(float(mu))

            if dil is not None:
                if np.isnan(dil) or np.isinf(dil):
                    issues.append(f"第 {t_idx} 幀發現 NaN 或 Inf dilation")
                else:
                    all_dil_values.append(float(dil))

    # 5. 整體統計邏輯判定
    if valid_transitions_count == 0:
        issues.append("整部影片完全沒有任何有效過渡幀 (valid_transition 全為 False)")

    if not has_facet_data:
        issues.append("整部影片沒有任何面片形變記錄 (facet_metrics 全部為空)")

    if not all_mu_values:
        issues.append("未擷取到任何有效的 abs_mu 數值")
    else:
        mu_arr = np.array(all_mu_values)
        non_zero_ratio = np.mean(mu_arr > zero_eps)
        if non_zero_ratio == 0:
            issues.append("所有 abs_mu 數值皆為 0 (臉部完全未產生任何動態形變)")
        if np.any(mu_arr < 0):
            issues.append("存在小於 0 的 abs_mu (違反幾何定義)")

    if all_dil_values:
        dil_arr = np.array(all_dil_values)
        if np.any(dil_arr < 0.99):  # Dilation 理論上 K >= 1.0 (容許微小浮點誤差)
            issues.append("存在小於 1.0 的 dilation (違反準共形膨脹定義)")

    # 摘要資訊
    summary = {
        "total_transitions": total_transitions,
        "valid_transitions": valid_transitions_count,
        "valid_transition_ratio": round(valid_transitions_count / max(1, total_transitions), 4),
        "mean_mu": round(float(np.mean(all_mu_values)), 6) if all_mu_values else 0.0,
        "max_mu": round(float(np.max(all_mu_values)), 6) if all_mu_values else 0.0,
        "mean_dilation": round(float(np.mean(all_dil_values)), 4) if all_dil_values else 0.0,
    }

    is_valid = len(issues) == 0
    return is_valid, list(set(issues)), summary


def scan_and_validate_directory(reels_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("【影片附屬 _qc.json 完整性與數據健康度檢查系統】")
    print(f"  - 掃描目標目錄: {os.path.abspath(reels_dir)}")
    print("=" * 80 + "\n")

    if not os.path.exists(reels_dir):
        print(f"[錯誤] 目錄不存在: {reels_dir}")
        return

    # 1. 遞迴尋找所有影片檔案
    print("正在檢索所有影片檔案...")
    all_video_files = []
    for root, _, files in os.walk(reels_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                all_video_files.append(os.path.join(root, f))

    all_video_files = sorted(all_video_files)
    total_videos = len(all_video_files)
    print(f"共找到 {total_videos} 部影片檔案。\n開始比對附屬 _qc.json 並校驗數值...\n")

    results = []
    missing_qc_count = 0
    corrupt_qc_count = 0
    healthy_qc_count = 0

    for idx, vid_path in enumerate(all_video_files, start=1):
        folder_dir, vid_name = os.path.split(vid_path)
        vid_stem = os.path.splitext(vid_name)[0]
        account_id = os.path.basename(folder_dir)

        # 預期附屬 QC 檔案名稱
        qc_candidate = os.path.join(folder_dir, f"{vid_stem}_qc.json")

        status_rec = {
            "account_id": account_id,
            "video_stem": vid_stem,
            "video_path": vid_path,
            "qc_path": qc_candidate,
            "qc_exists": False,
            "is_healthy": False,
            "issue_details": "",
            "total_transitions": 0,
            "valid_transitions": 0,
            "mean_mu": 0.0,
            "max_mu": 0.0,
        }

        if not os.path.exists(qc_candidate):
            missing_qc_count += 1
            status_rec["issue_details"] = "缺少 _qc.json 檔案"
            results.append(status_rec)
            continue

        status_rec["qc_exists"] = True

        # 檢查 QC 內容
        is_healthy, issues, summary = check_single_qc_json(qc_candidate)
        status_rec.update(summary)

        if is_healthy:
            healthy_qc_count += 1
            status_rec["is_healthy"] = True
            status_rec["issue_details"] = "正常"
        else:
            corrupt_qc_count += 1
            status_rec["is_healthy"] = False
            status_rec["issue_details"] = " | ".join(issues)

        results.append(status_rec)

        if idx % 100 == 0 or idx == total_videos:
            print(f"  進度: [{idx}/{total_videos}] (正常: {healthy_qc_count}, 損毀/異常: {corrupt_qc_count}, 缺失: {missing_qc_count})")

    df_results = pd.DataFrame(results)

    # 輸出彙總報表
    print("\n" + "=" * 80)
    print("【檢查結果總結】")
    print(f"  - 影片總數:                  {total_videos:>5} 支")
    print(f"  - 具備完整且健康 _qc.json:     {healthy_qc_count:>5} 支 ({healthy_qc_count / max(1, total_videos) * 100:.1f}%)")
    print(f"  - 存在 _qc.json 但數據異常:   {corrupt_qc_count:>5} 支 ({corrupt_qc_count / max(1, total_videos) * 100:.1f}%)")
    print(f"  - 完全缺少 _qc.json:          {missing_qc_count:>5} 支 ({missing_qc_count / max(1, total_videos) * 100:.1f}%)")
    print("=" * 80)

    # 匯出 CSV 詳細清單
    all_report_csv = os.path.join(output_dir, "qc_json_validation_full_report.csv")
    df_results.to_csv(all_report_csv, index=False, encoding="utf-8-sig")
    print(f"\n完整檢查報告已儲存: {os.path.abspath(all_report_csv)}")

    # 匯出僅含異常與缺失的檔案清單 (供後續重新計算使用)
    abnormal_df = df_results[~df_results["is_healthy"]]
    if not abnormal_df.empty:
        abnormal_csv = os.path.join(output_dir, "qc_json_missing_or_corrupt_list.csv")
        abnormal_df.to_csv(abnormal_csv, index=False, encoding="utf-8-sig")
        print(f"異常與缺失清單已匯出 (共 {len(abnormal_df)} 筆): {os.path.abspath(abnormal_csv)}")
    else:
        print("所有影片的 _qc.json 均健康無缺失！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批次檢查影片目錄中附屬 _qc.json 的存在性與數據健康度")
    parser.add_argument("--dir", type=str, default=DEFAULT_REELS_DIR, help=f"Reels 影片根目錄 (預設: {DEFAULT_REELS_DIR})")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_REPORT_DIR, help=f"報告輸出目錄 (預設: {OUTPUT_REPORT_DIR})")
    args = parser.parse_args()

    scan_and_validate_directory(args.dir, args.out_dir)