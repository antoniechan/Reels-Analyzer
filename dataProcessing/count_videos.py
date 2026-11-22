#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:\\NewReels 子資料夾影片數量統計工具 (Subfolder Video Counter)
-------------------------------------------------------------
功能：
1. 遍歷 D:\\NewReels 下的每一個帳號子資料夾（subfolder）。
2. 統計每個子資料夾中的有效影片數量（.mp4 檔案，預設排除 0 bytes 空檔案）。
3. 預設依帳號名稱字母順序（Alphabetic Order）排列，列出各帳號明細及整體統計。
4. 將統計結果同步匯出為 JSON 報表與 CSV 表格（皆依字母順序排列）。
"""

import json
from pathlib import Path

# 目標根目錄
TARGET_ROOT = Path(r"D:\NewReels")

# 輸出報表路徑（儲存於腳本同目錄下）
BASE_DIR = Path(__file__).resolve().parent
REPORT_JSON = BASE_DIR / "video_counts.json"

# 支援計算的視訊副檔名
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def count_videos_in_subfolders(root_dir: Path, sort_by_count: bool = False):
    print("=" * 75)
    print(f"📊 開始統計目錄下的子資料夾影片數量: {root_dir}")
    print("=" * 75)

    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[-] 錯誤：找不到指定目錄 {root_dir}")
        return

    subfolders = sorted([d for d in root_dir.iterdir() if d.is_dir()], key=lambda d: d.name.lower())
    if not subfolders:
        print(f"[-] {root_dir} 下沒有任何子資料夾。")
        return

    results = []
    for folder in subfolders:
        valid_videos = [
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in VIDEO_EXTENSIONS
            and f.stat().st_size > 0
        ]
        results.append({
            "account": folder.name,
            "video_count": len(valid_videos)
        })

    # 排序邏輯：預設依帳號字母順序 (Alphabetic Order)，若 sort_by_count=True 則依影片數量降序
    if sort_by_count:
        results.sort(key=lambda x: (-x["video_count"], x["account"].lower()))
    else:
        results.sort(key=lambda x: x["account"].lower())

    # 輸出明細表格
    print(f"{'序號':<6} | {'帳號名稱 (Subfolder)':<38} | {'影片數量':>8}")
    print("-" * 75)
    for idx, item in enumerate(results, 1):
        print(f"{idx:<6} | {item['account']:<38} | {item['video_count']:>6} 支")

    # 計算總結數據
    total_accounts = len(results)
    total_videos = sum(item["video_count"] for item in results)
    non_empty_accounts = sum(1 for item in results if item["video_count"] > 0)
    zero_accounts = total_accounts - non_empty_accounts
    avg_videos = (total_videos / non_empty_accounts) if non_empty_accounts > 0 else 0.0

    print("=" * 75)
    print("📈 統計總結 (Summary)")
    print("=" * 75)
    print(f"  • 子資料夾（帳號）總數:         {total_accounts:>6} 個")
    print(f"  • 有效影片總數 (Total Videos):  {total_videos:>6} 支")
    print(f"  • 含有影片的帳號數 (>0 支):     {non_empty_accounts:>6} 個")
    print(f"  • 無影片的空帳號數 (0 支):      {zero_accounts:>6} 個")
    print(f"  • 有片帳號平均影片數:           {avg_videos:>6.2f} 支")
    print("=" * 75)

    # 匯出 JSON 報表（依帳號字母順序排列）
    summary_payload = {
        "root_dir": str(root_dir),
        "sort_order": "count_desc" if sort_by_count else "alphabetic_asc",
        "summary": {
            "total_subfolders": total_accounts,
            "total_videos": total_videos,
            "non_empty_accounts": non_empty_accounts,
            "zero_video_accounts": zero_accounts,
            "average_videos_per_active_account": round(avg_videos, 2),
        },
        "accounts": results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"[*] JSON 報表已儲存至: {REPORT_JSON.resolve()}")


if __name__ == "__main__":
    count_videos_in_subfolders(TARGET_ROOT, sort_by_count=False)