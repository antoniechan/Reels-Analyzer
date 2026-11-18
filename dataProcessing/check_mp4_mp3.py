#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:\\NewReels 檔案完整性與連續編號稽核腳本
-----------------------------------------
檢查項目：
1. 遍歷 D:\\NewReels 下的每一個帳號子資料夾（subfolder）。
2. 檢查檔案編號是否嚴格由 video_001, video_002, ... 連續遞增（無跳號、無漏號）。
3. 檢查每一個編號是否同時齊備 video_xxx.mp4 與 video_xxx.mp3（且檔案大小 > 0 bytes）。
4. 偵測資料夾內是否有命名格式異常的多餘檔案或臨時檔案（如 *_temp_aligned.mp4）。
5. 輸出控制台摘要報告與詳細異常清單，並儲存 JSON 稽核報告。
"""

import re
import json
from pathlib import Path
from collections import defaultdict

# 目標根目錄
TARGET_ROOT = Path(r"D:\NewReels")
REPORT_PATH = Path(__file__).resolve().parent / "newreels_audit_report.json"

# 匹配標準檔名：video_001.mp4 或 video_001.mp3
FILENAME_PATTERN = re.compile(r"^video_(\d{3})\.(mp4|mp3)$", re.IGNORECASE)


def audit_subfolder(folder: Path) -> dict:
    """檢查單一帳號子資料夾內的檔案序列與成對完整性"""
    files = [f for f in folder.iterdir() if f.is_file()]

    # 記錄每個編號擁有的副檔名與大小：{1: {"mp4": size, "mp3": size}}
    index_map = defaultdict(dict)
    unexpected_files = []

    for f in files:
        match = FILENAME_PATTERN.match(f.name)
        if not match:
            unexpected_files.append(f.name)
            continue

        idx = int(match.group(1))
        ext = match.group(2).lower()
        size = f.stat().st_size
        index_map[idx][ext] = size

    issues = []

    if not index_map:
        if unexpected_files:
            issues.append(f"資料夾內無標準 video_xxx 檔案，但發現 {len(unexpected_files)} 個異常檔案: {unexpected_files}")
        else:
            issues.append("空資料夾（無任何檔案）")
        return {
            "account": folder.name,
            "status": "EMPTY_OR_INVALID",
            "total_indices": 0,
            "expected_range": None,
            "issues": issues,
            "unexpected_files": unexpected_files,
        }

    present_indices = sorted(index_map.keys())
    max_idx = present_indices[-1]
    expected_indices = list(range(1, max_idx + 1))

    # 1. 檢查編號是否由 001 開始連續排列（無跳號）
    missing_indices = sorted(set(expected_indices) - set(present_indices))
    if missing_indices:
        missing_str = ", ".join(f"video_{i:03d}" for i in missing_indices)
        issues.append(f"編號不連續（缺少序號）: {missing_str}")

    # 2. 檢查每個存在的編號是否齊備 .mp4 與 .mp3，且檔案不為 0 bytes
    missing_mp4 = []
    missing_mp3 = []
    empty_files = []

    for idx in present_indices:
        exts = index_map[idx]
        prefix = f"video_{idx:03d}"

        if "mp4" not in exts:
            missing_mp4.append(f"{prefix}.mp4")
        elif exts["mp4"] == 0:
            empty_files.append(f"{prefix}.mp4 (0 bytes)")

        if "mp3" not in exts:
            missing_mp3.append(f"{prefix}.mp3")
        elif exts["mp3"] == 0:
            empty_files.append(f"{prefix}.mp3 (0 bytes)")

    if missing_mp4:
        issues.append(f"缺少 MP4 視訊檔: {', '.join(missing_mp4)}")
    if missing_mp3:
        issues.append(f"缺少 MP3 音訊檔: {', '.join(missing_mp3)}")
    if empty_files:
        issues.append(f"發現空檔案 (0 bytes): {', '.join(empty_files)}")
    if unexpected_files:
        issues.append(f"存在非標準命名檔案: {', '.join(unexpected_files)}")

    status = "OK" if not issues else "ERROR"

    return {
        "account": folder.name,
        "status": status,
        "total_indices": len(present_indices),
        "expected_range": f"video_001 ~ video_{max_idx:03d}",
        "issues": issues,
        "unexpected_files": unexpected_files,
    }


def main():
    print("=" * 80)
    print(f"🔍 開始檢查資料夾完整性與連續編號: {TARGET_ROOT}")
    print("=" * 80)

    if not TARGET_ROOT.exists() or not TARGET_ROOT.is_dir():
        print(f"[-] 錯誤：找不到目錄 {TARGET_ROOT}")
        return

    subfolders = sorted([d for d in TARGET_ROOT.iterdir() if d.is_dir()])
    if not subfolders:
        print(f"[-] {TARGET_ROOT} 下沒有任何子資料夾。")
        return

    ok_folders = []
    error_folders = []
    empty_folders = []
    total_valid_pairs = 0

    for folder in subfolders:
        res = audit_subfolder(folder)
        if res["status"] == "OK":
            ok_folders.append(res)
            total_valid_pairs += res["total_indices"]
        elif res["status"] == "EMPTY_OR_INVALID":
            empty_folders.append(res)
        else:
            error_folders.append(res)

    # 輸出異常明細
    if error_folders:
        print(f"\n❌ 發現 {len(error_folders)} 個異常子資料夾明細：")
        print("-" * 80)
        for item in error_folders:
            print(f"📂 [{item['account']}] (範圍: {item['expected_range']})")
            for issue in item["issues"]:
                print(f"    - {issue}")
        print("-" * 80)

    if empty_folders:
        print(f"\n⚠️ 發現 {len(empty_folders)} 個空資料夾或無標準命名檔案的資料夾：")
        for item in empty_folders:
            print(f"    - [{item['account']}]: {', '.join(item['issues'])}")

    # 輸出總結統計
    print("\n" + "=" * 80)
    print("📊 稽核總結報告")
    print("=" * 80)
    print(f"  • 掃描子資料夾總數:           {len(subfolders):>4} 個")
    print(f"  • ✅ 完全正常（連號且成對）:   {len(ok_folders):>4} 個 (共 {total_valid_pairs} 組完整影片)")
    print(f"  • ❌ 存在缺號/缺檔/異常:       {len(error_folders):>4} 個")
    print(f"  • ⚠️ 空資料夾/無有效編號:      {len(empty_folders):>4} 個")
    print("=" * 80)

    # 儲存完整 JSON 報告
    report_data = {
        "root_dir": str(TARGET_ROOT),
        "summary": {
            "total_subfolders": len(subfolders),
            "ok_count": len(ok_folders),
            "error_count": len(error_folders),
            "empty_count": len(empty_folders),
            "total_complete_pairs_in_ok_folders": total_valid_pairs,
        },
        "errors": error_folders,
        "empty_or_invalid": empty_folders,
        "ok": ok_folders,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"[*] 詳細 JSON 稽核報告已儲存至: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()