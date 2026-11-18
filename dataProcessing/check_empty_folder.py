#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:\\NewReels 空資料夾檢查與清理工具 (Empty Subfolder Checker & Cleaner)
-------------------------------------------------------------------
功能：
1. 遍歷 D:\\NewReels 下的所有子資料夾（subfolder）。
2. 精準找出：
   - 完全沒有任何檔案或子目錄的「純空資料夾 (Completely Empty)」。
   - 僅包含 0 bytes 空檔或無有效視音訊檔的「無效資料夾 (No Valid Media)」。
3. 將所有空資料夾的帳號名稱輸出為 empty_accounts.txt（方便重新排查或從名單剔除）。
4. 提供互動式選項（或 --delete 參數），一鍵安全刪除純空資料夾。
"""

import sys
import json
from pathlib import Path

TARGET_ROOT = Path(r"D:\NewReels")
OUTPUT_TXT = Path(__file__).resolve().parent / "empty_accounts.txt"
OUTPUT_JSON = Path(__file__).resolve().parent / "empty_subfolders_report.json"


def check_empty_subfolders(root_dir: Path, auto_delete: bool = False):
    print("=" * 80)
    print(f"🔍 正在檢查空資料夾: {root_dir}")
    print("=" * 80)

    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[-] 錯誤：找不到指定目錄 {root_dir}")
        return

    subfolders = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    if not subfolders:
        print(f"[-] {root_dir} 下沒有任何子資料夾。")
        return

    strictly_empty = []       # 完全沒有任何內容的資料夾
    no_valid_media = []       # 沒有任何大於 0 bytes 的 .mp4/.mp3 檔案的資料夾

    for folder in subfolders:
        all_items = list(folder.iterdir())
        if len(all_items) == 0:
            strictly_empty.append(folder)
            continue

        valid_media = [
            f for f in all_items
            if f.is_file() and f.suffix.lower() in {".mp4", ".mp3"} and f.stat().st_size > 0
        ]
        if not valid_media:
            no_valid_media.append((folder, [f.name for f in all_items]))

    print(f"[*] 掃描總子資料夾數量:           {len(subfolders):>4} 個")
    print(f"  • 📭 完全空資料夾 (0 個檔案):    {len(strictly_empty):>4} 個")
    print(f"  • ⚠️ 無有效視音訊檔資料夾:       {len(no_valid_media):>4} 個")
    print("-" * 80)

    if strictly_empty:
        print("\n📭 [完全空資料夾清單]：")
        for f in strictly_empty:
            print(f"    - {f.name}")

    if no_valid_media:
        print("\n⚠️ [無有效視音訊檔資料夾清單]：")
        for f, items in no_valid_media:
            print(f"    - {f.name} (內含殘留項目: {items})")

    # 匯出純文字帳號名單與 JSON 報告
    all_empty_names = [f.name for f in strictly_empty] + [f.name for f, _ in no_valid_media]
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_empty_names))

    report = {
        "root_dir": str(root_dir),
        "total_subfolders": len(subfolders),
        "strictly_empty_count": len(strictly_empty),
        "no_valid_media_count": len(no_valid_media),
        "strictly_empty_accounts": [f.name for f in strictly_empty],
        "no_valid_media_accounts": [
            {"account": f.name, "residual_files": items} for f, items in no_valid_media
        ],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[*] 空資料夾帳號名單已匯出至: {OUTPUT_TXT.resolve()}")
    print(f"[*] 詳細檢查報告已儲存至:     {OUTPUT_JSON.resolve()}")

    # 詢問或執行刪除完全空資料夾
    if strictly_empty:
        if auto_delete:
            confirm = "y"
        else:
            confirm = input(f"\n❓ 是否要立即刪除這 {len(strictly_empty)} 個「完全空資料夾」？(y/N): ").strip().lower()

        if confirm == "y":
            deleted = 0
            for f in strictly_empty:
                try:
                    f.rmdir()
                    deleted += 1
                except Exception as e:
                    print(f"    [!] 無法刪除 {f.name}: {e}")
            print(f"[+] 已成功清理刪除 {deleted} 個完全空資料夾！")
        else:
            print("[*] 已保留所有資料夾，未執行刪除。")


if __name__ == "__main__":
    delete_flag = "--delete" in sys.argv
    check_empty_subfolders(TARGET_ROOT, auto_delete=delete_flag)