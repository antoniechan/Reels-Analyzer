#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:\\reelsData 與 D:\\NewReels 雙向一致性對賬腳本 (Cross-Check Reconciliation)
--------------------------------------------------------------------------
目的：
排查為何 D:\reelsData 有 3003 筆紀錄，而 D:\\NewReels 只有 2975 支影片（精確定位差異的 28 筆）。

檢查項目：
1. 逐帳號比對 D:\\reelsData\\<account>.json 中的記錄筆數 vs D:\\NewReels\\<account>\\ 內的實際有效 .mp4 數量。
2. 逐筆檢查 JSON 中每條紀錄的 local_video_file / video_id 對應的實體檔案是否存在且大小 > 0 bytes。
3. 檢查 JSON 內部是否存在重複的 timestamp / shortCode / video_id 紀錄。
4. 反向檢查 D:\\NewReels 中是否存在「有實體影片但 JSON 無記錄」的孤兒檔案或孤兒資料夾。
5. 匯出詳細對賬報告 reconciliation_report.json。
"""

import re
import json
from pathlib import Path
from collections import Counter

REELS_DATA_DIR = Path(r"D:\reelsData")
NEW_REELS_DIR = Path(r"D:\NewReels")
REPORT_PATH = Path(__file__).resolve().parent / "reconciliation_report.json"


def load_json_safe(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f, strict=False)
    except Exception:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content_cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
        return json.loads(content_cleaned, strict=False)


def main():
    print("=" * 85)
    print("🔍 開始進行 D:\\reelsData (JSON) 與 D:\\NewReels (實體影片) 雙向對賬")
    print("=" * 85)

    if not REELS_DATA_DIR.exists() or not NEW_REELS_DIR.exists():
        print("[-] 錯誤：找不到 D:\\reelsData 或 D:\\NewReels 目錄，請確認路徑。")
        return

    json_files = {p.stem: p for p in sorted(REELS_DATA_DIR.glob("*.json"), key=lambda x: x.stem.lower())}
    video_folders = {d.name: d for d in sorted(NEW_REELS_DIR.iterdir(), key=lambda x: x.name.lower()) if d.is_dir()}

    all_accounts = sorted(set(json_files.keys()) | set(video_folders.keys()), key=str.lower)

    total_json_records = 0
    total_disk_mp4 = 0

    discrepancy_accounts = []
    missing_video_files = []
    empty_video_files = []
    duplicate_json_entries = []
    orphan_disk_files = []
    json_without_folder = []
    folder_without_json = []

    for acc in all_accounts:
        jf = json_files.get(acc)
        vf = video_folders.get(acc)

        records = []
        if jf:
            data = load_json_safe(jf)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]

        disk_mp4s = {}
        if vf:
            for f in vf.iterdir():
                if f.is_file() and f.suffix.lower() == ".mp4":
                    disk_mp4s[f.name] = f.stat().st_size

        total_json_records += len(records)
        valid_disk_count = sum(1 for sz in disk_mp4s.values() if sz > 0)
        total_disk_mp4 += valid_disk_count

        if jf and not vf and len(records) > 0:
            json_without_folder.append({"account": acc, "json_count": len(records)})
        if vf and not jf and len(disk_mp4s) > 0:
            folder_without_json.append({"account": acc, "disk_mp4_count": len(disk_mp4s)})

        # 檢查帳號內部數量不一致
        if len(records) != valid_disk_count:
            discrepancy_accounts.append({
                "account": acc,
                "json_count": len(records),
                "valid_mp4_count": valid_disk_count,
                "diff (json - mp4)": len(records) - valid_disk_count,
            })

        # 逐筆比對 JSON 紀錄與實體檔案
        expected_filenames = set()
        seen_video_ids = Counter()
        seen_timestamps = Counter()

        for idx, item in enumerate(records, 1):
            if not isinstance(item, dict):
                continue

            vid_id = str(item.get("video_id", "")).strip()
            local_mp4 = str(item.get("local_video_file", "")).strip()
            if not local_mp4 and vid_id:
                local_mp4 = f"{vid_id}.mp4"

            ts = str(item.get("timestamp", "")).strip()
            short_code = str(item.get("shortCode") or item.get("id") or f"row_{idx}")

            if vid_id:
                seen_video_ids[vid_id] += 1
            if ts:
                seen_timestamps[ts] += 1

            if local_mp4:
                expected_filenames.add(local_mp4)
                if local_mp4 not in disk_mp4s:
                    missing_video_files.append({
                        "account": acc,
                        "video_id": vid_id,
                        "expected_file": local_mp4,
                        "shortCode": short_code,
                        "timestamp": ts,
                        "video_file_exists_flag": item.get("video_file_exists"),
                        "videoUrl": item.get("videoUrl", ""),
                    })
                elif disk_mp4s[local_mp4] == 0:
                    empty_video_files.append({
                        "account": acc,
                        "expected_file": local_mp4,
                        "shortCode": short_code,
                    })

        # 檢查 JSON 內重複項
        dup_vids = [k for k, v in seen_video_ids.items() if v > 1]
        dup_ts = [k for k, v in seen_timestamps.items() if v > 1]
        if dup_vids or dup_ts:
            duplicate_json_entries.append({
                "account": acc,
                "duplicate_video_ids": dup_vids,
                "duplicate_timestamps": dup_ts,
            })

        # 檢查實體資料夾內多出（未被 JSON 引用）的 mp4
        extra_on_disk = sorted(set(disk_mp4s.keys()) - expected_filenames)
        if extra_on_disk:
            orphan_disk_files.append({
                "account": acc,
                "unreferenced_mp4s": extra_on_disk,
            })

    # -----------------------------------------------------------------------
    # 終端機報告輸出
    # -----------------------------------------------------------------------
    print(f"  • JSON 紀錄總筆數 (D:\\reelsData):     {total_json_records:>6} 筆")
    print(f"  • 實體有效影片總數 (D:\\NewReels):    {total_disk_mp4:>6} 支")
    print(f"  • 總差額 (JSON - 實體影片):          {total_json_records - total_disk_mp4:>+6} 筆")
    print("=" * 85)

    if discrepancy_accounts:
        print(f"\n⚠️ [1] 數量不一致的帳號明細 (共 {len(discrepancy_accounts)} 個帳號)：")
        print(f"{'帳號 (Account)':<35} | {'JSON筆數':>8} | {'實體MP4':>8} | {'差額':>6}")
        print("-" * 70)
        for d in discrepancy_accounts:
            print(f"{d['account']:<35} | {d['json_count']:>8} | {d['valid_mp4_count']:>8} | {d['diff (json - mp4)']:>+6}")

    if missing_video_files:
        print(f"\n❌ [2] JSON 有記錄但實體 MP4 缺失的明細 (共 {len(missing_video_files)} 筆)：")
        for m in missing_video_files:
            print(f"    - [{m['account']}] {m['expected_file']} (shortCode: {m['shortCode']}, video_file_exists={m['video_file_exists_flag']})")

    if empty_video_files:
        print(f"\n🔸 [3] 實體檔案為 0 bytes 空檔明細 (共 {len(empty_video_files)} 筆)：")
        for e in empty_video_files:
            print(f"    - [{e['account']}] {e['expected_file']} (shortCode: {e['shortCode']})")

    if duplicate_json_entries:
        print(f"\n🔁 [4] JSON 內部存在重複 video_id 或 timestamp 的帳號 (共 {len(duplicate_json_entries)} 個)：")
        for dup in duplicate_json_entries:
            print(f"    - [{dup['account']}] 重複 video_id: {dup['duplicate_video_ids']}, 重複 timestamp: {dup['duplicate_timestamps']}")

    if orphan_disk_files:
        print(f"\n👻 [5] 實體有 MP4 但未被 JSON 記錄引用的檔案 (共 {len(orphan_disk_files)} 個帳號)：")
        for orp in orphan_disk_files:
            print(f"    - [{orp['account']}]: {orp['unreferenced_mp4s']}")

    if json_without_folder:
        print(f"\n📁 [6] 有 JSON 記錄但完全沒有對應資料夾的帳號: {json_without_folder}")

    if folder_without_json:
        print(f"\n📁 [7] 有實體資料夾但沒有對應 JSON 檔的帳號: {folder_without_json}")

    # 儲存完整 JSON 報告
    report = {
        "summary": {
            "total_json_records": total_json_records,
            "total_valid_disk_mp4": total_disk_mp4,
            "net_difference": total_json_records - total_disk_mp4,
            "discrepancy_accounts_count": len(discrepancy_accounts),
            "missing_video_files_count": len(missing_video_files),
            "empty_video_files_count": len(empty_video_files),
            "duplicate_json_accounts_count": len(duplicate_json_entries),
            "orphan_disk_accounts_count": len(orphan_disk_files),
        },
        "discrepancy_accounts": discrepancy_accounts,
        "missing_video_files": missing_video_files,
        "empty_video_files": empty_video_files,
        "duplicate_json_entries": duplicate_json_entries,
        "orphan_disk_files": orphan_disk_files,
        "json_without_folder": json_without_folder,
        "folder_without_json": folder_without_json,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 85)
    print(f"[*] 完整雙向對賬報告已儲存至: {REPORT_PATH.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
