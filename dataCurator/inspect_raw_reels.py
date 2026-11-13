#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw Reels JSON Diagnostic & Filter Audit Tool
---------------------------------------------
功能：
讀取 raw_data 資料夾下的指定帳號 JSON 檔案，統計並列出：
1. 原始總影片數量
2. 不符合日期限制的影片數量（早於 START_DATE 或晚於 CUTOFF_DATE）
3. 不符合帳號原創限制的影片數量（ownerUsername 與檔名 account_id 不一致，或屬於轉載/協作貼文）
4. 同時違反兩者的重疊數量，以及最終通過基礎過濾的剩餘數量
5. 列出具體被排除的貼文詳情（時間戳、實際 ownerUsername、短網址/ID）供排查。
"""

import sys
import re
import json
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# 過濾參數設定（與 auto_scrape_and_curate.py 保持一致）
# ---------------------------------------------------------------------------
START_DATE = None   # 發布日期下限 (早於此日期視為不符；設為 None 則不檢查下限)
CUTOFF_DATE = "2026-10-01"   # 發布日期上限 (晚於此日期視為不符；設為 None 則不檢查上限)

RAW_DATA_DIR = Path(__file__).resolve().parent / "raw_data"


def load_raw_json(json_path: Path) -> list:
    """安全讀取 JSON 檔案（含控制字元容錯處理）"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f, strict=False)
    except Exception:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content_cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
        return json.loads(content_cleaned, strict=False)


def inspect_raw_file(target_input: str):
    # 解析輸入路徑或帳號名稱
    p = Path(target_input.strip('"').strip("'"))
    if p.exists() and p.is_file():
        json_path = p
    else:
        filename = target_input if target_input.endswith(".json") else f"{target_input}.json"
        json_path = RAW_DATA_DIR / filename

    if not json_path.exists():
        print(f"[-] 找不到指定檔案: {json_path}")
        return

    account_id = json_path.stem.strip()
    raw_items = load_raw_json(json_path)

    if not isinstance(raw_items, list):
        print(f"[-] 檔案格式錯誤：{json_path.name} 的根節點不是 JSON 陣列 (List)。")
        return

    total_count = len(raw_items)
    start_ts = f"{START_DATE}T00:00:00.000Z" if START_DATE and "T" not in START_DATE else START_DATE
    cutoff_ts = f"{CUTOFF_DATE}T23:59:59.999Z" if CUTOFF_DATE and "T" not in CUTOFF_DATE else CUTOFF_DATE

    too_early_items = []
    too_late_items = []
    non_original_items = []
    both_failed_count = 0
    passed_items = []

    other_owners_counter = Counter()

    for idx, item in enumerate(raw_items, 1):
        ts = str(item.get("timestamp", "")).strip()
        owner = str(item.get("ownerUsername", "")).strip()
        short_code = item.get("shortCode") or item.get("id") or f"item_{idx}"
        url = item.get("url") or f"https://www.instagram.com/reel/{short_code}/"

        is_date_invalid = False
        date_reason = []

        if start_ts and ts and ts < start_ts:
            is_date_invalid = True
            date_reason.append(f"早於 {START_DATE}")
            too_early_items.append((ts, owner, url))

        if cutoff_ts and ts and ts > cutoff_ts:
            is_date_invalid = True
            date_reason.append(f"晚於 {CUTOFF_DATE}")
            too_late_items.append((ts, owner, url))

        is_owner_invalid = (owner != account_id)
        if is_owner_invalid:
            non_original_items.append((ts, owner, url))
            other_owners_counter[owner or "<空值>"] += 1

        if is_date_invalid and is_owner_invalid:
            both_failed_count += 1

        if not is_date_invalid and not is_owner_invalid:
            passed_items.append(item)

    date_invalid_total = len(too_early_items) + len(too_late_items)

    # -----------------------------------------------------------------------
    # 輸出統計報告
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"📊 原始資料診斷報告：@{account_id} ({json_path.name})")
    print("=" * 75)
    print(f"  • 原始爬取總數 (Total Raw Items):          {total_count:>4} 支")
    print("-" * 75)
    print(f"  • 不符合日期限制總計:                      {date_invalid_total:>4} 支")
    if START_DATE:
        print(f"      ├─ 早於起始日期 (< {START_DATE}):      {len(too_early_items):>4} 支")
    if CUTOFF_DATE:
        print(f"      └─ 晚於截止日期 (> {CUTOFF_DATE}):      {len(too_late_items):>4} 支")
    print(f"  • 不符合帳號原創限制 (owner != @{account_id}): {len(non_original_items):>4} 支")
    print(f"  • 同時違反日期與原創限制 (重疊部分):       {both_failed_count:>4} 支")
    print("-" * 75)
    print(f"  ✅ 通過此兩項基礎條件（進入後續視聽審查）: {len(passed_items):>4} 支")
    print("=" * 75)

    # 列出非原創帳號分佈
    if other_owners_counter:
        print("\n🔍 [非原創 / 協作貼文來源帳號統計]：")
        for other_acc, cnt in other_owners_counter.most_common():
            print(f"    - @{other_acc}: {cnt} 支")

    # 列出詳細被排除清單（最多顯示前 15 筆，避免洗版）
    if too_early_items or too_late_items:
        print("\n📅 [不符合日期限制明細 (前 15 筆)]：")
        for ts, owner, url in (too_early_items + too_late_items)[:15]:
            print(f"    - [{ts[:10] if ts else '無日期'}] (owner: @{owner}) -> {url}")

    if non_original_items:
        print("\n👤 [不符合原創帳號限制明細 (前 15 筆)]：")
        for ts, owner, url in non_original_items[:15]:
            print(f"    - [{ts[:10] if ts else '無日期'}] 實際發佈者: @{owner} -> {url}")
    print()


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # 列出 raw_data 下現有檔案方便參考
        if RAW_DATA_DIR.exists():
            files = sorted(RAW_DATA_DIR.glob("*.json"))
            if files:
                print(f"[*] {RAW_DATA_DIR.name}/ 內共有 {len(files)} 個檔案（範例: {', '.join(f.stem for f in files[:5])}...）")
        target = input("👉 請輸入要檢查的帳號 ID 或 JSON 檔名 (例如 eat.well.nutrition): ").strip()

    if target:
        inspect_raw_file(target)


if __name__ == "__main__":
    main()
