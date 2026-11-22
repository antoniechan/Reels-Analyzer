#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instagram Account FollowersCount Batch Scraper (via Apify API)
-------------------------------------------------------------
功能：
1. 掃描指定資料夾下的所有 .json 檔案（例如 D:\reelsData），提取不帶副檔名的檔案名稱作為 Instagram account name。
2. 支援斷點續傳：自動讀取已存在的輸出報表（accounts_followers.json），跳過已成功抓到 followersCount 的帳號。
3. 調用 Apify 官方 `apify/instagram-scraper`（或 `apify/instagram-profile-scraper`）批次抓取帳號主頁資訊，提取 `followersCount`。
4. 支援分批請求（BATCH_SIZE）以大幅節省 Apify Actor 啟動時間與額度，並同時匯出 JSON 與 CSV（可直接用 Excel 開啟）。
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from apify_client import ApifyClient

# ---------------------------------------------------------------------------
# 預設路徑與環境設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# 預設目標資料夾（裝有 <account_id>.json 的目錄，可透過命令列參數覆蓋）
DEFAULT_TARGET_DIR = Path(r"D:\reelsData")

# 輸出報表檔案
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = BASE_DIR / "accounts_followers.json"
OUTPUT_CSV = BASE_DIR / "accounts_followers.csv"

# 每次送給 Apify 批次查詢的帳號數量（建議 10~20 個一批，減少冷啟動耗時）
BATCH_SIZE = 15

# 選用的 Apify Actor（apify/instagram-scraper 支援 directUrls + resultsType="details"）
APIFY_ACTOR_ID = "apify/instagram-scraper"


def extract_usernames_from_folder(folder_path: Path) -> list[str]:
    """讀取資料夾下所有 .json 檔案，提取純檔名作為 Instagram username"""
    json_files = sorted(folder_path.glob("*.json"), key=lambda p: p.stem.lower())
    usernames = []
    seen = set()
    for jf in json_files:
        acc = jf.stem.strip()
        if acc and acc not in seen:
            seen.add(acc)
            usernames.append(acc)
    return usernames


def fetch_followers_batch(client: ApifyClient, usernames: list[str]) -> dict[str, dict]:
    """
    呼叫 Apify Instagram Scraper 取得一批帳號的詳細資料 (resultsType='details')。
    回傳字典：{username_lower: {"username": ..., "followersCount": ..., "followsCount": ..., "postsCount": ..., "fullName": ...}}
    """
    direct_urls = [f"https://www.instagram.com/{u}/" for u in usernames]

    run_input = {
        "directUrls": direct_urls,
        "resultsType": "details",
        "resultsLimit": 1,
        "searchLimit": 1,
        "addParentData": False,
    }

    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
    dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
    if dataset_id is None and isinstance(run, dict):
        dataset_id = run.get("defaultDatasetId")

    items = client.dataset(dataset_id).list_items().items
    batch_results = {}

    for item in items:
        uname = str(item.get("username") or item.get("ownerUsername") or "").strip()
        if not uname:
            # 嘗試從輸入 URL 反推
            inp_url = str(item.get("inputUrl") or item.get("url") or "").rstrip("/")
            if "instagram.com/" in inp_url:
                uname = inp_url.split("instagram.com/")[-1].split("/")[0].strip()

        if uname:
            batch_results[uname.lower()] = {
                "username": uname,
                "fullName": item.get("fullName", ""),
                "followersCount": item.get("followersCount"),
                "followsCount": item.get("followsCount"),
                "postsCount": item.get("postsCount"),
                "verified": item.get("verified", False),
                "biography": item.get("biography", ""),
            }

    return batch_results


def save_reports(records_map: dict[str, dict], ordered_accounts: list[str]):
    """同步儲存 JSON 與 CSV 報表（依帳號字母順序排列）"""
    ordered_list = []
    for acc in ordered_accounts:
        info = records_map.get(acc.lower(), {
            "username": acc,
            "fullName": "",
            "followersCount": None,
            "followsCount": None,
            "postsCount": None,
            "verified": None,
            "biography": "",
            "status": "NOT_FOUND_OR_FAILED",
        })
        if "status" not in info:
            info["status"] = "OK" if info.get("followersCount") is not None else "MISSING_COUNT"
        ordered_list.append(info)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ordered_list, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Account_ID", "Full_Name", "Followers_Count", "Follows_Count", "Posts_Count", "Verified", "Status"])
        for r in ordered_list:
            writer.writerow([
                r.get("username", ""),
                r.get("fullName", ""),
                r.get("followersCount") if r.get("followersCount") is not None else "",
                r.get("followsCount") if r.get("followsCount") is not None else "",
                r.get("postsCount") if r.get("postsCount") is not None else "",
                r.get("verified") if r.get("verified") is not None else "",
                r.get("status", ""),
            ])


def main():
    print("=" * 80)
    print("🚀 Instagram 帳號 FollowersCount 批次抓取工具 (Apify Instagram Scraper)")
    print("=" * 80)

    # 1. 解析目標資料夾路徑
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1].strip('"').strip("'"))
    elif DEFAULT_TARGET_DIR.exists():
        target_dir = DEFAULT_TARGET_DIR
    else:
        user_dir = input("👉 請輸入包含 .json 檔案的資料夾路徑: ").strip('"').strip("'")
        target_dir = Path(user_dir)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[-] 錯誤：找不到指定資料夾 {target_dir}")
        return

    # 2. 讀取 Apify API Token
    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        api_token = input("👉 請輸入 Apify API Token (或於 .env 設定 APIFY_API_TOKEN): ").strip()
        if not api_token:
            print("[-] 缺少 Apify API Token，終止執行。")
            return

    # 3. 提取所有 .json 檔名作為帳號名稱
    all_accounts = extract_usernames_from_folder(target_dir)
    if not all_accounts:
        print(f"[-] 在 {target_dir} 中找不到任何 .json 檔案。")
        return

    print(f"[*] 從 {target_dir} 共提取出 {len(all_accounts)} 個帳號名稱。")

    # 4. 載入既有結果（斷點續傳）
    existing_map = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                old_list = json.load(f)
            for item in old_list:
                u = str(item.get("username", "")).strip().lower()
                if u and item.get("followersCount") is not None:
                    existing_map[u] = item
            print(f"[*] 載入既有快取紀錄：已有 {len(existing_map)} 個帳號具備有效 followersCount。")
        except Exception:
            pass

    pending_accounts = [acc for acc in all_accounts if acc.lower() not in existing_map]
    print(f"[*] 本次待查詢帳號數量: {len(pending_accounts)} 個")

    if not pending_accounts:
        print("[+] 所有帳號均已完成抓取！正在重新整理輸出報表...")
        save_reports(existing_map, all_accounts)
        print(f"[*] JSON 報表: {OUTPUT_JSON.resolve()}")
        print(f"[*] CSV 表格:  {OUTPUT_CSV.resolve()}")
        return

    client = ApifyClient(api_token)
    total_batches = (len(pending_accounts) + BATCH_SIZE - 1) // BATCH_SIZE

    # 5. 分批呼叫 Apify
    for b_idx in range(total_batches):
        batch = pending_accounts[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        print(f"\n[{b_idx + 1}/{total_batches}] 正在查詢批次 ({len(batch)} 個帳號): {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")

        try:
            batch_res = fetch_followers_batch(client, batch)
            found_cnt = 0
            for acc in batch:
                info = batch_res.get(acc.lower())
                if info and info.get("followersCount") is not None:
                    existing_map[acc.lower()] = info
                    found_cnt += 1
                    print(f"    ✓ @{acc:<28} -> followersCount: {info['followersCount']:,}")
                else:
                    print(f"    ✗ @{acc:<28} -> 未回傳 followersCount (帳號可能改名、設為私人或不存在)")

            print(f"[+] 本批次成功取得 {found_cnt}/{len(batch)} 個帳號資料。")

        except Exception as e:
            print(f"[-] 本批次 Apify 呼叫發生錯誤: {e}")

        # 每完成一批即即時存檔，避免中途斷線遺失進度
        save_reports(existing_map, all_accounts)
        if b_idx < total_batches - 1:
            time.sleep(1.0)

    # 6. 輸出最終統計
    success_total = sum(1 for acc in all_accounts if existing_map.get(acc.lower(), {}).get("followersCount") is not None)
    print("\n" + "=" * 80)
    print("🎉 全部批次抓取作業完成！")
    print(f"  • 目標帳號總數:         {len(all_accounts):>5} 個")
    print(f"  • 成功取得粉絲數帳號:   {success_total:>5} 個")
    print(f"  • 缺失或無法取得帳號:   {len(all_accounts) - success_total:>5} 個")
    print(f"  • JSON 報表輸出位置:    {OUTPUT_JSON.resolve()}")
    print(f"  • CSV 表格輸出位置:     {OUTPUT_CSV.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
