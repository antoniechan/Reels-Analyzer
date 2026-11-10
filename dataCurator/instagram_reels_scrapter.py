#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instagram Reels Apify Scraper Integration
-----------------------------------------
將 Apify 的 `apify/instagram-reel-scraper` 搬移至本地 Python 調用。
依據指定 Instagram 使用者名稱抓取 Reels，並輸出為標準的 <username>.json。
"""

import os
import json
from pathlib import Path
from apify_client import ApifyClient
from dotenv import load_dotenv

# 載入專案的 .env 以讀取 APIFY_API_TOKEN
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------------------------
# 全域參數設定
# ---------------------------------------------------------------------------
TARGET_USERNAME = "dietitianlizachan"
MAX_REELS_PER_PROFILE = 50
NEWER_THAN_DATE = "2025-09-01"

SKIP_TRIAL_REELS = True
SKIP_PINNED_REELS = False
INCLUDE_SHARES_COUNT = True

OUTPUT_DIR = Path(__file__).resolve().parent / "raw_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scrape_reels_for_user(
    username: str,
    max_reels: int = MAX_REELS_PER_PROFILE,
    newer_than: str = NEWER_THAN_DATE,
    output_dir: Path = OUTPUT_DIR
):
    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        api_token = input("👉 請輸入你的 Apify API Token (或在 .env 設定 APIFY_API_TOKEN): ").strip()
        if not api_token:
            print("[-] 錯誤：缺少 Apify API Token，終止執行。")
            return

    client = ApifyClient(api_token)
    target_url = f"https://www.instagram.com/{username}/"

    run_input = {
        "username": [target_url],
        "resultsLimit": max_reels,
        "scrapeNewerThan": newer_than,
        "skipTrialReels": SKIP_TRIAL_REELS,
        "skipPinnedReels": SKIP_PINNED_REELS,
        "includeSharesCount": INCLUDE_SHARES_COUNT,
        "includeTranscript": False,
        "includeDownloadedVideo": False
    }

    print("=" * 70)
    print("🚀 啟動 Apify Instagram Reel Scraper")
    print(f"  - 目標帳號:         {username}")
    print(f"  - 抓取上限:         {max_reels} 支")
    print(f"  - 發布日期大於:     {newer_than}")
    print(f"  - 略過試播影片:     {SKIP_TRIAL_REELS}")
    print(f"  - 包含分享數統計:   {INCLUDE_SHARES_COUNT}")
    print("=" * 70)

    try:
        print("[*] 正在向 Apify 提交任務並等待執行完成（通常耗時 30-90 秒）...")
        run = client.actor("apify/instagram-reel-scraper").call(run_input=run_input)

        # 修正：相容 Run 物件與 dict 取值
        dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
        if dataset_id is None and isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId")

        print(f"[+] Apify 任務執行完畢！Dataset ID: {dataset_id}")

        print("[*] 正在取得爬取資料...")
        dataset_items = client.dataset(dataset_id).list_items().items

        print(f"[+] 成功獲取 {len(dataset_items)} 筆 Reels 數據。")

        output_file = output_dir / f"{username}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_items, f, ensure_ascii=False, indent=2)

        print(f"🎉 檔案已成功儲存至: {output_file.resolve()}")
        return output_file

    except Exception as e:
        print(f"[-] Apify 爬取失敗: {e}")
        return None


if __name__ == "__main__":
    scrape_reels_for_user(TARGET_USERNAME)