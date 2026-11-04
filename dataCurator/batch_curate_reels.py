#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Reels Dataset Curator & Ingestion Engine (SenseVoice-Small + Netlify Serverless Gemini)
---------------------------------------------------------------------------------------------
功能：
1. 遍歷指定目錄下的所有帳號 .json 檔案。
2. 按照每筆 data 的 "timestamp" 欄位降序排序（最新排在最前面）。
3. 本地 SenseVoice-Small 轉錄逐字稿（整段乾淨繁中文字，無需 timestamp）。
4. 呼叫部署於 Netlify 美國節點的 Serverless Function 進行口播主題與健康觀念審核。
5. 連續命名機制：僅對合格收錄樣本依序分配 video_001.mp4, video_001.mp3...，被淘汰者不佔號。
6. 輸出標準欄位：transcript, local_video_file, local_audio_file。
"""

import re
import sys
import json
import time
import tempfile
from pathlib import Path
import requests
from dotenv import load_dotenv
from config import PYTHON_PROJECT_DIR, REELS_DATA_DIR
from dataProcessing.sensevoice_engine import transcribe_cantonese_with_sensevoice

# 載入專案環境設定與 .env
ENV_PATH = PYTHON_PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Netlify Serverless 審核端點
EVALUATOR_ENDPOINT = "https://reelsanalyzercuratorservice.netlify.app/.netlify/functions/evaluate_reels"

OUTPUT_REELS_DIR = REELS_DATA_DIR
OUTPUT_REELS_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


# ---------------------------------------------------------------------------
# 1. 網路請求工具函式
# ---------------------------------------------------------------------------
def download_file(url: str, dest_path: Path, timeout: int = 40) -> bool:
    """串流下載檔案至指定路徑"""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"      [!] 下載失敗: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def evaluate_transcript_via_netlify(transcript_text: str, duration_sec: float) -> dict:
    """呼叫 Netlify 美國伺服器進行 Gemini 口播主題審查"""
    payload = {
        "transcript_text": transcript_text,
        "duration_sec": duration_sec
    }
    try:
        response = requests.post(EVALUATOR_ENDPOINT, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"      [!] Netlify 審核端點異常: {e}")
        return {"qualified": False, "reasoning": f"遠端連線異常: {e}"}


# ---------------------------------------------------------------------------
# 2. 單一帳號 JSON 處理管線
# ---------------------------------------------------------------------------
def process_account_json(json_path: Path):
    account_id = json_path.stem.replace(".json", "").strip()
    print("\n" + "=" * 80)
    print(f"📂 正在處理帳號: [{account_id}] (來源檔案: {json_path.name})")
    print("=" * 80)

    # 讀取 JSON 並容錯處理未轉義控制字元
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f, strict=False)
    except Exception:
        try:
            with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            content_cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
            raw_items = json.loads(content_cleaned, strict=False)
        except Exception as e:
            print(f"[-] 無法讀取 JSON 檔案: {e}")
            return

    if not isinstance(raw_items, list):
        print("[-] JSON 格式不符合預期（必須為包含多條 Reels 的 List 陣列）。")
        return

    # 1. 按照 "timestamp" 降序排序，最新排在最前面
    raw_items.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
    print(f"[*] 該帳號共有 {len(raw_items)} 條待審查 Reels，已按 timestamp 由最新到最舊排序。")

    account_dest_dir = OUTPUT_REELS_DIR / account_id
    account_dest_dir.mkdir(parents=True, exist_ok=True)

    # 讀取已存在的審查紀錄，支援斷點續跑
    curated_records_path = account_dest_dir / f"{account_id}.json"
    curated_records = []
    processed_timestamps = set()
    next_video_idx = 1

    if curated_records_path.exists():
        try:
            with open(curated_records_path, "r", encoding="utf-8") as f:
                curated_records = json.load(f, strict=False)
            processed_timestamps = {r.get("timestamp") for r in curated_records if r.get("timestamp")}

            for r in curated_records:
                vid = r.get("video_id", "")
                m = re.search(r"video_(\d+)", vid)
                if m:
                    idx_val = int(m.group(1))
                    if idx_val >= next_video_idx:
                        next_video_idx = idx_val + 1
            print(f"[*] 載入現有紀錄: 已有 {len(processed_timestamps)} 條合格記錄，下一支收錄編號為: video_{next_video_idx:03d}")
        except Exception:
            pass

    passed_count = 0
    rejected_count = 0

    for item_idx, item in enumerate(raw_items, 1):
        item_ts = item.get("timestamp", "")

        if item_ts and item_ts in processed_timestamps:
            continue

        caption = item.get("caption", "") or ""
        duration = float(item.get("videoDuration", 0.0) or 0.0)
        audio_url = item.get("audioUrl") or item.get("videoUrl")
        video_url = item.get("videoUrl")

        print(f"\n[{item_idx}/{len(raw_items)}] 正在審查原始候選影片 (時間: {item_ts}, 長度: {duration:.1f}s)")

        first_line = caption.strip().split("\n")[0][:40] if caption else "無文案"
        print(f"   💬 首行文案: {first_line}...")

        if not audio_url:
            print("   [-] 無法取得音訊 URL，略過。")
            continue

        # 下載暫存音軌進行本地 SenseVoice 辨識
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)

        try:
            print("   ⬇️ 正在下載音軌小檔 (~1MB)...")
            if not download_file(audio_url, tmp_audio_path):
                continue

            print("   🎙️ 本地 SenseVoice-Small 轉錄逐字稿中...")
            segments = transcribe_cantonese_with_sensevoice(tmp_audio_path)

            full_text = "".join([s.get("text", "") for s in segments]).strip()
            if not full_text:
                full_text = caption.strip()

            # 發送純文字至 Netlify 進行美國節點 Gemini 審核
            print("   ☁️ 傳送逐字稿至 Netlify Serverless 進行口播主題評估...")
            eval_res = evaluate_transcript_via_netlify(full_text, duration)
            is_qualified = eval_res.get("qualified", False) and eval_res.get("confidence_score", 0.0) >= 0.70
            reasoning = eval_res.get("reasoning", "無說明")
            topic = eval_res.get("primary_topic", "未分類")

        finally:
            if tmp_audio_path.exists():
                tmp_audio_path.unlink()

        # -------------------------------------------------------------
        # 判定分流與正式收錄命名 (排除不佔號)
        # -------------------------------------------------------------
        if is_qualified:
            video_prefix = f"video_{next_video_idx:03d}"
            video_filename = f"{video_prefix}.mp4"
            audio_filename = f"{video_prefix}.mp3"

            print(f"   [✓ 合格通過] 分配命名: {video_prefix} | 主題: {topic}")
            print(f"     理由: {reasoning}")

            final_video_path = account_dest_dir / video_filename
            final_audio_path = account_dest_dir / audio_filename

            video_downloaded = False
            audio_downloaded = False

            if video_url:
                print(f"   📥 下載正式影片本體至: {video_filename}...")
                video_downloaded = download_file(video_url, final_video_path)

            if item.get("audioUrl"):
                print(f"   📥 下載正式音訊檔案至: {audio_filename}...")
                audio_downloaded = download_file(item["audioUrl"], final_audio_path)

            record = {
                "account_id": account_id,
                "video_id": video_prefix,
                "local_video_file": video_filename,
                "local_audio_file": audio_filename,
                "timestamp": item.get("timestamp"),
                "duration_sec": duration,
                "play_count": item.get("videoPlayCount"),
                "likes_count": item.get("likesCount"),
                "comments_count": item.get("commentsCount"),
                "caption": caption,
                "video_file_exists": video_downloaded,
                "audio_file_exists": audio_downloaded,
                "transcript": full_text,
                "curation_result": eval_res
            }
            curated_records.append(record)
            passed_count += 1
            next_video_idx += 1

            with open(curated_records_path, "w", encoding="utf-8") as f:
                json.dump(curated_records, f, ensure_ascii=False, indent=2)

        else:
            print(f"   [✗ 排除淘汰] 理由: {reasoning} (不佔用編號)")
            rejected_count += 1

        time.sleep(0.5)

    print("\n" + "-" * 80)
    print(f"[+] 帳號 [{account_id}] 批次審查完畢！")
    print(f"    - 本次通過收錄: {passed_count} 支")
    print(f"    - 本次淘汰排除: {rejected_count} 支")
    print(f"    - 清洗後清單已存至: {curated_records_path}")


# ---------------------------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【Reels 批次自動篩選與入庫引擎 (SenseVoice + Netlify Serverless)】")
    print("=" * 80)

    if len(sys.argv) > 1:
        target_folder_str = sys.argv[1].strip('"').strip("'")
    else:
        target_folder_str = input("👉 請輸入裝有多個帳號 .json 的資料夾路徑: ").strip('"').strip("'")

    target_folder = Path(target_folder_str)
    if not target_folder.exists() or not target_folder.is_dir():
        print(f"[-] 找不到指定目錄: {target_folder}")
        return

    json_files = list(target_folder.glob("*.json"))
    if not json_files:
        print(f"[-] 在 {target_folder} 下找不到任何 .json 檔案。")
        return

    print(f"[*] 在目標目錄發現 {len(json_files)} 個帳號資料檔案：")
    for f in json_files:
        print(f"    - {f.name}")

    for jf in json_files:
        process_account_json(jf)

    print("\n" + "=" * 80)
    print(f"🎉 全部帳號已批次審查完畢！合格素材已存放於: {OUTPUT_REELS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()