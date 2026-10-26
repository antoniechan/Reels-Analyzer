#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Reels Dataset Curator & Ingestion Engine
---------------------------------------------
功能：
1. 遍歷指定目錄下的所有帳號 .json 檔案（每個檔案包含一組 Reels metadata 清單）。
2. 三道漏斗過濾：
   - 階段 1：文案關鍵主題文字初篩 (純本地與 Gemini 快速文字判定)
   - 階段 2：音軌下載 (~1MB) + Gemini 深度健康知識口播審查與逐字稿提取
   - 階段 3：只有合格樣本才下載完整影片 (.mp4) 進入訓練資料庫
3. 自動依帳號分門別類存檔，具備斷點續跑（已存在的跳過）功能。
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import PYTHON_PROJECT_DIR, REELS_DATA_DIR

# 載入專案環境設定與 .env
ENV_PATH = PYTHON_PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------------------------
# 預設儲存路徑設定
# ---------------------------------------------------------------------------
# 存放下載合格影片的根目錄 (可改為你的 D:\reels 或本機路徑)
OUTPUT_REELS_DIR = REELS_DATA_DIR
OUTPUT_REELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 網路請求工具函式 (帶防盜鏈 Header 與串流下載)
# ---------------------------------------------------------------------------
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

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


# ---------------------------------------------------------------------------
# 2. Gemini 內容審核模組
# ---------------------------------------------------------------------------
def evaluate_audio_with_gemini(client: genai.Client, audio_path: Path) -> dict:
    """將音訊傳入 Gemini 進行健康觀念口播評級與逐字稿提取"""
    uploaded_file = None
    try:
        uploaded_file = client.files.upload(file=str(audio_path))

        system_instruction = """
你是一位嚴謹的短影音資料集審核專家。你的任務是評估一段 Reels 短影音是否符合「健康知識與營養觀念科普短影音」的收錄標準。

【嚴格收錄標準 (qualified: true)】：
1. 形式：主要由單一講者對鏡頭進行口播（Monologue）傳遞知識、解答迷思。
2. 主題：健康、營養素作用、人體生理機能、亞健康改善、預防醫學。
3. 商業容許度：【可以推薦產品或保健成分】，但前提必須是「以解釋該成分/營養素對身體有何重要性與科學原理」為切入點。

【淘汰標準 (qualified: false)】：
1. 純促銷/硬廣：只強調折扣特價、開箱，無人體機理說明。
2. 純情境短劇、多人綜藝訪談、搞笑娛樂、純背景音樂跳舞。
3. 主題與人體健康完全無關（如純美妝穿搭、旅遊、理財、生活雜談）。

請同時輸出精確的逐字稿（含時間戳記），並以純 JSON 輸出：
{
  "qualified": true / false,
  "confidence_score": 0.0 ~ 1.0,
  "primary_topic": "簡短主題（如：心血管硬化預防 / 蛋白質吸收迷思）",
  "reasoning": "簡短繁體中文判定依據",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "話語片段..."},
    {"start": 2.5, "end": 5.0, "text": "話語片段..."}
  ]
}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded_file, "請聽取這段語音，評估是否符合健康知識口播標準，並提供帶時間戳記的逐字稿。"],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text.strip())

    except Exception as e:
        print(f"      [!] Gemini 審核異常: {e}")
        return {"qualified": False, "reasoning": f"API 異常: {e}", "segments": []}
    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 3. 單一帳號 JSON 處理管線
# ---------------------------------------------------------------------------
def process_account_json(json_path: Path, client: genai.Client):
    account_id = json_path.stem.replace(".json", "").strip()
    print("\n" + "=" * 80)
    print(f"📂 正在處理帳號: [{account_id}] (來源檔案: {json_path.name})")
    print("=" * 80)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
    except Exception as e:
        print(f"[-] 無法讀取 JSON 檔案: {e}")
        return

    if not isinstance(raw_items, list):
        print("[-] JSON 格式不符合預期（必須為包含多條 Reels 的 List 陣列）。")
        return

    print(f"[*] 該帳號共有 {len(raw_items)} 條待審查 Reels。")

    # 該帳號的目標資料夾
    account_dest_dir = OUTPUT_REELS_DIR / account_id
    account_dest_dir.mkdir(parents=True, exist_ok=True)

    # 讀取已存在的審查紀錄，支援斷點續跑
    curated_records_path = account_dest_dir / f"{account_id}.json"
    curated_records = []
    processed_video_stems = set()
    if curated_records_path.exists():
        try:
            with open(curated_records_path, "r", encoding="utf-8") as f:
                curated_records = json.load(f)
            processed_video_stems = {r["video_id"] for r in curated_records}
            print(f"[*] 載入現有紀錄: 已有 {len(processed_video_stems)} 條處理記錄，將自動略過已審樣本。")
        except Exception:
            pass

    passed_count = 0
    rejected_count = 0

    for idx, item in enumerate(raw_items, 1):
        # 產生唯一識別碼 (以 timestamp 或網址特徵為 stem)
        ts_str = item.get("timestamp", "").replace(":", "").replace("-", "").replace(".", "")
        video_id = f"{account_id}_{ts_str}" if ts_str else f"{account_id}_{idx:04d}"

        if video_id in processed_video_stems:
            continue

        caption = item.get("caption", "") or ""
        duration = float(item.get("videoDuration", 0.0) or 0.0)
        audio_url = item.get("audioUrl") or item.get("videoUrl")
        video_url = item.get("videoUrl")

        print(f"\n[{idx}/{len(raw_items)}] 正在審查影片: {video_id} (長度: {duration:.1f}s)")

        # -------------------------------------------------------------
        # 階段 1：文字初篩 (Caption 快速檢視)
        # -------------------------------------------------------------
        # 若文案過短或完全沒有關鍵字，但包含音訊，我們讓它進入音訊檢驗
        # 若明顯是純英文/純生活無關文案可在此快篩
        first_line = caption.strip().split("\n")[0][:40] if caption else "無文案"
        print(f"   💬 首行文案: {first_line}...")

        if not audio_url:
            print("   [-] 無法取得音訊 URL，略過。")
            continue

        # -------------------------------------------------------------
        # 階段 2：下載音軌小檔 (~1MB) 並進行 Gemini 深度審核
        # -------------------------------------------------------------
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)

        try:
            print("   ⬇️ 正在下載音軌小檔 (~1MB)...")
            if not download_file(audio_url, tmp_audio_path):
                continue

            print("   🧠 傳送至 Gemini 評估主題與提取逐字稿...")
            eval_res = evaluate_audio_with_gemini(client, tmp_audio_path)
            is_qualified = eval_res.get("qualified", False) and eval_res.get("confidence_score", 0.0) >= 0.70
            reasoning = eval_res.get("reasoning", "無說明")
            topic = eval_res.get("primary_topic", "未分類")
            segments = eval_res.get("segments", [])

        finally:
            if tmp_audio_path.exists():
                tmp_audio_path.unlink()

        # -------------------------------------------------------------
        # 階段 3：判定分流與合格影片本體下載
        # -------------------------------------------------------------
        if is_qualified:
            print(f"   [✓ 合格通過] 主題: {topic}")
            print(f"     理由: {reasoning}")

            # 下載完整影片 (.mp4)
            final_video_path = account_dest_dir / f"{video_id}.mp4"
            video_downloaded = False
            if video_url:
                print(f"   📥 下載正式影片本體至: {final_video_path.name}...")
                video_downloaded = download_file(video_url, final_video_path)

            # 存下對應的 Gemini 逐字稿 (供後續 Transformer 提取時直接秒讀)
            transcript_cache_path = account_dest_dir / f"{video_id}_gemini_transcript.json"
            with open(transcript_cache_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            record = {
                "account_id": account_id,
                "video_id": video_id,
                "timestamp": item.get("timestamp"),
                "duration_sec": duration,
                "play_count": item.get("videoPlayCount"),
                "likes_count": item.get("likesCount"),
                "comments_count": item.get("commentsCount"),
                "caption": caption,
                "video_file_exists": video_downloaded,
                "curation_result": eval_res
            }
            curated_records.append(record)
            passed_count += 1

            # 即時寫入 JSON 檔，避免中途中斷資料遺失
            with open(curated_records_path, "w", encoding="utf-8") as f:
                json.dump(curated_records, f, ensure_ascii=False, indent=2)

        else:
            print(f"   [✗ 排除淘汰] 理由: {reasoning}")
            rejected_count += 1

        # 避免觸發 API 頻率限制
        time.sleep(1.0)

    print("\n" + "-" * 80)
    print(f"[+] 帳號 [{account_id}] 批次審查完畢！")
    print(f"    - 本次通過收錄: {passed_count} 支")
    print(f"    - 本次淘汰排除: {rejected_count} 支")
    print(f"    - 清洗後清單已存至: {curated_records_path}")


# ---------------------------------------------------------------------------
# 4. 主程序：掃描指定目錄下的所有 .json 檔
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【Reels 批次自動篩選與入庫引擎 (Batch Curator)】")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[-] 錯誤：找不到 GEMINI_API_KEY，請確認 .env 檔案設定。")
        return

    client = genai.Client(api_key=api_key)

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
        process_account_json(jf, client)

    print("\n" + "=" * 80)
    print(f"🎉 全部帳號已批次審查完畢！合格素材已存放於: {OUTPUT_REELS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()