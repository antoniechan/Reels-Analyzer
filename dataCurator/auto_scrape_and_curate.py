#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Instagram Reels Scraper & Dataset Curator Pipeline
------------------------------------------------------------
功能特點：
1. 從 accounts.txt 讀取 Instagram 帳號清單（支援 # 註解與空白行跳過）。
2. 自動呼叫 Apify Actor (apify/instagram-reel-scraper) 抓取原始短影音數據。
3. 支援日期區間、試播排除、分享數抓取、每帳號上限等全域參數[cite: 4]。
4. 自動銜接 Curator 引擎：
   - 時間門檻過濾 (CUTOFF_DATE) 與 ownerUsername 嚴格比對。
   - MediaPipe 真人出鏡時長快篩 (排除料理/純食物空鏡，門檻 >= 40%)。
   - 實體音軌串流 (Audio Stream) 存在性檢查。
   - SenseVoice-Small 本地語音轉錄[cite: 3]。
   - Netlify / Gemini Serverless 口播專業度審核[cite: 3]。
   - 720x1280 畫布等比例補黑邊尺寸對齊[cite: 3]。
5. 統計匯總：在終端機輸出匯總表格，並將每個帳號的入選影片數量記錄至 curated_summary.json[cite: 3]。
"""

import os
import sys
import re
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv
from apify_client import ApifyClient
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 專案路徑設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ROOT_DIR, NEW_REELS_DIR, NEW_REELS_DATA_DIR, VIDEO_PROC_DIR
from dataProcessing.sensevoice_engine import transcribe_chinese_with_sensevoice

# 載入環境變數
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# 模型與服務端點[cite: 3]
FACE_MODEL_PATH = str(VIDEO_PROC_DIR / "face_landmarker.task")
EVALUATOR_ENDPOINT = "https://reelsanalyzercuratorservice.netlify.app/.netlify/functions/evaluate_reels"

# ---------------------------------------------------------------------------
# 全域參數設定
# ---------------------------------------------------------------------------
# 1. 爬蟲參數 (Apify)
MAX_REELS_PER_PROFILE = 50          # 每個帳號抓取上限[cite: 4]
SCRAPE_NEWER_THAN = "2025-09-01"    # 爬取日期下限[cite: 4]
SKIP_TRIAL_REELS = True             # 排除測試短片[cite: 4]
SKIP_PINNED_REELS = False           # 是否排除置頂影片[cite: 4]
INCLUDE_SHARES_COUNT = True         # 包含分享數[cite: 4]

# 2. 篩選與標準化參數 (Curator)
CUTOFF_DATE = "2026-10-01"          # 歷史發布時間上限 (晚於此日期直接排除)[cite: 3]
MIN_FACE_PRESENCE_RATIO = 0.40      # 真人出鏡率門檻 (低於 40% 判定為料理/圖卡空鏡)[cite: 3]
TARGET_WIDTH = 720                  # 影片寬度標準規格[cite: 3]
TARGET_HEIGHT = 1280                # 影片高度標準規格[cite: 3]

# 3. 目錄結構
RAW_DATA_DIR = Path(__file__).resolve().parent / "raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REELS_DIR = NEW_REELS_DIR
OUTPUT_REELS_DATA_DIR = NEW_REELS_DATA_DIR
OUTPUT_REELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REELS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 匯總報表檔案
SUMMARY_REPORT_FILE = Path(__file__).resolve().parent / "curated_summary.json"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


# ---------------------------------------------------------------------------
# 輔助模組：網路請求與遠端評審
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


def evaluate_transcript_via_gemini(transcript_text: str, duration_sec: float, max_retries: int = 3) -> dict:
    """呼叫 Netlify Gemini 審核口播專業度"""
    payload = {"transcript_text": transcript_text, "duration_sec": duration_sec}
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(EVALUATOR_ENDPOINT, json=payload, timeout=25)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5)
            else:
                return {"qualified": False, "reasoning": f"遠端連線異常: {e}"}


# ---------------------------------------------------------------------------
# 輔助模組：視訊、音訊串流與出鏡檢測
# ---------------------------------------------------------------------------
def check_audio_stream_exists(audio_file_path: Path) -> bool:
    """利用 ffprobe / ffmpeg 驗證物理音軌串流是否存在"""
    if not audio_file_path.exists() or audio_file_path.stat().st_size == 0:
        return False

    if shutil.which("ffprobe") is not None:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
            return "audio" in res.stdout.strip().lower()
        except Exception:
            pass

    if shutil.which("ffmpeg") is not None:
        cmd = ["ffmpeg", "-i", str(audio_file_path)]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = res.stderr.lower()
            return "audio:" in output or "stream #0:1" in output or "stream #0:0: audio" in output
        except Exception:
            pass

    return audio_file_path.stat().st_size > 2048


def is_talking_head_video(video_url: str, min_presence_ratio: float = MIN_FACE_PRESENCE_RATIO) -> tuple[bool, str]:
    """抽樣檢測整支影片真人出鏡時長比例，排除做菜空鏡與純食物圖卡短片"""
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        return True, "無法讀取視訊串流，跳過視覺快篩"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0 or not Path(FACE_MODEL_PATH).exists():
        cap.release()
        return True, "缺少模型或畫面無效，跳過快篩"

    duration_sec = total_frames / fps
    num_samples = min(25, max(8, int(duration_sec)))
    step = total_frames // num_samples

    face_detected_count = 0
    checked_count = 0

    try:
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for i in range(num_samples):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                checked_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection = detector.detect(mp_image)
                if detection.face_landmarks:
                    face_detected_count += 1
    except Exception as e:
        cap.release()
        return True, f"視覺快篩異常 ({e})，跳過"

    cap.release()
    if checked_count == 0:
        return True, "抽樣幀為空"

    face_ratio = face_detected_count / checked_count
    if face_ratio < min_presence_ratio:
        return False, f"非真人持續口播 (人臉出鏡率僅 {face_ratio*100:.1f}%，低於 {min_presence_ratio*100:.0f}% 門檻，多為料理/食物空鏡)"

    return True, f"通過口播檢驗 (人臉出鏡率: {face_ratio*100:.1f}%)"


def resize_and_pad_frame(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def ensure_video_aligned(video_path: Path) -> bool:
    """自動等比例縮放並置中補黑邊至 720x1280"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w != TARGET_WIDTH or h != TARGET_HEIGHT:
        print(f"   📐 檢測到原始解析度為 {w}x{h}，正在對齊至 {TARGET_WIDTH}x{TARGET_HEIGHT}...")
        temp_output = video_path.parent / f"{video_path.stem}_temp_aligned.mp4"

        # 優先使用 FFmpeg 封裝以保留最高畫質與原始音軌
        if shutil.which("ffmpeg") is not None:
            vf = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "copy", str(temp_output)]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                video_path.unlink()
                temp_output.rename(video_path)
                return True
            except Exception:
                pass

        # OpenCV 備援
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter(str(temp_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (TARGET_WIDTH, TARGET_HEIGHT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(resize_and_pad_frame(frame, TARGET_WIDTH, TARGET_HEIGHT))
        cap.release()
        out.release()
        video_path.unlink()
        temp_output.rename(video_path)

    return True


# ---------------------------------------------------------------------------
# 核心任務 1：呼叫 Apify 爬取特定帳號
# ---------------------------------------------------------------------------
def scrape_account_reels(username: str, apify_token: str) -> Path | None:
    """執行 Apify 爬蟲並輸出 raw_data/<username>.json"""
    client = ApifyClient(apify_token)
    target_url = f"https://www.instagram.com/{username}/"

    run_input = {
        "username": [target_url],
        "resultsLimit": MAX_REELS_PER_PROFILE,
        "scrapeNewerThan": SCRAPE_NEWER_THAN,
        "skipTrialReels": SKIP_TRIAL_REELS,
        "skipPinnedReels": SKIP_PINNED_REELS,
        "includeSharesCount": INCLUDE_SHARES_COUNT,
        "includeTranscript": False,
        "includeDownloadedVideo": False
    }

    print("\n" + "=" * 80)
    print(f"🚀 [Scraper] 啟動 Apify 爬取帳號: @{username}")
    print(f"   - 抓取上限: {MAX_REELS_PER_PROFILE} | 起始日期: {SCRAPE_NEWER_THAN}")
    print("=" * 80)

    try:
        run = client.actor("apify/instagram-reel-scraper").call(run_input=run_input)
        dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
        if dataset_id is None and isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId")

        print(f"[+] Apify 抓取完成！Dataset ID: {dataset_id}")
        dataset_items = client.dataset(dataset_id).list_items().items
        print(f"[+] 成功獲取 {len(dataset_items)} 條原始短影音數據。")

        output_file = RAW_DATA_DIR / f"{username}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_items, f, ensure_ascii=False, indent=2)

        return output_file
    except Exception as e:
        print(f"[-] Apify 爬取失敗: {e}")
        return None


# ---------------------------------------------------------------------------
# 核心任務 2：清洗、轉錄與入庫過濾
# ---------------------------------------------------------------------------
def curate_account_reels(json_path: Path) -> int:
    """對單一帳號的 raw JSON 執行完整的篩選、轉錄與標準化入庫，回傳合格入選總數"""
    account_id = json_path.stem.replace(".json", "").strip()
    print("\n" + "-" * 80)
    print(f"🔍 [Curator] 開始審查帳號: [{account_id}]")
    print("-" * 80)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f, strict=False)
    except Exception:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        raw_items = json.loads(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content), strict=False)

    if not isinstance(raw_items, list):
        print("[-] 原始 JSON 非陣列格式，略過。")
        return 0

    # 1. 依發布時間降序排序
    raw_items.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)

    # 2. 發布日期截止門檻
    if CUTOFF_DATE:
        threshold = CUTOFF_DATE if "T" in CUTOFF_DATE else f"{CUTOFF_DATE}T23:59:59.999Z"
        raw_items = [item for item in raw_items if str(item.get("timestamp", "")) <= threshold]

    # 3. 嚴格帳號比對：ownerUsername 必須與檔名一致
    raw_items = [item for item in raw_items if str(item.get("ownerUsername", "")).strip() == account_id]

    print(f"[*] 基礎篩選後共有 {len(raw_items)} 條待審查短影音。")

    account_dest_dir = OUTPUT_REELS_DIR / account_id
    account_dest_dir.mkdir(parents=True, exist_ok=True)
    curated_records_path = OUTPUT_REELS_DATA_DIR / f"{account_id}.json"

    curated_records = []
    processed_timestamps = set()
    next_video_idx = 1

    if curated_records_path.exists():
        try:
            with open(curated_records_path, "r", encoding="utf-8") as f:
                curated_records = json.load(f, strict=False)
            processed_timestamps = {r.get("timestamp") for r in curated_records if r.get("timestamp")}
            for r in curated_records:
                m = re.search(r"video_(\d+)", r.get("video_id", ""))
                if m:
                    next_video_idx = max(next_video_idx, int(m.group(1)) + 1)
        except Exception:
            pass

    passed_count = 0

    for item_idx, item in enumerate(raw_items, 1):
        item_ts = item.get("timestamp", "")
        if item_ts and item_ts in processed_timestamps:
            continue

        duration = float(item.get("videoDuration", 0.0) or 0.0)
        audio_url = item.get("audioUrl") or item.get("videoUrl")
        video_url = item.get("videoUrl")

        print(f"\n[{item_idx}/{len(raw_items)}] 審查影片 (時間: {item_ts}, 長度: {duration:.1f}s)")

        # 1. 音軌狀態檢查
        music_info = item.get("musicInfo") or {}
        if music_info.get("should_mute_audio") is True or not audio_url:
            print("   [✗ 排除] 音訊無效或標記為靜音")
            continue

        # 2. 真人出鏡時長檢測 (MediaPipe)
        if video_url:
            has_person, visual_reason = is_talking_head_video(video_url)
            if not has_person:
                print(f"   [✗ 排除] {visual_reason}")
                continue

        # 3. 下載音軌並進行轉錄與審核
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)

        try:
            if not download_file(audio_url, tmp_audio_path):
                print("   [✗ 排除] 音訊小檔下載失敗")
                continue

            if not check_audio_stream_exists(tmp_audio_path):
                print("   [✗ 排除] 實體無有效音軌串流")
                continue

            # SenseVoice 語音辨識
            segments = transcribe_chinese_with_sensevoice(tmp_audio_path)
            text_chunks = []
            for s in segments:
                t = s.get("text", "") if isinstance(s, dict) else s
                if isinstance(t, (tuple, list)):
                    text_chunks.append(" ".join(str(x) for x in t))
                elif t:
                    text_chunks.append(str(t))

            full_text = "".join(text_chunks).strip()
            if not full_text:
                print("   [✗ 排除] 逐字稿為空 (無清晰語音口播)")
                continue

            # Gemini 口播主題評估
            eval_res = evaluate_transcript_via_gemini(full_text, duration)
            is_qualified = eval_res.get("qualified", False) and eval_res.get("confidence_score", 0.0) >= 0.70
            reasoning = eval_res.get("reasoning", "無說明")
            topic = eval_res.get("primary_topic", "未分類")
        finally:
            if tmp_audio_path.exists():
                tmp_audio_path.unlink()

        # 4. 合格入庫
        if is_qualified:
            video_prefix = f"video_{next_video_idx:03d}"
            video_filename = f"{video_prefix}.mp4"
            audio_filename = f"{video_prefix}.mp3"

            print(f"   [✓ 合格入庫] 分配編號: {video_prefix} | 主題: {topic}")
            print(f"     評審理由: {reasoning}")

            final_video_path = account_dest_dir / video_filename
            final_audio_path = account_dest_dir / audio_filename

            v_ok = download_file(video_url, final_video_path) if video_url else False
            a_ok = download_file(item["audioUrl"], final_audio_path) if item.get("audioUrl") else False

            if v_ok and final_video_path.exists():
                ensure_video_aligned(final_video_path)

            record = dict(item)
            record.update({
                "video_id": video_prefix,
                "local_video_file": video_filename,
                "local_audio_file": audio_filename,
                "video_file_exists": v_ok,
                "audio_file_exists": a_ok,
                "transcript": full_text,
                "curation_result": eval_res,
            })
            curated_records.append(record)
            passed_count += 1
            next_video_idx += 1

            with open(curated_records_path, "w", encoding="utf-8") as f:
                json.dump(curated_records, f, ensure_ascii=False, indent=2)
        else:
            print(f"   [✗ 排除] 理由: {reasoning}")

        time.sleep(0.5)

    total_curated_now = len(curated_records)
    print(f"\n[+] 帳號 [{account_id}] 處理完成：本次入選 {passed_count} 支 (累計入庫: {total_curated_now} 支)")
    return total_curated_now


# ---------------------------------------------------------------------------
# 批次處理主入口
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【Instagram Reels 端到端自動採集與清洗入庫引擎】")
    print("=" * 80)

    # 1. 取得 API Token
    apify_token = os.getenv("APIFY_API_TOKEN")
    if not apify_token:
        apify_token = input("👉 請輸入你的 Apify API Token: ").strip()
        if not apify_token:
            print("[-] 缺少 Apify Token，程式終止。")
            return

    # 2. 取得 accounts.txt 路徑
    default_txt = Path(__file__).resolve().parent / "accounts.txt"
    if len(sys.argv) > 1:
        txt_path = Path(sys.argv[1].strip('"').strip("'"))
    elif default_txt.exists():
        txt_path = default_txt
    else:
        user_input = input("👉 請輸入 accounts.txt 檔案路徑: ").strip('"').strip("'")
        txt_path = Path(user_input)

    if not txt_path.exists():
        print(f"[-] 找不到帳號清單檔案: {txt_path}")
        return

    # 讀取帳號列表（過濾空行與 # 註解）
    with open(txt_path, "r", encoding="utf-8") as f:
        accounts = [
            line.strip().lstrip("@") for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not accounts:
        print("[-] accounts.txt 中沒有有效的帳號。")
        return

    print(f"[*] 成功讀取 {len(accounts)} 個目標帳號: {', '.join(accounts)}")

    # 載入現有匯總統計
    summary_data = {}
    if SUMMARY_REPORT_FILE.exists():
        try:
            with open(SUMMARY_REPORT_FILE, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    # 3. 迴圈執行 Scrape -> Curate
    for idx, account in enumerate(accounts, 1):
        print("\n" * 2 + "#" * 80)
        print(f"🚀 進度 [{idx}/{len(accounts)}]: 正在處理帳號 @{account}")
        print("#" * 80)

        # 步驟 A: 爬取
        raw_json_file = scrape_account_reels(account, apify_token)
        if not raw_json_file or not raw_json_file.exists():
            print(f"[-] 帳號 @{account} 爬取失敗，略過清洗步驟。")
            summary_data[account] = summary_data.get(account, 0)
            continue

        # 步驟 B: 清洗與入庫
        qualified_count = curate_account_reels(raw_json_file)
        summary_data[account] = qualified_count

        # 即時更新統計報表檔案
        with open(SUMMARY_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # 4. 最終報表輸出
    print("\n" * 2 + "=" * 80)
    print("🎉 所有帳號批次處理完成！最終合格入庫統計表：")
    print("=" * 80)
    print(f"{'帳號 (Account ID)':<35} | {'入選合格影片數量':<15}")
    print("-" * 55)
    for acc in accounts:
        count = summary_data.get(acc, 0)
        print(f"@{acc:<34} | {count:>4} 支")
    print("-" * 55)
    total_videos = sum(summary_data.get(acc, 0) for acc in accounts)
    print(f"{'總計 (Total Qualified)':<35} | {total_videos:>4} 支")
    print(f"\n[*] 匯總 JSON 已保存至: {SUMMARY_REPORT_FILE.resolve()}")
    print(f"[*] 視訊與音訊檔案存放在: {OUTPUT_REELS_DIR.resolve()}")


if __name__ == "__main__":
    main()