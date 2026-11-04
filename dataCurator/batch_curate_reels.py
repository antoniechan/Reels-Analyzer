#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Reels Dataset Curator & Ingestion Engine
(SenseVoice + Netlify Gemini Evaluator + Dimension Alignment + Face Detection)
-----------------------------------------------------------------------------
功能：
1. 遍歷指定目錄下的所有帳號 .json 檔案。
2. 按照每筆 data 的 "timestamp" 欄位降序排序（最新排在最前面）。
3. 支援 CUTOFF_DATE 門檻過濾，自動剔除晚於指定日期的樣本。
4. 支援 TARGET_OWNER_USERNAME 門檻過濾，剔除非指定帳號的樣本。
5. 【真人出鏡視覺快篩】：在轉錄前透過 OpenCV 與 MediaPipe 抽樣畫面，畫面全無人臉直接排除。
6. 本地 SenseVoice-Small 轉錄逐字稿（整段乾淨繁中文字）。
7. 呼叫 Netlify 美國節點 Serverless Function 評估口播內容與主題。
8. 連續命名機制：合格樣本分配 video_001.mp4, video_001.mp3...，淘汰者不佔號。
9. 【自動 Dimension 對齊】：下載影片後自動檢查尺寸，非 720x1280 則透過 ffmpeg/OpenCV
   等比例縮放並置中補黑邊（保持人臉長寬比不變形）。
"""

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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import ROOT_DIR, NEW_REELS_DIR, NEW_REELS_DATA_DIR, VIDEO_PROC_DIR
from dataProcessing.sensevoice_engine import transcribe_chinese_with_sensevoice

# 將專案根目錄加入 Python 搜尋路徑
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 載入專案環境設定與 .env
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# 定義人臉模型路徑
FACE_MODEL_PATH = str(VIDEO_PROC_DIR / "face_landmarker.task")

# Netlify Serverless 審核端點
EVALUATOR_ENDPOINT = "https://reelsanalyzercuratorservice.netlify.app/.netlify/functions/evaluate_reels"

# 預設儲存根目錄
OUTPUT_REELS_DIR = NEW_REELS_DIR
OUTPUT_REELS_DATA_DIR = NEW_REELS_DATA_DIR
OUTPUT_REELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REELS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 參數與門檻設定
# ---------------------------------------------------------------------------
# 日期篩選門檻 (晚於此日期的資料將在排序後直接刪除，設為 None 則不限制)
CUTOFF_DATE = "2026-10-01"

# 帳號擁有者篩選 (只保留 ownerUsername 等於此名稱的資料，若設為 None 則不限制)
TARGET_OWNER_USERNAME = "annhealthy_diet"

# 影片標準化規格
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280

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


def evaluate_transcript_via_netlify(transcript_text: str, duration_sec: float, max_retries: int = 3) -> dict:
    """呼叫 Netlify 美國伺服器進行 Gemini 口播主題審查（具備重試機制）"""
    payload = {
        "transcript_text": transcript_text,
        "duration_sec": duration_sec
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(EVALUATOR_ENDPOINT, json=payload, timeout=25)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries:
                print(f"      [!] 連線重試 ({e})，正在進行第 {attempt}/{max_retries} 次...")
                time.sleep(1.5)
            else:
                print(f"      [!] Netlify 審核端點異常: {e}")
                return {"qualified": False, "reasoning": f"遠端連線異常: {e}"}


# ---------------------------------------------------------------------------
# 2. 影片尺寸標準化與視覺檢驗模組
# ---------------------------------------------------------------------------
def resize_and_pad_frame(frame, target_w, target_h):
    """將單幀等比例縮放並置中補黑邊至 (target_w, target_h)"""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded_frame = cv2.copyMakeBorder(
        resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_frame


def convert_video_dimension(input_path: Path, output_path: Path, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT) -> bool:
    """優先使用 ffmpeg 保持原音軌轉換尺寸；失敗時回退 OpenCV"""
    if shutil.which("ffmpeg") is not None:
        vf_filter = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", vf_filter,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            pass

    # 回退 OpenCV
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = resize_and_pad_frame(frame, target_w, target_h)
        out.write(processed)

    cap.release()
    out.release()
    return True


def is_talking_head_video(video_url: str, sample_frames: int = 4) -> tuple[bool, str]:
    """
    輕量視覺快篩：透過 OpenCV 讀取串流，並使用 MediaPipe Tasks FaceLandmarker
    抽樣檢測是否含真人臉孔。若完全無人臉出鏡，直接判定不合格，排除純圖文簡報/投影片。
    """
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        return True, "無法讀取視訊串流，跳過視覺快篩"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return True, "幀數無效，跳過視覺快篩"

    # 若模型不存在則跳過視覺快篩，避免中斷批次
    if not Path(FACE_MODEL_PATH).exists():
        cap.release()
        return True, "找不到 face_landmarker.task，跳過視覺快篩"

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
            step = total_frames // (sample_frames + 1)
            for i in range(1, sample_frames + 1):
                frame_idx = step * i
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                checked_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                if detection_result.face_landmarks:
                    face_detected_count += 1
                    # 只要確認有真人露臉，即可提前中斷抽樣加速
                    break

    except Exception as e:
        cap.release()
        return True, f"視覺快篩異常 ({e})，安全跳過"

    cap.release()

    if checked_count > 0 and face_detected_count == 0:
        return False, f"畫面全無真人出鏡 (抽檢 {checked_count} 幀皆無人臉，判定為純圖文簡報/插圖)"

    return True, "通過視覺出鏡檢驗"


def ensure_video_aligned(video_path: Path) -> bool:
    """檢查尺寸並對齊為 720x1280"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w != TARGET_WIDTH or h != TARGET_HEIGHT:
        print(f"   📐 檢測到原始解析度為 {w}x{h}，正在對齊轉換為 {TARGET_WIDTH}x{TARGET_HEIGHT}...")
        temp_output = video_path.parent / f"{video_path.stem}_temp_aligned.mp4"
        if convert_video_dimension(video_path, temp_output, TARGET_WIDTH, TARGET_HEIGHT):
            video_path.unlink()
            temp_output.rename(video_path)
            print("   [+] 尺寸對齊與原音軌封裝完成！")
        else:
            print("   [!] 尺寸轉換失敗，保留原始影片。")

    return True


# ---------------------------------------------------------------------------
# 3. 單一帳號批次處理主迴圈
# ---------------------------------------------------------------------------
def process_account_json(json_path: Path):
    account_id = json_path.stem.replace(".json", "").strip()
    print("\n" + "=" * 80)
    print(f"📂 正在處理帳號: [{account_id}] (來源檔案: {json_path.name})")
    print("=" * 80)

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

    # 2. 檢查日期門檻：若 timestamp 比 CUTOFF_DATE 更晚，則剔除
    if CUTOFF_DATE:
        threshold = CUTOFF_DATE if "T" in CUTOFF_DATE else f"{CUTOFF_DATE}T23:59:59.999Z"
        original_count = len(raw_items)
        raw_items = [
            item for item in raw_items
            if str(item.get("timestamp", "")) <= threshold
        ]
        filtered_out = original_count - len(raw_items)
        if filtered_out > 0:
            print(f"[*] 依據日期門檻 [{CUTOFF_DATE}]，已剔除更晚發布的 {filtered_out} 條樣本。")

    # 3. 檢查擁有者名稱：刪除所有 ownerUsername 不等於指定名字的資料
    expected_username = TARGET_OWNER_USERNAME or account_id
    if expected_username:
        before_user_filter = len(raw_items)
        raw_items = [
            item for item in raw_items
            if str(item.get("ownerUsername", "")).strip().lower() == expected_username.strip().lower()
        ]
        filtered_user_count = before_user_filter - len(raw_items)
        if filtered_user_count > 0:
            print(f"[*] 依據 ownerUsername 門檻 [{expected_username}]，已剔除非該帳號的 {filtered_user_count} 條樣本。")

    print(f"[*] 該帳號篩選後共有 {len(raw_items)} 條待審查 Reels。")

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
                vid = r.get("video_id", "")
                m = re.search(r"video_(\d+)", vid)
                if m:
                    idx_val = int(m.group(1))
                    if idx_val >= next_video_idx:
                        next_video_idx = idx_val + 1
            print(f"[*] 載入現有紀錄: 已有 {len(processed_timestamps)} 條合格記錄，下一支編號: video_{next_video_idx:03d}")
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

        # -------------------------------------------------------------
        # 階段 1.5：視覺出鏡快篩 (杜絕純圖卡、投影片短片)
        # -------------------------------------------------------------
        if video_url:
            has_person, visual_reason = is_talking_head_video(video_url)
            if not has_person:
                print(f"   [✗ 排除淘汰] 理由: {visual_reason} (不佔用編號)")
                rejected_count += 1
                continue

        # 下載暫存音軌進行本地 SenseVoice 辨識
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)

        try:
            print("   ⬇️ 正在下載音軌小檔 (~1MB)...")
            if not download_file(audio_url, tmp_audio_path):
                continue

            print("   🎙️ 本地 SenseVoice 轉錄逐字稿中...")
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
                full_text = caption.strip()

            print("   ☁️ 傳送逐字稿至 Netlify 進行口播審核...")
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

            # 【執行尺寸對齊 720x1280】
            if video_downloaded and final_video_path.exists():
                ensure_video_aligned(final_video_path)

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
                "curation_result": eval_res,
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
# 4. 主程序
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【Reels 批次自動篩選與入庫引擎 (含自動尺寸對齊與出鏡快篩)】")
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