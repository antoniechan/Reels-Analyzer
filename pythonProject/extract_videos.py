import json
from pathlib import Path
import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# CHANGE THE FOLDER NAME AND PATH
SUBFOLDER_NAME = "rita_wang_health"
BASE_DOWNLOAD_DIR = "D:\\reels\\"

# DO NOT AMEND THE FOLLOWING
MAX_WORKERS = 4

def extract_media_from_data(data):
    media_list = []

    if isinstance(data, list):
        for item in data:
            media_list.extend(extract_media_from_data(item))
    elif isinstance(data, dict):
        has_video = bool(data.get("videoUrl"))
        has_audio = bool(data.get("audioUrl"))

        if has_video or has_audio:
            media_list.append({
                "videoUrl": data.get("videoUrl"),
                "audioUrl": data.get("audioUrl")
            })

        for value in data.values():
            if isinstance(value, (dict, list)):
                media_list.extend(extract_media_from_data(value))

    return media_list


def extract_from_file_or_dir(input_path, output_file="media_urls.json"):
    path = Path(input_path)
    all_media = []

    if path.is_file():
        json_files = [path]
    elif path.is_dir():
        json_files = list(path.glob("*.json"))
    else:
        print(f"錯誤：找不到路徑 {input_path}")
        return False

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = json.load(f)
                items = extract_media_from_data(content)
                all_media.extend(items)
                print(f"已從 {file.name} 擷取到 {len(items)} 筆媒體資料")
        except Exception as e:
            print(f"解析 {file.name} 時出錯: {e}")

    seen = set()
    unique_media = []
    for item in all_media:
        key = (item.get("videoUrl"), item.get("audioUrl"))
        if key not in seen:
            seen.add(key)
            unique_media.append(item)

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(unique_media, out, indent=2, ensure_ascii=False)

    print(f"\n完成！共取得 {len(unique_media)} 筆唯一媒體資料，已寫入至 {output_file}")
    return True


def download_file(url, file_path, desc_label):
    if not url:
        return True, "無連結，略過"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024

        with open(file_path, "wb") as f, tqdm(
                desc=desc_label,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        return True, f"下載完成: {os.path.basename(file_path)}"
    except Exception as e:
        return False, f"下載失敗: {os.path.basename(file_path)} ({e})"


def download_media_pair(item, index, total, target_dir):
    results = []

    if item.get("videoUrl"):
        video_filename = f"video_{index:03d}.mp4"
        v_path = os.path.join(target_dir, video_filename)
        v_ok, v_msg = download_file(item["videoUrl"], v_path, video_filename)
        results.append((v_ok, v_msg))

    if item.get("audioUrl"):
        audio_filename = f"video_{index:03d}.mp3"
        a_path = os.path.join(target_dir, audio_filename)
        a_ok, a_msg = download_file(item["audioUrl"], a_path, audio_filename)
        results.append((a_ok, a_msg))

    all_success = all(ok for ok, _ in results)
    msgs = "\n".join(msg for _, msg in results)
    return all_success, msgs


def batch_download_from_file(media_file_path, custom_subfolder=None):
    subfolder_name = custom_subfolder or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target_dir = os.path.join(BASE_DOWNLOAD_DIR, subfolder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"檔案儲存路徑: {target_dir}")

    with open(media_file_path, "r", encoding="utf-8") as f:
        media_items = json.load(f)

    total_items = len(media_items)
    print(f"共讀取到 {total_items} 筆項目，開始下載...\n")

    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(download_media_pair, item, i + 1, total_items, target_dir): i
            for i, item in enumerate(media_items)
        }

        for future in as_completed(future_to_index):
            success, msg = future.result()
            print(msg)
            if success:
                success_count += 1

    print(f"\n全部任務結束！成功處理: {success_count}/{total_items} 組，儲存於: {target_dir}")


if __name__ == "__main__":
    json_path = f"./reelsData/{SUBFOLDER_NAME}.json"
    temp_media_file = "media_urls.json"

    if os.path.exists(json_path):
        success = extract_from_file_or_dir(json_path, temp_media_file)
        if success and os.path.exists(temp_media_file):
            batch_download_from_file(temp_media_file, custom_subfolder=SUBFOLDER_NAME)
    else:
        print(f"錯誤：找不到來源檔案 {json_path}")