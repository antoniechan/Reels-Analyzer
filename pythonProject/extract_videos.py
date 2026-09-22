import json
from pathlib import Path
import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SUBFOLDER_NAME = "rd.grace.f"
BASE_DOWNLOAD_DIR = "D:\\reels\\"
MAX_WORKERS = 4

def extract_urls_from_data(data):
    """遞迴或遍歷提取所有 videoUrl"""
    urls = []
    if isinstance(data, list):
        for item in data:
            urls.extend(extract_urls_from_data(item))
    elif isinstance(data, dict):
        if "videoUrl" in data and data["videoUrl"]:
            urls.append(data["videoUrl"])
        # 若 JSON 結構有巢狀層級，繼續往下找
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                urls.extend(extract_urls_from_data(value))
    return urls


def extract_from_file_or_dir(input_path, output_file="urls.txt"):
    """
    從指定檔案或資料夾讀取 JSON 並匯出 videoUrl
    :param input_path: JSON 檔案路徑 或 包含 JSON 的資料夾路徑
    :param output_file: 匯出的 txt 檔名
    """
    path = Path(input_path)
    all_urls = []

    if path.is_file():
        json_files = [path]
    elif path.is_dir():
        json_files = list(path.glob("*.json"))
    else:
        print(f"錯誤：找不到路徑 {input_path}")
        return

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = json.load(f)
                urls = extract_urls_from_data(content)
                all_urls.extend(urls)
                print(f"已從 {file.name} 擷取到 {len(urls)} 個連結")
        except Exception as e:
            print(f"解析 {file.name} 時出錯: {e}")

    # 去除重複項（若需要保留順序可改用 list(dict.fromkeys(all_urls))）
    unique_urls = list(dict.fromkeys(all_urls))

    # 寫入文字檔
    with open(output_file, "w", encoding="utf-8") as out:
        for url in unique_urls:
            out.write(url + "\n")

    print(f"\n完成！從 {input_path} 共取得 {len(unique_urls)} 個唯一 videoUrl，已寫入至 {output_file}")

def download_video(url, index, total, target_dir):
    """下載單個影片到指定子資料夾"""
    filename = f"video_{index:03d}.mp4"
    file_path = os.path.join(target_dir, filename)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024

        with open(file_path, "wb") as f, tqdm(
                desc=filename,
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

        return True, f"✅ 下載完成: {filename}"
    except Exception as e:
        return False, f"❌ 下載失敗 (第 {index} 部): {e}"


def batch_download_from_file(url_file_path, custom_subfolder=None):
    # 決定子資料夾名稱：未指定就用時間戳記 (例如: batch_20260921_110000)
    if not custom_subfolder:
        subfolder_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        subfolder_name = custom_subfolder

    # 組合出完整路徑: downloaded_videos/batch_xxxxxx/
    target_dir = os.path.join(BASE_DOWNLOAD_DIR, subfolder_name)

    # 建立目錄 (包含父層與子層)
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 影片儲存路徑: {target_dir}")

    with open(url_file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    total_videos = len(urls)
    print(f"共讀取到 {total_videos} 個影片連結，開始下載...\n")

    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {
            executor.submit(download_video, url, i + 1, total_videos, target_dir): i
            for i, url in enumerate(urls)
        }

        for future in as_completed(future_to_url):
            success, msg = future.result()
            print(msg)
            if success:
                success_count += 1

    print(f"\n全部任務結束！成功: {success_count}/{total_videos}，儲存於: {target_dir}")


if __name__ == "__main__":
    # 將 'data.json' 改為你的 JSON 檔案路徑，或填入資料夾路徑（例如：'./json_folder')

    extract_from_file_or_dir(f"./reelsData/{SUBFOLDER_NAME}.json", "video_urls.txt")

    if os.path.exists("video_urls.txt"):
        batch_download_from_file("video_urls.txt", custom_subfolder=SUBFOLDER_NAME)
    else:
        print(f"找不到檔案 {URL_FILE}，請先執行抽取 URL 的程式。")
