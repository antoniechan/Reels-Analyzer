import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

URL_FILE = "video_urls.txt"
BASE_DOWNLOAD_DIR = "D:\\reels\\"
SUBFOLDER_NAME = "nuture_fit_taichung"
MAX_WORKERS = 4


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
    if os.path.exists(URL_FILE):
        batch_download_from_file(URL_FILE, custom_subfolder=SUBFOLDER_NAME)
    else:
        print(f"找不到檔案 {URL_FILE}，請先執行抽取 URL 的程式。")