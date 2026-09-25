from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import requests
from tqdm import tqdm

# CHANGE THE FOLDER NAME AND PATH
SUBFOLDER_NAME = "simplylita26726"
BASE_DOWNLOAD_DIR = "D:\\reels\\"
SCORE_CUTOFF = 0

# DO NOT AMEND THE FOLLOWING
MAX_WORKERS = 4


def extract_media_from_data(data, cutoff=0):
  """遞迴走訪資料結構，提取 score >= cutoff 且包含影音連結的完整物件。"""
  filtered_list = []

  if isinstance(data, list):
    for item in data:
      filtered_list.extend(extract_media_from_data(item, cutoff=cutoff))
  elif isinstance(data, dict):
    has_video = bool(data.get("videoUrl"))
    has_audio = bool(data.get("audioUrl"))

    # 取得 score 欄位
    raw_score = data.get("score")
    try:
      score_val = (
          float(raw_score)
          if raw_score is not None
          else (10.0 if not cutoff else 0.0)
      )
    except (ValueError, TypeError):
      score_val = 0.0

    # 必須包含真實影音連結且分數達到門檻 (同時排除占位字串)
    is_valid_url = (
        data.get("videoUrl", "").startswith("http")
        or data.get("audioUrl", "").startswith("http")
    )
    if (has_video or has_audio) and is_valid_url and (score_val >= cutoff):
      item_copy = dict(data)
      item_copy["score"] = score_val
      filtered_list.append(item_copy)

    # 繼續向巢狀層級搜尋 (若該 dict 本身已是單筆 reel 則略過其子結構走訪)
    if not (has_video or has_audio):
      for value in data.values():
        if isinstance(value, (dict, list)):
          filtered_list.extend(extract_media_from_data(value, cutoff=cutoff))

  return filtered_list


def extract_from_file_or_dir(
    input_path, output_file="media_urls.json", cutoff=0
):
  path = Path(input_path)
  all_items = []

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
        items = extract_media_from_data(content, cutoff=cutoff)
        all_items.extend(items)
        print(
            f"已從 {file.name} 擷取到 {len(items)} 筆符合門檻 (score >="
            f" {cutoff}) 的資料"
        )
    except Exception as e:
      print(f"解析 {file.name} 時出錯: {e}")

  # 去重處理（依 timestamp 與 caption，若無則依 videoUrl）
  seen = set()
  unique_items = []
  for item in all_items:
    key = (
        item.get("timestamp"),
        (item.get("caption") or "").strip(),
        item.get("videoUrl"),
    )
    if key not in seen:
      seen.add(key)
      unique_items.append(item)

  # 儲存篩選後的完整結構至暫存檔
  with open(output_file, "w", encoding="utf-8") as out:
    json.dump(unique_items, out, indent=2, ensure_ascii=False)

  print(
      f"\n完成！共取得 {len(unique_items)} 筆符合門檻的唯一資料，已寫入至"
      f" {output_file}"
  )
  return True


def download_file(url, file_path, desc_label):
  if not url or not url.startswith("http"):
    return True, "無有效連結，略過"

  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/120.0.0.0 Safari/537.36"
      )
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
        leave=False,
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
  score = item.get("score", "N/A")

  # 建立下載後對應的本地檔名記錄
  if item.get("videoUrl") and item["videoUrl"].startswith("http"):
    video_filename = f"video_{index:03d}.mp4"
    v_path = os.path.join(target_dir, video_filename)
    v_ok, v_msg = download_file(
        item["videoUrl"], v_path, f"{video_filename} (score: {score})"
    )
    results.append((v_ok, v_msg))
    item["local_video_file"] = video_filename

  if item.get("audioUrl") and item["audioUrl"].startswith("http"):
    audio_filename = f"video_{index:03d}.mp3"
    a_path = os.path.join(target_dir, audio_filename)
    a_ok, a_msg = download_file(
        item["audioUrl"], a_path, f"{audio_filename} (score: {score})"
    )
    results.append((a_ok, a_msg))
    item["local_audio_file"] = audio_filename

  all_success = all(ok for ok, _ in results) if results else True
  msgs = "\n".join(msg for _, msg in results) if results else "無可下載媒體"
  return all_success, msgs


def batch_download_from_file(media_file_path, custom_subfolder=None):
  subfolder_name = (
      custom_subfolder or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  )
  target_dir = os.path.join(BASE_DOWNLOAD_DIR, subfolder_name)
  os.makedirs(target_dir, exist_ok=True)
  print(f"檔案儲存路徑: {target_dir}")

  with open(media_file_path, "r", encoding="utf-8") as f:
    media_items = json.load(f)

  total_items = len(media_items)
  if total_items == 0:
    print("沒有任何符合條件的項目可供下載。")
    return

  print(f"共讀取到 {total_items} 筆項目，開始下載媒體...\n")

  success_count = 0
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_index = {
        executor.submit(
            download_media_pair, item, i + 1, total_items, target_dir
        ): i
        for i, item in enumerate(media_items)
    }

    for future in as_completed(future_to_index):
      success, msg = future.result()
      if msg:
        print(msg)
      if success:
        success_count += 1

  # 將裁剪後的 JSON（帶有 local_video_file 檔名映射）分別存回下載目錄與本地目錄
  output_json_name = f"{subfolder_name}_filtered.json"
  local_json_path = os.path.join(".\\reelsData\\", output_json_name)

  with open(local_json_path, "w", encoding="utf-8") as f:
    json.dump(media_items, f, indent=2, ensure_ascii=False)

  print(
      f"\n全部任務結束！成功處理: {success_count}/{total_items}"
      f" 組，儲存於: {target_dir}"
  )
  print(f"裁剪後 JSON 已同步輸出至：{local_json_path}")


if __name__ == "__main__":
  json_path = f"./reelsData/{SUBFOLDER_NAME}.json"
  temp_media_file = "media_urls.json"

  if os.path.exists(json_path):
    success = extract_from_file_or_dir(
        json_path, temp_media_file, cutoff=SCORE_CUTOFF
    )
    if success and os.path.exists(temp_media_file):
      batch_download_from_file(temp_media_file, custom_subfolder=SUBFOLDER_NAME)
  else:
    print(f"錯誤：找不到來源檔案 {json_path}")
