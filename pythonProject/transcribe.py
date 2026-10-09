import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

# 路徑設定
REELS_DATA_DIR = Path(
    r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\reelsData"
)
REELS_AUDIO_BASE_DIR = Path(r"D:\Reels")

# 初始化 Gemini Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def get_file_index_key(item: dict) -> int:
  """從項目的 local_video_file 或 local_audio_file 提取序號數字作為排序依據；

  若無任何標註則賦予極大值排在後面。
  """
  target_str = item.get("local_video_file", "") or item.get(
      "local_audio_file", ""
  )
  match = re.search(r"video_(\d+)", target_str)
  if match:
    return int(match.group(1))
  return 999999


def transcribe_audio_file(audio_path: Path) -> str:
  """直接讀取本機音訊以二進位 (Inline) 方式傳入 Gemini 進行轉錄"""
  try:
    print(f"      讀取音訊: {audio_path.name}...")

    with open(audio_path, "rb") as f:
      audio_bytes = f.read()

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/mp3",
    )

    prompt = (
        "請將這段音訊完整轉錄為繁體中文逐字稿。"
        "僅輸出轉錄文字內容，不要添加任何前言、總結或多餘解說。"
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[audio_part, prompt],
    )

    return response.text.strip()

  except Exception as e:
    print(f"      轉錄失敗 ({audio_path.name}): {e}")
    return ""


def process_json_file(json_path: Path):
  file_stem = json_path.stem
  audio_folder = REELS_AUDIO_BASE_DIR / file_stem

  print(f"\n==========================================")
  print(f"開始處理: {json_path.name}")
  print(f"對應音訊資料夾: {audio_folder}")

  try:
    with open(json_path, "r", encoding="utf-8") as f:
      items = json.load(f)
  except Exception as e:
    print(f"讀取 {json_path.name} 失敗: {e}")
    return

  if not isinstance(items, list):
    print(f"檔案格式非列表，略過: {json_path.name}")
    return

  # 1. 按照 video_001, video_002... 的序號重新排序整個 JSON 陣列
  items.sort(key=get_file_index_key)

  has_audio_folder = audio_folder.exists() and audio_folder.is_dir()
  if not has_audio_folder:
    print(f"⚠️ 找不到對應音訊資料夾，將僅重整排序並校正檔名欄位。")

  # 2. 遍歷排序後的資料，重新校正/補上標準三位數檔名，並執行轉錄
  for idx, item in enumerate(items, start=1):
    expected_video_name = f"video_{idx:03d}.mp4"
    expected_audio_name = f"video_{idx:03d}.mp3"

    # 校正/補充序號欄位
    item["local_video_file"] = expected_video_name
    item["local_audio_file"] = expected_audio_name

    # 檢查音訊實體並轉錄
    if has_audio_folder:
      target_audio_file = audio_folder / expected_audio_name
      if target_audio_file.exists():
        # 若已有 transcript 且不為空可考慮略過，此處依邏輯直接轉錄
        print(f"   [{idx}/{len(items)}] 正在轉錄 {expected_audio_name}...")
        transcript = transcribe_audio_file(target_audio_file)
        if transcript:
          item["transcript"] = transcript
          print(f"      轉錄成功，已更新至 transcript。")
      else:
        print(f"   [{idx}/{len(items)}] 找不到檔案: {expected_audio_name}")

  # 寫回原 JSON 檔案
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

  print(f"已依序重排、更新完成並存檔: {json_path.name}")


def main():
  if not REELS_DATA_DIR.exists():
    print(f"找不到資料目錄: {REELS_DATA_DIR}")
    return

  json_files = list(REELS_DATA_DIR.glob("*.json"))
  print(f"共找到 {len(json_files)} 個 JSON 檔案。")

  for json_file in json_files:
    process_json_file(json_file)

  print("\n所有檔案處理完畢！")


if __name__ == "__main__":
  main()