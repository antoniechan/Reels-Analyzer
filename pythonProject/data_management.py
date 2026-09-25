import json
import os

# 定義檔案路徑
FILE_A_PATH = "reelsData/fit_withdavidmiao.json"
FILE_B_PATH = "temp.json"
OUTPUT_PATH = "reelsData/fit_withdavidmiao_updated.json"

# 設定需要由檔案 B 覆蓋至檔案 A 的指定欄位
FIELDS_TO_UPDATE = [
    "videoUrl",
    "audioUrl",
    "videoPlayCount",
    "videoViewCount",
    "videoDuration",
    "likesCount",
    "commentsCount",
]


def update_json_fields(file_a, file_b, output_file, fields):
  if not os.path.exists(file_a) or not os.path.exists(file_b):
    print(f"錯誤：找不到檔案 {file_a} 或 {file_b}")
    return

  with open(file_a, "r", encoding="utf-8") as f:
    data_a = json.load(f)

  with open(file_b, "r", encoding="utf-8") as f:
    data_b = json.load(f)

  # 建立檔案 B 的查找字典：以 (timestamp, caption) 作為唯一鍵值
  # 若特定項目 caption 可能有微小首尾空白差異，使用 strip() 增加容錯
  lookup_b = {
      (item.get("timestamp"), (item.get("caption") or "").strip()): item
      for item in data_b
  }

  matched_count = 0
  unmatched_count = 0

  # 遍歷檔案 A 並用檔案 B 的相應欄位覆蓋
  for item_a in data_a:
    key = (item_a.get("timestamp"), (item_a.get("caption") or "").strip())

    if key in lookup_b:
      source_item = lookup_b[key]
      for field in fields:
        if field in source_item:
          item_a[field] = source_item[field]
      matched_count += 1
    else:
      unmatched_count += 1

  # 寫入更新後的結果
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_a, f, ensure_ascii=False, indent=4)

  print(f"處理完成！已輸出至: {output_file}")
  print(f"成功匹配並更新: {matched_count} 筆")
  if unmatched_count > 0:
    print(f"未在檔案 B 中找到對應資料: {unmatched_count} 筆")


if __name__ == "__main__":
  update_json_fields(FILE_A_PATH, FILE_B_PATH, OUTPUT_PATH, FIELDS_TO_UPDATE)