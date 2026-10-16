import glob
import json
import os

TARGET_DIR = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\reelsData"


def check_transcript_field(target_dir):
    if not os.path.exists(target_dir):
        print(f"錯誤：找不到目錄 -> {target_dir}")
        return

    # 遞迴搜尋目標資料夾及其子目錄下的所有 .json 檔案
    json_pattern = os.path.join(target_dir, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)

    print(f"掃描目錄: {target_dir}")
    print(f"共找到 {len(json_files)} 個 JSON 檔案。\n" + "-" * 60)

    total_files = len(json_files)
    files_with_missing_transcripts = 0
    total_records_checked = 0
    total_records_missing = 0

    for file_path in json_files:
        rel_path = os.path.relpath(file_path, target_dir)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[讀取失敗] {rel_path} -> 錯誤原因: {e}")
            continue

        records = []
        # 判斷 JSON 格式是 List 還是 單一 Dict
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # 若最外層是一個包裝物件，嘗試檢查內部的 records/items/data 陣列
            if (
                "records" in data
                and isinstance(data["records"], list)
            ):
                records = data["records"]
            elif "data" in data and isinstance(data["data"], list):
                records = data["data"]
            elif "items" in data and isinstance(data["items"], list):
                records = data["items"]
            else:
                # 視為單一筆資料物件
                records = [data]
        else:
            print(f"[格式略過] {rel_path} (不是 dict 或 list)")
            continue

        file_missing_count = 0
        for idx, item in enumerate(records):
            if isinstance(item, dict):
                total_records_checked += 1
                if "transcript" not in item:
                    file_missing_count += 1
                    total_records_missing += 1

        # 若該檔案有缺失，印出提示
        if file_missing_count > 0:
            files_with_missing_transcripts += 1
            print(
                f"[缺失] {rel_path} -> 共有 {file_missing_count}/{len(records)} 筆資料缺少 'transcript' 欄位"
            )

    print("-" * 60)
    print("【檢查統計結果】")
    print(f"總檢查檔案數: {total_files}")
    print(
        f"缺少 transcript 的檔案數: {files_with_missing_transcripts}"
    )
    print(f"總檢查資料筆數: {total_records_checked}")
    print(f"缺少 transcript 的總筆數: {total_records_missing}")

    if total_records_missing == 0 and total_records_checked > 0:
        print("\n 完美！所有 JSON 檔案中的資料均已包含 'transcript' 欄位。")
    elif total_records_missing > 0:
        print(f"\n 注意：尚有 {total_records_missing} 筆資料未填入 'transcript'。")


if __name__ == "__main__":
    check_transcript_field(TARGET_DIR)