import json
from pathlib import Path

INPUT_PATH = "./reelsData/nuture_fit_taichung.json"

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

    print(f"\n完成！共取得 {len(unique_urls)} 個唯一 videoUrl，已寫入至 {output_file}")


if __name__ == "__main__":
    # 將 'data.json' 改為你的 JSON 檔案路徑，或填入資料夾路徑（例如：'./json_folder'）
    OUTPUT_FILE = "video_urls.txt"

    extract_from_file_or_dir(INPUT_PATH, OUTPUT_FILE)
