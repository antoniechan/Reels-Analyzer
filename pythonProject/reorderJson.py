import json
import os

# CHANGE THE FILE NAME
SAVED_FILE = 'rita_wang_health'

# DO NOT AMEND THE FOLLOWING
ORIGINAL_FILE = 'temp.json'
output_dir = 'reelsData'
output_path = os.path.join(output_dir, f'{SAVED_FILE}.json')

if os.path.exists(output_path):
    print(f"⚠檔案 '{output_path}' 已經存在，為避免覆蓋已取消儲存操作。")
else:
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"成功將排序後的資料儲存至 '{output_path}'！")