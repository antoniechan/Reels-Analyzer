import argparse
import json
import os
from config import DEFAULT_REELS_DIR, ACCOUNT_PROC_DIR

DEFAULT_ROOT_DIR = DEFAULT_REELS_DIR
DEFAULT_OUTPUT_JSON = str(ACCOUNT_PROC_DIR / "accounts_followers.json")


def generate_followers_template(
    root_dir=DEFAULT_ROOT_DIR, output_json=DEFAULT_OUTPUT_JSON
):
    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        return

    # 1. 抓取 root_dir 下的所有子資料夾名稱 (每個資料夾視為一個 account)
    folder_names = sorted(
        [
            f
            for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
            and not f.startswith(".")  # 排除隱藏資料夾
        ]
    )

    print(f"掃描根目錄: {root_dir}")
    print(f"找到 {len(folder_names)} 個帳號資料夾。\n" + "-" * 50)

    # 2. 檢查是否已有既有的 JSON 檔，若有則保留已填入的資料
    existing_data = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # 相容如果是字典或 accounts 清單結構
                if isinstance(loaded, dict) and "accounts" in loaded:
                    existing_data = loaded["accounts"]
                elif isinstance(loaded, dict):
                    existing_data = loaded
            print(f"發現既有設定檔: {output_json}，將自動保留已填入的數據。")
        except Exception as e:
            print(f"讀取既有檔案失敗: {e}，將建立全新檔案。")

    # 3. 組合輸出結構
    accounts_dict = {}
    new_added = 0
    preserved = 0

    for name in folder_names:
        if name in existing_data and existing_data[name] is not None:
            accounts_dict[name] = existing_data[name]
            preserved += 1
        else:
            # 預設 followersCount 為 None，亦可附帶已統計的影片數量方便核對
            accounts_dict[name] = {
                "followersCount": None,
                "video_count": len(
                    [
                        f
                        for f in os.listdir(os.path.join(root_dir, name))
                        if f.lower().endswith((".mp4", ".mov", ".mkv"))
                    ]
                ),
            }
            new_added += 1

    output_payload = {
        "total_accounts": len(accounts_dict),
        "accounts": accounts_dict,
    }

    # 4. 寫入 JSON 檔
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("【完成統計】")
    print(f"總帳號數:   {len(accounts_dict)}")
    print(f"新加入帳號: {new_added}")
    print(f"保留已存在: {preserved}")
    print(f"已輸出至:   {os.path.abspath(output_json)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="抓取所有目錄名稱並建立紀錄 followersCount 的 JSON 檔案"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_ROOT_DIR,
        help=f"Reels 目錄路徑 (預設: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f"輸出 JSON 路徑 (預設: {DEFAULT_OUTPUT_JSON})",
    )
    args = parser.parse_args()

    generate_followers_template(args.dir, args.output)