import os
import re
import argparse


def shift_down_files(folder_path: str, threshold_num: int, execute: bool = False):
    """
    將資料夾中所有編號大於 threshold_num 的 video_xxx 系列檔案全部減 1。
    例如 threshold_num = 4:
      video_005.* -> video_004.*
      video_006.* -> video_005.*
      video_007.* -> video_006.*
    """
    if not os.path.exists(folder_path):
        print(f"[錯誤] 指定的資料夾不存在: {folder_path}")
        return

    # 正則表達式：匹配開頭為 video_ 加上純數字，並保留原數字字串與後續附屬名稱
    # 例如：video_005_qc.json -> group(1)="005", group(2)="_qc.json"
    pattern = re.compile(r"^video_(\d+)([\._].*)$")

    all_files = os.listdir(folder_path)
    tasks = []

    for fname in all_files:
        match = pattern.match(fname)
        if match:
            num_str = match.group(1)
            suffix = match.group(2)
            current_val = int(num_str)

            # 只有當數字嚴格大於 threshold_num 時才處理 (例如 > 4)
            if current_val > threshold_num:
                new_val = current_val - 1
                # 保持原本的位數補零格式 (如 3 位數 005 -> 004)
                pad_len = len(num_str)
                new_num_str = f"{new_val:0{pad_len}d}"
                new_fname = f"video_{new_num_str}{suffix}"

                old_path = os.path.join(folder_path, fname)
                new_path = os.path.join(folder_path, new_fname)

                tasks.append({
                    "old_val": current_val,
                    "old_fname": fname,
                    "new_fname": new_fname,
                    "old_path": old_path,
                    "new_path": new_path,
                })

    print("=" * 75)
    print("【Video 附屬檔案批次前移 (減 1) 工具】")
    print(f"  - 目標資料夾: {os.path.abspath(folder_path)}")
    print(f"  - 觸發門檻值: 編號 > {threshold_num}")
    print(f"  - 預計處理檔案數: {len(tasks)} 個")
    print("=" * 75)

    if not tasks:
        print(f"[提示] 資料夾中沒有找到編號大於 {threshold_num} 的 video 系列檔案。")
        return

    # 關鍵：依原數字「由小到大 (昇冪)」排序執行，避免目標檔名尚未釋放產生衝突
    tasks.sort(key=lambda x: (x["old_val"], x["old_fname"]))

    # 檢查第一棒是否會撞車
    first_target = tasks[0]["new_path"]
    if os.path.exists(first_target):
        print(f"\n[衝突警告] 目標檔案已存在: {tasks[0]['new_fname']}")
        print(f"請確認編號為 {tasks[0]['old_val'] - 1} 的檔案是否已被移除或刪除！")
        return

    # 列出變更預覽
    print("\n即將進行以下檔名變更（依序前移）：")
    for item in tasks:
        print(f"  {item['old_fname']:<35}  ->  {item['new_fname']}")

    # 人工確認
    if not execute:
        print("\n" + "-" * 75)
        confirm = input("確認執行上述前移（減 1）操作？(y/N): ").strip().lower()
        if confirm != "y":
            print("[取消] 操作已中止，未更改任何檔案。")
            return

    # 正式依序重命名
    success_count = 0
    for item in tasks:
        try:
            os.rename(item["old_path"], item["new_path"])
            success_count += 1
        except Exception as e:
            print(f"  [失敗] 重命名 {item['old_fname']} 時發生錯誤: {e}")
            break  # 若中間出錯立即中斷，避免後續鏈式碰撞

    print("\n" + "=" * 75)
    print(f"【操作完成】成功將 {success_count}/{len(tasks)} 個檔案前移（減 1）！")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將指定資料夾中大於指定門檻值的 video 系列檔案編號全部減 1")
    parser.add_argument("--dir", type=str, required=True, help="資料夾路徑 (例如: D:\\reels\\anminclinic)")
    parser.add_argument("--above", type=int, required=True, help="指定數字門檻 (例如: 4，則所有 > 004 的編號都會減 1)")
    parser.add_argument("--yes", action="store_true", help="略過人工確認，直接執行")
    args = parser.parse_args()

    shift_down_files(
        folder_path=args.dir,
        threshold_num=args.above,
        execute=args.yes
    )