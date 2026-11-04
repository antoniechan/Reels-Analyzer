import os
import re
import argparse


def rename_files(folder_path: str, old_num: str, new_num: str, execute: bool = False):
    """
    批次將指定 folder 中開頭為 video_{old_num} 的相關檔案重新命名為 video_{new_num}。
    """
    if not os.path.exists(folder_path):
        print(f"[錯誤] 指定的資料夾不存在: {folder_path}")
        return

    # 正規表達式：匹配檔名開頭為 video_{old_num} 且後面接著非數字字符（如底線、點）的檔案
    # 例如：video_002.mp4, video_002_qc.json, video_002_audio_saliency.json
    pattern = re.compile(rf"^video_{re.escape(old_num)}([\._].*)$")

    all_files = sorted(os.listdir(folder_path))
    matched_tasks = []

    for fname in all_files:
        match = pattern.match(fname)
        if match:
            suffix = match.group(1)  # 取得後半部分，如 .mp4 或 _qc.json
            new_fname = f"video_{new_num}{suffix}"
            old_full_path = os.path.join(folder_path, fname)
            new_full_path = os.path.join(folder_path, new_fname)
            matched_tasks.append((old_full_path, new_full_path, fname, new_fname))

    print("=" * 75)
    print("【Video 附屬檔案批次重新命名工具】")
    print(f"  - 目標資料夾: {os.path.abspath(folder_path)}")
    print(f"  - 舊數字編號: {old_num}")
    print(f"  - 新數字編號: {new_num}")
    print(f"  - 配對到檔案數: {len(matched_tasks)} 個")
    print("=" * 75)

    if not matched_tasks:
        print("[提示] 沒有找到符合命名規則的檔案。")
        return

    # 檢查目標檔名是否已經存在，避免覆蓋現有檔案
    conflict_found = False
    for _, new_path, _, new_fname in matched_tasks:
        if os.path.exists(new_path):
            print(f"  [衝突警告] 目標檔案已存在: {new_fname}")
            conflict_found = True

    if conflict_found:
        print("\n[終止] 因存在檔名衝突，未執行任何重命名操作，請先確認目標編號是否已被佔用。")
        return

    # 列出變更預覽
    print("\n即將進行以下變更：")
    for _, _, old_fname, new_fname in matched_tasks:
        print(f"  {old_fname}  ->  {new_fname}")

    # 確認執行
    if not execute:
        print("\n" + "-" * 75)
        confirm = input("確認執行上述重命名？(y/N): ").strip().lower()
        if confirm != "y":
            print("[取消] 操作已中止，未更改任何檔案。")
            return

    # 開始重新命名
    success_count = 0
    for old_path, new_path, old_fname, new_fname in matched_tasks:
        try:
            os.rename(old_path, new_path)
            success_count += 1
        except Exception as e:
            print(f"  [失敗] 重命名 {old_fname} 時發生錯誤: {e}")

    print("\n" + "=" * 75)
    print(f"【操作完成】成功更改 {success_count}/{len(matched_tasks)} 個檔案！")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批次修改指定資料夾內特定數字的 video 系列附屬檔案名稱")
    parser.add_argument("--dir", type=str, required=True, help="檔案所在的資料夾路徑 (例如: D:\\reels\\anminclinic)")
    parser.add_argument("--old_num", type=str, required=True, help="欲更改的原數字 (例如: 002 或 2)")
    parser.add_argument("--new_num", type=str, required=True, help="新的數字 (例如: 005 或 5)")
    parser.add_argument("--yes", action="store_true", help="略過人工確認，直接執行")
    args = parser.parse_args()

    rename_files(
        folder_path=args.dir,
        old_num=args.old_num,
        new_num=args.new_num,
        execute=args.yes
    )