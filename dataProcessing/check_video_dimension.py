import glob
import os
import cv2
from config import DEFAULT_REELS_DIR

TARGET_DIR = DEFAULT_REELS_DIR
VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.avi", "*.mkv")


def check_video_resolutions(root_dir):
    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        return

    # 搜尋所有支援的影片格式
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        pattern = os.path.join(root_dir, "**", ext)
        video_files.extend(glob.glob(pattern, recursive=True))

    total_videos = len(video_files)
    print(f"掃描目錄: {root_dir}")
    print(f"共找到 {total_videos} 個影片檔案。\n" + "-" * 50)

    if total_videos == 0:
        return

    # 記錄解析度分組: {(width, height): [file_paths]}
    resolutions = {}
    failed_files = []

    for file_path in video_files:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            failed_files.append(file_path)
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        res_key = (width, height)
        resolutions.setdefault(res_key, []).append(file_path)

    # 輸出檢查結果
    print("【解析度統計結果】")
    for (w, h), files in resolutions.items():
        print(f"解析度 {w} x {h}: 共 {len(files)} 部影片")

    if failed_files:
        print(f"\n[無法讀取的檔案]: {len(failed_files)} 部")
        for f in failed_files:
            print(f"  - {f}")

    print("-" * 50)
    if len(resolutions) == 1 and not failed_files:
        w, h = list(resolutions.keys())[0]
        print(f"確認完畢：所有影片尺寸完全一致（均為 {w} x {h}）。")
    elif len(resolutions) > 1:
        print(f"注意：發現 {len(resolutions)} 種不同的解析度，尺寸不完全一致！")
        # 找出數量最多者視為基準，印出不符合基準的檔案
        main_res = max(resolutions, key=lambda k: len(resolutions[k]))
        print(f"主要解析度為 {main_res[0]} x {main_res[1]}，以下為不符合的影片：")
        for res, files in resolutions.items():
            if res != main_res:
                print(f"\n* 尺寸 {res[0]} x {res[1]} ({len(files)} 部):")
                for f in files:
                    rel_path = os.path.relpath(f, root_dir)
                    print(f"    {rel_path}")


if __name__ == "__main__":
    check_video_resolutions(TARGET_DIR)