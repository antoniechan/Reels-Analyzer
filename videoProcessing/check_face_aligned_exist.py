import argparse
import glob
import os

DEFAULT_DIR = r"D:\reels"
VIDEO_EXTS = ("*.mp4", "*.mov", "*.avi", "*.mkv")


def check_unaligned_files(target_dir=DEFAULT_DIR):
    if not os.path.exists(target_dir):
        print(f"錯誤：目錄不存在 -> {target_dir}")
        return

    print(f"正在掃描目錄: {target_dir}\n" + "-" * 65)

    # 1. 搜尋所有影片檔案
    all_videos = []
    for ext in VIDEO_EXTS:
        all_videos.extend(
            glob.glob(os.path.join(target_dir, "**", ext), recursive=True)
        )

    # 2. 搜尋所有原始特徵點 JSON (_face.json)，排除已對齊的 (*_aligned.json)
    all_face_jsons = [
        f
        for f in glob.glob(
            os.path.join(target_dir, "**", "*_face.json"), recursive=True
        )
        if not f.endswith("_face_aligned.json")
    ]

    # 分類記錄列表
    missing_aligned_from_face = (
        []
    )  # 有 _face.json 但缺少 _face_aligned.json
    completely_unprocessed_videos = (
        []
    )  # 有影片但連 _face.json 與 aligned.json 都沒有
    successfully_aligned = []  # 已順利生成 _face_aligned.json

    # --- 檢查已有 _face.json 的對齊狀態 ---
    for face_json in all_face_jsons:
        dir_name, file_name = os.path.split(face_json)
        base_name, _ = os.path.splitext(file_name)  # 例如 "video_001_face"
        aligned_json_path = os.path.join(
            dir_name, f"{base_name}_aligned.json"
        )

        if not os.path.exists(aligned_json_path):
            missing_aligned_from_face.append(face_json)
        else:
            successfully_aligned.append(aligned_json_path)

    # --- 交叉比對影片檔案 ---
    for video in all_videos:
        dir_name, file_name = os.path.split(video)
        base_name, _ = os.path.splitext(file_name)
        face_json_path = os.path.join(dir_name, f"{base_name}_face.json")
        aligned_json_path = os.path.join(
            dir_name, f"{base_name}_face_aligned.json"
        )

        # 若影片既沒有 _face.json 也沒有 _face_aligned.json
        if not os.path.exists(face_json_path) and not os.path.exists(
            aligned_json_path
        ):
            completely_unprocessed_videos.append(video)

    # 3. 輸出詳細報告
    print("【掃描與檢查結果摘要】")
    print(f"總影片數量 (Videos):           {len(all_videos)}")
    print(f"已提取特徵點檔案 (_face.json):  {len(all_face_jsons)}")
    print(f"已成功完成對齊 (_aligned.json): {len(successfully_aligned)}")
    print(
        f"待對齊數量 (缺少 _aligned.json): {len(missing_aligned_from_face)}"
    )
    if completely_unprocessed_videos:
        print(
            f"完全尚未提取特徵點之影片:      {len(completely_unprocessed_videos)}"
        )
    print("-" * 65)

    # 列出有 _face.json 但缺少 _aligned.json 的檔案
    if missing_aligned_from_face:
        print("\n[待執行剛體對齊的檔案清單] (有 _face.json 但無對應的 _aligned.json):")
        for idx, path in enumerate(missing_aligned_from_face, 1):
            rel_path = os.path.relpath(path, target_dir)
            print(f"  {idx}. {rel_path}")

    # 列出完全無 JSON 的影片
    if completely_unprocessed_videos:
        print("\n[尚未進行 Face Mesh 提取的影片清單] (完全缺少 JSON):")
        for idx, path in enumerate(completely_unprocessed_videos, 1):
            rel_path = os.path.relpath(path, target_dir)
            print(f"  {idx}. {rel_path}")

    if not missing_aligned_from_face and not completely_unprocessed_videos:
        print("\n 狀態良好！所有特徵點檔案均已成功生成對應的 `_face_aligned.json`。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="檢查目錄中尚未生成 video_xxx_face_aligned.json 的檔案"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help=f"搜尋目標根目錄 (預設: {DEFAULT_DIR})",
    )
    args = parser.parse_args()

    check_unaligned_files(args.dir)