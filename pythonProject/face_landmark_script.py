import argparse
import glob
import json
import os
import sys
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 預設模型下載連結與檔名
MODEL_FILE_NAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
TARGET_DIR = r"D:/Reels"


def ensure_model_exists(model_path):
    """若本機不存在模型檔，則自動自 Google 下載"""
    if not os.path.exists(model_path):
        print(f"找不到模型檔案 {model_path}，正在下載...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("模型下載完成！")


def process_single_video(video_path, landmarker):
    """提取單個影片每秒第 1 幀的特徵點並返回清單"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [錯誤] 無法開啟影片: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        print(f"  [警告] 無法讀取有效的 FPS 或幀數: {video_path}")
        cap.release()
        return None

    frames_data = []
    current_sec = 0

    while True:
        # 計算每秒第 1 幀的 index (0, round(1*fps), round(2*fps), ...)
        target_frame_idx = int(round(current_sec * fps))
        if target_frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 轉換格式供 MediaPipe Tasks 使用
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )

        detection_result = landmarker.detect(mp_image)

        # 序列化 478 個特徵點 (x, y, z 及可選的世界座標/像素座標)
        landmarks_list = None
        if detection_result.face_landmarks:
            first_face = detection_result.face_landmarks[0]
            landmarks_list = [
                {
                    "id": idx,
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "px": int(round(lm.x * video_width)),
                    "py": int(round(lm.y * video_height)),
                }
                for idx, lm in enumerate(first_face)
            ]

        frames_data.append(
            {
                "second": current_sec,
                "frame_index": target_frame_idx,
                "face_detected": landmarks_list is not None,
                "landmarks": landmarks_list,
            }
        )

        current_sec += 1

    cap.release()

    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "resolution": {"width": video_width, "height": video_height},
        "sampled_seconds": len(frames_data),
        "frames": frames_data,
    }


def batch_process_reels(root_dir=TARGET_DIR, model_path=MODEL_FILE_NAME):
    ensure_model_exists(model_path)

    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        sys.exit(1)

    # 遞迴搜尋 root_dir 下所有資料夾中的 .mp4 檔案
    search_pattern = os.path.join(root_dir, "**", "*.mp4")
    video_files = glob.glob(search_pattern, recursive=True)

    print(f"搜尋目錄: {root_dir}")
    print(f"找到 {len(video_files)} 個 .mp4 檔案。\n")

    if not video_files:
        print("未找到任何影片檔案，程式結束。")
        return

    # 初始化 MediaPipe FaceLandmarker（只初始化一次，避免重複加載模型的開銷）
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for idx, video_path in enumerate(video_files, 1):
            folder_dir, file_name = os.path.split(video_path)
            base_name, _ = os.path.splitext(file_name)
            output_json_path = os.path.join(folder_dir, f"{base_name}_face.json")

            # 斷點檢查：若已經處理過則略過
            if os.path.exists(output_json_path):
                print(f"[{idx}/{len(video_files)}] 已存在，跳過: {output_json_path}")
                continue

            print(f"[{idx}/{len(video_files)}] 處理中: {file_name}")
            result_data = process_single_video(video_path, landmarker)

            if result_data is not None:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                print(f"  -> 已儲存: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批次提取 D:/Reels/ 下所有 MP4 影片每秒第一幀的 Face Mesh 特徵點並輸出 JSON"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=TARGET_DIR,
        help="Reels 影片根目錄 (預設為 D:/Reels)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_FILE_NAME,
        help="face_landmarker.task 模型路徑",
    )
    args = parser.parse_args()

    batch_process_reels(args.root_dir, args.model)