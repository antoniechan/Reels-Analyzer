import argparse
import glob
import json
import os
import shutil
import subprocess
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
from config import VIDEO_PROC_DIR

# 預設參數設定
TARGET_DIR = r"D:\reels"
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
MODEL_FILE_NAME = str(VIDEO_PROC_DIR / "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
VIDEO_EXTS = ("*.mp4", "*.mov", "*.avi", "*.mkv")


def ensure_model_exists(model_path):
    """若本機不存在模型檔，則自動下載"""
    if not os.path.exists(model_path):
        print(f"找不到模型檔案 {model_path}，正在下載...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("模型下載完成！")


def resize_and_pad_frame(frame, target_w, target_h):
    """將單幀等比例縮放並置中補黑邊至 (target_w, target_h)，保持人臉長寬比不變形"""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded_frame = cv2.copyMakeBorder(
        resized_frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded_frame


def convert_video_dimension(
    input_path, output_path, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT
):
    """優先調用系統 ffmpeg 保持音軌，若無則使用 OpenCV VideoWriter 轉檔"""
    if shutil.which("ffmpeg") is not None:
        vf_filter = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            vf_filter,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-c:a",
            "copy",
            output_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            pass

    # 回退使用 OpenCV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = resize_and_pad_frame(frame, target_w, target_h)
        out.write(processed)

    cap.release()
    out.release()
    return True


def extract_landmarks_for_video(video_path, landmarker):
    """提取影片每秒第 1 幀的面部特徵點"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [錯誤] 無法讀取影片: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    frames_data = []
    current_sec = 0

    while True:
        target_frame_idx = int(round(current_sec * fps))
        if target_frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

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


def process_all_reels(
    root_dir=TARGET_DIR,
    target_w=TARGET_WIDTH,
    target_h=TARGET_HEIGHT,
    model_path=MODEL_FILE_NAME,
):
    ensure_model_exists(model_path)

    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        sys.exit(1)

    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(
            glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
        )

    print(f"掃描目錄: {root_dir}")
    print(
        f"找到 {len(video_files)} 個影片檔案，目標尺寸: {target_w} x"
        f" {target_h}\n" + "-" * 60
    )

    # 初始化 MediaPipe Face Landmarker（僅初始化一次）
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
        for idx, file_path in enumerate(video_files, 1):
            dir_name, file_name = os.path.split(file_path)
            base_name, _ = os.path.splitext(file_name)
            json_path = os.path.join(dir_name, f"{base_name}_face.json")

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(
                    f"[{idx}/{len(video_files)}] 無法開啟檔案，跳過:"
                    f" {file_name}"
                )
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            needs_rescale = w != target_w or h != target_h
            current_video_path = file_path

            # --- 步驟 1: 尺寸標準化 ---
            if needs_rescale:
                print(
                    f"[{idx}/{len(video_files)}] 尺寸不符 ({w}x{h})，調整至"
                    f" {target_w}x{target_h}: {file_name}"
                )
                temp_output = os.path.join(
                    dir_name, f"{base_name}_temp_{target_w}x{target_h}.mp4"
                )

                if convert_video_dimension(
                    file_path, temp_output, target_w, target_h
                ):
                    final_target_path = os.path.join(dir_name, f"{base_name}.mp4")
                    if (
                        os.path.exists(file_path)
                        and file_path != final_target_path
                    ):
                        os.remove(file_path)
                    shutil.move(temp_output, final_target_path)
                    current_video_path = final_target_path
                    print(f"    -> 尺寸調整成功: {os.path.basename(final_target_path)}")
                else:
                    print(f"    -> 尺寸轉換失敗，跳過該影片")
                    continue
            else:
                print(
                    f"[{idx}/{len(video_files)}] 尺寸已是"
                    f" {target_w}x{target_h}: {file_name}"
                )

            # --- 步驟 2: 計算 / 重新計算 Face Landmark ---
            # 條件：如果影片曾被縮放，或是 _face.json 本身不存在，則觸發計算
            if needs_rescale or not os.path.exists(json_path):
                action = (
                    "影片尺寸已變更，重新計算"
                    if needs_rescale
                    else "尚未提取特徵點，開始計算"
                )
                print(f"    -> {action} Face Landmark...")
                landmark_results = extract_landmarks_for_video(
                    current_video_path, landmarker
                )

                if landmark_results is not None:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            landmark_results, f, ensure_ascii=False, indent=2
                        )
                    print(f"    -> 已儲存/更新 JSON: {os.path.basename(json_path)}")
            else:
                print(f"    -> 尺寸一致且 {os.path.basename(json_path)} 已存在，略過計算。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "統一影片尺寸為 720x1080 並連動計算/更新 Face Landmarker JSON"
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=TARGET_DIR,
        help="Reels 影片目錄 (預設為 D:/reels)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_FILE_NAME,
        help="face_landmarker.task 模型路徑",
    )
    args = parser.parse_args()

    process_all_reels(args.root_dir, model_path=args.model)