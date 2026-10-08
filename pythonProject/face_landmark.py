import argparse
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


def ensure_model_exists(model_path):
    """若本機不存在模型檔，則自動下載"""
    if not os.path.exists(model_path):
        print(f"找不到模型檔案 {model_path}，正在自 Google 下載...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("模型下載完成！")


def process_video_landmarks(video_path, model_path=MODEL_FILE_NAME):
    ensure_model_exists(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps) if fps > 0 else 0

    print(f"影片載入成功:")
    print(f" - FPS: {fps:.2f}")
    print(f" - 總幀數: {total_frames}")
    print(f" - 預估時長: {duration_sec} 秒")
    print("正在初始化 MediaPipe FaceLandmarker (Tasks API)...")

    # 配置新版 Tasks API FaceLandmarker
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

    extracted_data = []

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        current_sec = 0
        while True:
            target_frame_idx = int(round(current_sec * fps))
            if target_frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # 轉換為 RGB 並包裝為 mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )

            # 執行臉部特徵點檢測
            detection_result = landmarker.detect(mp_image)

            face_landmarks = None
            if detection_result.face_landmarks:
                # 取得第一張臉的 478 個特徵點 (含虹膜)
                face_landmarks = detection_result.face_landmarks[0]

            extracted_data.append(
                {
                    "sec": current_sec,
                    "frame_idx": target_frame_idx,
                    "image": frame,
                    "landmarks": face_landmarks,
                }
            )

            current_sec += 1

    cap.release()
    print(f"提取完成！共提取 {len(extracted_data)} 個關鍵秒數幀。")
    return extracted_data


def draw_face_landmarks(image, landmarks):
    """繪製特徵點（若環境未自帶舊版 drawing_utils，此處以原生 OpenCV 繪製）"""
    h, w, _ = image.shape
    for idx, lm in enumerate(landmarks):
        # 取得像素座標
        px, py = int(lm.x * w), int(lm.y * h)

        # 區分虹膜點 (index 468~477) 與面部點
        if idx >= 468:
            cv2.circle(image, (px, py), 2, (0, 0, 255), -1)  # 紅色標示虹膜
        else:
            cv2.circle(
                image, (px, py), 1, (0, 255, 0), -1
            )  # 綠色標示面部特徵點


def interactive_viewer(extracted_data):
    if not extracted_data:
        print("沒有提取到任何有效幀。")
        return

    window_name = "MediaPipe FaceLandmarker Viewer (Press 'q' to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_steps = len(extracted_data)
    current_idx = 0

    def on_trackbar(val):
        nonlocal current_idx
        current_idx = val

    cv2.createTrackbar("Sec", window_name, 0, total_steps - 1, on_trackbar)

    print("\n--- 操作說明 ---")
    print(" [A] 或 [<-] ：上一秒")
    print(" [D] 或 [->] ：下一秒")
    print(" [Q] 或 [ESC]：離開視窗")
    print(" 亦可直接拖曳上方滾動條 (Trackbar)")

    while True:
        data = extracted_data[current_idx]
        display_frame = data["image"].copy()

        if data["landmarks"]:
            draw_face_landmarks(display_frame, data["landmarks"])
            status_text = "Face Detected (478 pts)"
            status_color = (0, 255, 0)
        else:
            status_text = "No Face Detected"
            status_color = (0, 0, 255)

        info_text = f"Time: {data['sec']}s / {total_steps-1}s (Frame #{data['frame_idx']}) - {status_text}"
        cv2.putText(
            display_frame,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            break
        elif key in [ord("a"), ord("A"), 81]:
            if current_idx > 0:
                current_idx -= 1
                cv2.setTrackbarPos("Sec", window_name, current_idx)
        elif key in [ord("d"), ord("D"), 83]:
            if current_idx < total_steps - 1:
                current_idx += 1
                cv2.setTrackbarPos("Sec", window_name, current_idx)

        if cv2.getTrackbarPos("Sec", window_name) != current_idx:
            cv2.setTrackbarPos("Sec", window_name, current_idx)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用新版 MediaPipe Face Landmarker (Tasks API) 提取影片關鍵幀特徵點"
    )
    parser.add_argument("video_path", type=str, help="輸入的 MP4 影片路徑")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_FILE_NAME,
        help="face_landmarker.task 模型路徑",
    )
    args = parser.parse_args()

    results = process_video_landmarks(args.video_path, args.model)
    interactive_viewer(results)