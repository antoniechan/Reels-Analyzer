import argparse
import glob
import json
import os
import sys
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import VIDEO_PROC_DIR, DEFAULT_REELS_DIR

# 預設模型下載連結與路徑設定
MODEL_FILE_NAME = str(VIDEO_PROC_DIR / "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
TARGET_DIR = DEFAULT_REELS_DIR
DEFAULT_VIDEO_CLASS_JSON = str(VIDEO_PROC_DIR / "video_classification_summary.json")


def ensure_model_exists(model_path):
    """若本機不存在模型檔，則自動自 Google 下載"""
    if not os.path.exists(model_path):
        print(f"找不到模型檔案 {model_path}，正在下載...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("模型下載完成！")


def load_video_classification_map(json_path: str) -> dict:
    """
    讀取 video_classification_summary.json，建立雙重索引字典：
    Key 為 (account_id, video_id) 與 (account_id, local_index_name)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"找不到影片分類檔案: {os.path.abspath(json_path)}。\n"
            f"請確認路徑，或使用 --class_json 指定正確位置。"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos_list = data.get("videos", data) if isinstance(data, dict) else data
    lookup_map = {}

    for item in videos_list:
        if not isinstance(item, dict):
            continue
        acc_id = str(item.get("account_id", "")).strip()
        vid_id = str(item.get("video_id", "")).strip()
        local_name = str(item.get("local_index_name", "")).strip()

        if acc_id and vid_id:
            lookup_map[(acc_id, vid_id)] = item
        if acc_id and local_name:
            lookup_map[(acc_id, local_name)] = item

    return lookup_map


def get_classified_video_tasks(root_dir: str, lookup_map: dict, target_tier: str = "all") -> list:
    """
    掃描 root_dir 下所有 .mp4 檔案，並依據 video_classification_summary.json 分組與排序
    """
    target_tier = target_tier.lower()
    search_pattern = os.path.join(root_dir, "**", "*.mp4")
    all_mp4_files = sorted(glob.glob(search_pattern, recursive=True))

    matched_tasks = []
    unmatched_count = 0

    for video_path in all_mp4_files:
        folder_dir, file_name = os.path.split(video_path)
        account_id = os.path.basename(folder_dir)
        video_stem, _ = os.path.splitext(file_name)

        class_info = lookup_map.get((account_id, video_stem))
        if class_info is None:
            unmatched_count += 1
            continue

        vid_tier = str(class_info.get("video_classification", "neutral")).lower()
        if target_tier != "all" and vid_tier != target_tier:
            continue

        matched_tasks.append({
            "video_path": video_path,
            "account_id": account_id,
            "video_id": video_stem,
            "class_info": class_info,
        })

    # 按 good -> neutral -> bad 及影片品質分數由高至低排序
    tier_order = {"good": 0, "neutral": 1, "bad": 2}
    matched_tasks.sort(
        key=lambda x: (
            tier_order.get(str(x["class_info"].get("video_classification", "neutral")).lower(), 3),
            -float(x["class_info"].get("video_quality_score", 0.0)),
        )
    )

    if unmatched_count > 0:
        print(f"  [提示] 共有 {unmatched_count} 個 .mp4 檔案不在分類名單內，已自動略過。")

    return matched_tasks


def process_single_video(video_path, landmarker, class_info=None):
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
    detected_count = 0

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
            detected_count += 1
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

    sampled_secs = len(frames_data)
    face_presence_ratio = (detected_count / sampled_secs) if sampled_secs > 0 else 0.0

    payload = {
        "video_path": video_path,
        "account_id": class_info.get("account_id") if class_info else None,
        "video_id": class_info.get("video_id") if class_info else None,
        "video_classification_info": {
            "video_classification": class_info.get("video_classification", "neutral"),
            "video_quality_score": class_info.get("video_quality_score"),
            "raw_residual": class_info.get("raw_residual"),
            "plays": class_info.get("plays"),
            "likes": class_info.get("likes"),
            "comments": class_info.get("comments"),
            "followers": class_info.get("followers"),
        } if class_info else {},
        "fps": fps,
        "total_frames": total_frames,
        "resolution": {"width": video_width, "height": video_height},
        "sampled_seconds": sampled_secs,
        "face_detected_seconds": detected_count,
        "face_presence_ratio": round(float(face_presence_ratio), 4),
        "frames": frames_data,
    }
    return payload


def batch_process_reels(
    root_dir=TARGET_DIR,
    model_path=MODEL_FILE_NAME,
    class_json_path=DEFAULT_VIDEO_CLASS_JSON,
    target_tier="all",
    overwrite=False,
):
    ensure_model_exists(model_path)

    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        sys.exit(1)

    lookup_map = load_video_classification_map(class_json_path)
    tasks = get_classified_video_tasks(root_dir, lookup_map, target_tier=target_tier)

    print("=" * 70)
    print("【MediaPipe Face Mesh 提取系統 — Video Classification 分組模式】")
    print(f"  - 搜尋根目錄:               {root_dir}")
    print(f"  - 分類依據檔案:             {class_json_path}")
    print(f"  - 目標影片分組 (Tier):      {target_tier.upper()}")
    print(f"  - 符合條件的 MP4 影片總數:   {len(tasks)}")
    print("=" * 70 + "\n")

    if not tasks:
        print("未找到符合分組條件的影片檔案，程式結束。")
        return

    with open(model_path, "rb") as f:
        model_buffer = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    tier_presence_stats = {"good": [], "neutral": [], "bad": []}

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for idx, task in enumerate(tasks, 1):
            video_path = task["video_path"]
            class_info = task["class_info"]
            vid_tier = str(class_info.get("video_classification", "neutral")).lower()
            vid_score = float(class_info.get("video_quality_score", 0.0))

            folder_dir, file_name = os.path.split(video_path)
            base_name, _ = os.path.splitext(file_name)
            output_json_path = os.path.join(folder_dir, f"{base_name}_face.json")
            rel_path = os.path.relpath(video_path, root_dir)

            # 斷點檢查：若已存在則自動補齊/更新 video_classification_info 並統計人臉出現率
            if os.path.exists(output_json_path) and not overwrite:
                print(f"[{idx}/{len(tasks)}] [{vid_tier.upper()}] 已存在，更新分類標籤並跳過: {rel_path}")
                try:
                    with open(output_json_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    frames = existing.get("frames", [])
                    det_cnt = sum(1 for fr in frames if fr.get("face_detected"))
                    ratio = (det_cnt / len(frames)) if frames else 0.0

                    existing["video_classification_info"] = {
                        "video_classification": vid_tier,
                        "video_quality_score": vid_score,
                        "raw_residual": class_info.get("raw_residual"),
                        "plays": class_info.get("plays"),
                        "likes": class_info.get("likes"),
                        "comments": class_info.get("comments"),
                        "followers": class_info.get("followers"),
                    }
                    existing["face_detected_seconds"] = det_cnt
                    existing["face_presence_ratio"] = round(float(ratio), 4)

                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(existing, f, ensure_ascii=False, indent=2)

                    if vid_tier in tier_presence_stats:
                        tier_presence_stats[vid_tier].append(ratio)
                except Exception:
                    pass
                continue

            print(f"[{idx}/{len(tasks)}] [{vid_tier.upper()} | Score: {vid_score:.1f}] 處理中: {rel_path}")
            result_data = process_single_video(video_path, landmarker, class_info=class_info)

            if result_data is not None:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                ratio = result_data["face_presence_ratio"]
                if vid_tier in tier_presence_stats:
                    tier_presence_stats[vid_tier].append(ratio)
                print(f"  -> 已儲存 (人臉出現率: {ratio*100:.1f}%): {output_json_path}")

    # 印出各 Tier 的人臉出現比例統計
    print("\n" + "=" * 70)
    print("【各 Video Tier 人臉偵測覆蓋率統計 (Face Presence Ratio)】")
    for t in ["good", "neutral", "bad"]:
        vals = tier_presence_stats[t]
        if vals:
            print(f"  - {t.upper():<8} (共 {len(vals):>4} 支影片): 平均人臉出現率 = {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}%")
        else:
            print(f"  - {t.upper():<8} (共    0 支影片): N/A")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="依據 video_classification_summary.json 分組提取 MP4 影片每秒第一幀的 Face Mesh 特徵點"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=TARGET_DIR,
        help="Reels 影片根目錄 (預設為 D:\\reels)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_FILE_NAME,
        help="face_landmarker.task 模型路徑",
    )
    parser.add_argument(
        "--class_json",
        type=str,
        default=DEFAULT_VIDEO_CLASS_JSON,
        help="video_classification_summary.json 檔案路徑",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "good", "neutral", "bad"],
        help="選擇要處理的影片級別 (預設: all，可選: good, neutral, bad)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若加上此參數，將強制重新提取並覆寫既有的 *_face.json",
    )
    args = parser.parse_args()

    batch_process_reels(
        root_dir=args.root_dir,
        model_path=args.model,
        class_json_path=args.class_json,
        target_tier=args.tier,
        overwrite=args.overwrite,
    )