#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Face Landmark Extraction & Kabsch Rigid Alignment Engine
-----------------------------------------------------------------
功能：
1. 依據 video_classification_summary.json 篩選目標影片 (支援 good/neutral/bad/all)。
2. 使用 MediaPipe Face Landmarker 每秒提取第 1 幀之 478 個 3D 特徵點。
3. 記憶體內即時使用 Kabsch Algorithm (SVD) 剛體對齊至基準人臉 M_0。
4. 一步到位輸出標準化的 *_face_aligned.json，包含人臉出現率、剛體運動參數與品質評級標籤。
"""

import argparse
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import VIDEO_PROC_DIR, DEFAULT_REELS_DIR

# ---------------------------------------------------------------------------
# 常數與路徑設定
# ---------------------------------------------------------------------------
MODEL_FILE_NAME = str(VIDEO_PROC_DIR / "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_VIDEO_CLASS_JSON = str(VIDEO_PROC_DIR / "video_classification_summary.json")

# MediaPipe Face Mesh 中不受表情牽動的解剖剛性錨點 (Rigid Anchors)
RIGID_ANCHOR_INDICES = [
    6,    # 鼻樑中心 (Nasion / Mid-nose)
    168,  # 雙眼眉間骨 (Glabella)
    197,  # 鼻樑上方骨骼點
    195,  # 鼻樑骨上部
    5,    # 鼻下緣上方骨質處
    33,   # 右眼內眼角骨質邊緣
    133,  # 右眼外側眼眶骨
    263,  # 左眼內眼角骨質邊緣
    362,  # 左眼外側眼眶骨
]


# ---------------------------------------------------------------------------
# 1. 基礎工具函式與 Kabsch 演算法
# ---------------------------------------------------------------------------
def ensure_model_exists(model_path: str):
    """若本機不存在模型檔，則自動下載"""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[*] 找不到模型檔案，正在下載: {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[+] 模型下載完成！")


def kabsch_algorithm(source_pts: np.ndarray, target_pts: np.ndarray):
    """
    使用 Kabsch Algorithm (SVD) 求解最佳剛體旋轉 R 與平移 t
    滿足: R @ source_pts + t ≈ target_pts
    """
    centroid_src = np.mean(source_pts, axis=0)
    centroid_tgt = np.mean(target_pts, axis=0)

    p = source_pts - centroid_src
    q = target_pts - centroid_tgt

    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 防止鏡像反射 (det(R) 必須為 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_tgt - R @ centroid_src
    return R, t


def load_video_classification_map(json_path: str) -> dict:
    """讀取分類 JSON，建立雙重查找表"""
    if not os.path.exists(json_path):
        return {}

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


# ---------------------------------------------------------------------------
# 2. 核心處理函式 (單一影片：提取 + 即時剛體對齊)
# ---------------------------------------------------------------------------
def process_and_align_video(video_path: str, landmarker, class_info: dict = None, custom_m0: np.ndarray = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [錯誤] 無法開啟影片: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    raw_frames = []
    current_sec = 0
    detected_count = 0

    # 1. 逐秒偵測臉部特徵
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

        raw_frames.append({
            "second": current_sec,
            "frame_index": target_frame_idx,
            "face_detected": landmarks_list is not None,
            "landmarks": landmarks_list,
        })
        current_sec += 1

    cap.release()

    # 2. 確定基準 M_0 人臉 (預設以第一張成功偵測的臉為 M_0)
    m0_landmarks = custom_m0
    m0_frame_sec = None

    if m0_landmarks is None:
        for f in raw_frames:
            if f["face_detected"] and f["landmarks"]:
                m0_landmarks = np.array([[lm["x"], lm["y"], lm["z"]] for lm in f["landmarks"]])
                m0_frame_sec = f["second"]
                break

    # 3. 記憶體內即時進行 Kabsch 剛體對齊
    aligned_frames_count = 0
    if m0_landmarks is not None:
        target_anchors = m0_landmarks[RIGID_ANCHOR_INDICES]

        for f in raw_frames:
            if not f["face_detected"] or not f["landmarks"]:
                f["rigid_alignment"] = {"aligned": False, "error": "No face detected"}
                continue

            current_pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in f["landmarks"]])
            source_anchors = current_pts[RIGID_ANCHOR_INDICES]

            R, t = kabsch_algorithm(source_anchors, target_anchors)
            aligned_pts = (R @ current_pts.T).T + t

            aligned_anchors = (R @ source_anchors.T).T + t
            rmse = float(np.sqrt(np.mean(np.sum((aligned_anchors - target_anchors) ** 2, axis=1))))

            for idx, lm in enumerate(f["landmarks"]):
                lm["aligned_x"] = float(aligned_pts[idx, 0])
                lm["aligned_y"] = float(aligned_pts[idx, 1])
                lm["aligned_z"] = float(aligned_pts[idx, 2])

            f["rigid_alignment"] = {
                "aligned": True,
                "rotation_matrix": R.tolist(),
                "translation_vector": t.tolist(),
                "anchor_rmse": round(rmse, 6),
            }
            aligned_frames_count += 1

    sampled_secs = len(raw_frames)
    face_presence_ratio = (detected_count / sampled_secs) if sampled_secs > 0 else 0.0

    return {
        "video_path": os.path.basename(video_path),
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
        "aligned_frames_count": aligned_frames_count,
        "m0_reference": {
            "source": "first_detected_frame" if custom_m0 is None else "custom_template",
            "reference_second": m0_frame_sec,
            "anchor_indices": RIGID_ANCHOR_INDICES,
        },
        "frames": raw_frames,
    }


# ---------------------------------------------------------------------------
# 3. 批次處理主程序
# ---------------------------------------------------------------------------
def batch_extract_and_align(
    root_dir=DEFAULT_REELS_DIR,
    model_path=MODEL_FILE_NAME,
    class_json_path=DEFAULT_VIDEO_CLASS_JSON,
    target_tier="all",
    overwrite=False,
):
    ensure_model_exists(model_path)

    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        return

    lookup_map = load_video_classification_map(class_json_path)

    # 搜尋所有 .mp4 檔案
    search_pattern = os.path.join(root_dir, "**", "*.mp4")
    all_videos = sorted(glob.glob(search_pattern, recursive=True))

    tasks = []
    for vp in all_videos:
        folder_dir, file_name = os.path.split(vp)
        account_id = os.path.basename(folder_dir)
        video_stem, _ = os.path.splitext(file_name)

        class_info = lookup_map.get((account_id, video_stem), {})
        tier = str(class_info.get("video_classification", "neutral")).lower()
        if target_tier != "all" and tier != target_tier.lower():
            continue

        tasks.append({"video_path": vp, "class_info": class_info, "tier": tier})

    print("=" * 75)
    print("【Face Mesh 提取與 Kabsch 剛體對齊整合引擎】")
    print(f"  - 影片根目錄:       {root_dir}")
    print(f"  - 分類檔案:         {class_json_path if os.path.exists(class_json_path) else '未提供 (全量處理)'}")
    print(f"  - 目標分組:         {target_tier.upper()}")
    print(f"  - 待處理影片數:     {len(tasks)}")
    print("=" * 75 + "\n")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for idx, task in enumerate(tasks, 1):
            vp = task["video_path"]
            folder_dir, file_name = os.path.split(vp)
            base_name, _ = os.path.splitext(file_name)
            output_json_path = os.path.join(folder_dir, f"{base_name}_face_aligned.json")
            rel_path = os.path.relpath(vp, root_dir)

            if os.path.exists(output_json_path) and not overwrite:
                print(f"[{idx}/{len(tasks)}] [已存在略過] {rel_path}")
                continue

            print(f"[{idx}/{len(tasks)}] 正在提取與對齊: {rel_path}...")
            result = process_and_align_video(vp, landmarker, class_info=task["class_info"])

            if result:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                ratio = result["face_presence_ratio"]
                aligned = result["aligned_frames_count"]
                print(f"  -> 完成！人臉覆蓋率: {ratio*100:.1f}%, 成功剛體對齊: {aligned} 幀 -> {os.path.basename(output_json_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 Face Landmarks 並直接執行 Kabsch 剛體對齊")
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_REELS_DIR), help="影片根目錄")
    parser.add_argument("--model", type=str, default=MODEL_FILE_NAME, help="模型路徑")
    parser.add_argument("--class_json", type=str, default=DEFAULT_VIDEO_CLASS_JSON, help="分類摘要 JSON")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "good", "neutral", "bad"], help="目標分組")
    parser.add_argument("--overwrite", action="store_true", help="強制重新提取並覆寫")
    args = parser.parse_args()

    batch_extract_and_align(
        root_dir=args.root_dir,
        model_path=args.model,
        class_json_path=args.class_json,
        target_tier=args.tier,
        overwrite=args.overwrite,
    )