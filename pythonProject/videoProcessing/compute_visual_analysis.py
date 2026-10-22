#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Color & Lighting Feature Extractor for Reels Analysis (Resumable & Safe Overwrite)
----------------------------------------------------------------------------------------
功能：
1. 將輸出的 *_color_lighting.json 直接儲存在 D:\\reels 中各影片所屬的對應子資料夾內 (比照 _qc.json, _face.json)。
2. 自動跳過已提取成功的影片 (支援 Ctrl+C 中斷後隨時接續執行)。
3. 提供 --overwrite 參數供使用者決定是否強制重新計算。
4. 支援 --max_videos 限制處理上限。
5. 整合 MediaPipe 468 點人臉定位與純物理光影/色彩特徵計算。
"""

import json
import argparse
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import VIDEO_PROC_DIR, DEFAULT_REELS_DIR

# ---------------------------------------------------------------------------
# 1. 動態定位專案根目錄與設定路徑
# ---------------------------------------------------------------------------
MODEL_PATH = str(VIDEO_PROC_DIR / "face_landmarker.task")

# 初始化 MediaPipe Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
LANDMARKER = vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# 2. 電腦視覺色彩與光影計算核心函式
# ---------------------------------------------------------------------------
def detect_face_bbox_mediapipe(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """回傳與 Haar 相同格式的 [(x, y, w, h)]，若無臉則回傳空 list"""
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = LANDMARKER.detect(mp_image)

    if not result.face_landmarks:
        return []

    landmarks = result.face_landmarks[0]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min, x_max = int(max(0, min(xs))), int(min(w, max(xs)))
    y_min, y_max = int(max(0, min(ys))), int(min(h, max(ys)))

    return [(x_min, y_min, x_max - x_min, y_max - y_min)]


def compute_colorfulness(image_bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) 經典色彩豐富度指標"""
    B = image_bgr[:, :, 0].astype(np.float32)
    G = image_bgr[:, :, 1].astype(np.float32)
    R = image_bgr[:, :, 2].astype(np.float32)

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, mean_rg = float(np.std(rg)), float(np.mean(rg))
    std_yb, mean_yb = float(np.std(yb)), float(np.mean(yb))

    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + (0.3 * mean_root))


def extract_facial_lighting_and_skin(
        image_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """分析人臉區域內的光影層次、RMS 對比度與膚色特徵"""
    h_img, w_img = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = face_bbox

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w_img, x_max), min(h_img, y_max)

    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return {
            "face_luminance_mean": 0.0,
            "face_rms_contrast": 0.0,
            "face_dynamic_range": 0.0,
            "skin_exposure": 0.0,
            "skin_warmth_cbcr": 0.0,
        }

    face_crop = image_bgr[y_min:y_max, x_min:x_max]

    # 1. 灰階光影與 RMS 對比度
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lum_mean = float(np.mean(gray_face))
    rms_contrast = float(np.std(gray_face) / 255.0)

    # 2. 面部動態範圍 (95 百分位亮度 - 5 百分位亮度)
    p95 = np.percentile(gray_face, 95)
    p05 = np.percentile(gray_face, 5)
    dynamic_range = float((p95 - p05) / 255.0)

    # 3. YCbCr 膚色通透度與血色感
    ycrcb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    y_chan = ycrcb_face[:, :, 0]
    cr_chan = ycrcb_face[:, :, 1]
    cb_chan = ycrcb_face[:, :, 2]

    skin_mask = (y_chan > 60) & (cr_chan >= 130) & (cr_chan <= 175) & (cb_chan >= 75) & (cb_chan <= 130)

    if np.sum(skin_mask) > 30:
        skin_exp = float(np.mean(y_chan[skin_mask]) / 255.0)
        skin_warmth = float(np.mean(cr_chan[skin_mask].astype(np.float32) - cb_chan[skin_mask].astype(np.float32)))
    else:
        skin_exp = float(lum_mean / 255.0)
        skin_warmth = 0.0

    return {
        "face_luminance_mean": lum_mean,
        "face_rms_contrast": rms_contrast,
        "face_dynamic_range": dynamic_range,
        "skin_exposure": skin_exp,
        "skin_warmth_cbcr": skin_warmth,
    }


def compute_figure_ground_separation(
        image_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """計算人物前景與背景的明暗與色度反差"""
    h_img, w_img = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = face_bbox
    bw = x_max - x_min
    bh = y_max - y_min

    person_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    person_mask[max(0, y_min):min(h_img, y_max), max(0, x_min):min(w_img, x_max)] = 1

    body_ymin = y_max
    body_ymax = min(h_img, int(y_max + 2.0 * bh))
    body_xmin = max(0, int(x_min - 0.4 * bw))
    body_xmax = min(w_img, int(x_max + 0.4 * bw))

    person_mask[body_ymin:body_ymax, body_xmin:body_xmax] = 1
    bg_mask = (person_mask == 0)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fg_lum = np.mean(gray[person_mask == 1]) if np.sum(person_mask) > 0 else 128.0
    bg_lum = np.mean(gray[bg_mask]) if np.sum(bg_mask) > 0 else 128.0

    lum_contrast = float(np.abs(fg_lum - bg_lum) / (fg_lum + bg_lum + 1e-5))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    fg_sat = np.mean(hsv[:, :, 1][person_mask == 1]) if np.sum(person_mask) > 0 else 0.0
    bg_sat = np.mean(hsv[:, :, 1][bg_mask]) if np.sum(bg_mask) > 0 else 0.0
    sat_contrast = float(np.abs(fg_sat - bg_sat) / 255.0)

    return {
        "fg_luminance": float(fg_lum),
        "bg_luminance": float(bg_lum),
        "figure_ground_lum_contrast": lum_contrast,
        "figure_ground_sat_contrast": sat_contrast,
    }


# ---------------------------------------------------------------------------
# 3. 影片秒級取樣流水線
# ---------------------------------------------------------------------------
def process_single_video(video_path: Path) -> Optional[Dict[str, Any]]:
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    duration_sec = total_frames / fps
    step_frames = max(1, int(round(fps)))  # 每秒採樣 1 幀

    time_series: List[Dict[str, float]] = []
    current_frame_idx = 0
    last_known_bbox = None

    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        sec_stamp = round(current_frame_idx / fps, 2)
        h, w = frame.shape[:2]

        faces = detect_face_bbox_mediapipe(frame)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            bbox = (fx, fy, fx + fw, fy + fh)
            last_known_bbox = bbox
        elif last_known_bbox is not None:
            bbox = last_known_bbox
        else:
            bbox = (int(w * 0.3), int(h * 0.2), int(w * 0.7), int(h * 0.6))

        colorfulness = compute_colorfulness(frame)
        face_stats = extract_facial_lighting_and_skin(frame, bbox)
        separation_stats = compute_figure_ground_separation(frame, bbox)

        frame_metrics = {
            "second": sec_stamp,
            "colorfulness": colorfulness,
            **face_stats,
            **separation_stats,
        }
        time_series.append(frame_metrics)
        current_frame_idx += step_frames

    cap.release()

    if not time_series:
        return None

    def aggregate_field(keys: List[str], data: List[Dict[str, float]]) -> Dict[str, float]:
        res = {}
        for k in keys:
            vals = [d[k] for d in data if k in d]
            if vals:
                res[f"{k}_mean"] = float(np.mean(vals))
                res[f"{k}_std"] = float(np.std(vals))
            else:
                res[f"{k}_mean"] = 0.0
                res[f"{k}_std"] = 0.0
        return res

    metric_keys = [
        "colorfulness", "face_luminance_mean", "face_rms_contrast",
        "face_dynamic_range", "skin_exposure", "skin_warmth_cbcr",
        "figure_ground_lum_contrast", "figure_ground_sat_contrast"
    ]

    whole_video_agg = aggregate_field(metric_keys, time_series)

    hook_data = [d for d in time_series if d["second"] <= 3.0]
    if not hook_data:
        hook_data = time_series[:min(3, len(time_series))]
    hook_agg = {f"hook_{k}": v for k, v in aggregate_field(metric_keys, hook_data).items()}

    return {
        "video_name": video_path.name,
        "duration_sec": duration_sec,
        "total_sampled_seconds": len(time_series),
        "whole_video_summary": whole_video_agg,
        "hook_window_summary": hook_agg,
        "second_level_time_series": time_series,
    }


# ---------------------------------------------------------------------------
# 4. 主執行入口 (儲存至各影片所屬的子目錄，具備斷點續跑與安全覆蓋)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="提取短影音色彩豐富度與光影對比度特徵")
    parser.add_argument("--video", type=str, default="", help="單支影片路徑 (測試用)")
    parser.add_argument("--reels_dir", type=str, default=str(DEFAULT_REELS_DIR), help="影片存放根目錄")
    parser.add_argument("--max_videos", type=int, default=0, help="處理影片上限數量 (0 代表全部)")
    parser.add_argument("--overwrite", action="store_true", help="強制重新提取並覆蓋已存在的 JSON 檔案")
    args = parser.parse_args()

    # 單支影片模式
    if args.video:
        v_path = Path(args.video)
        # 直接存放在影片同一層子資料夾中 (與 _qc.json, _face.json 同級)
        out_file = v_path.parent / f"{v_path.stem}_color_lighting.json"
        if out_file.exists() and not args.overwrite:
            print(f"[*] 檔案已存在，跳過: {out_file.name} (若需重新計算請加入 --overwrite)")
            return

        print(f"[*] 正在處理單支影片: {v_path.name}...")
        result = process_single_video(v_path)
        if result:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[+] 特徵提取成功！儲存至: {out_file}")
        return

    # 批次模式
    reels_root = Path(args.reels_dir)
    if not reels_root.exists():
        print(f"[-] 找不到影片目錄: {reels_root}")
        return

    # 遞迴搜尋子目錄下的所有 mp4 檔案
    video_files = sorted(list(reels_root.rglob("*.mp4")))
    if not video_files:
        video_files = sorted(list(reels_root.glob("*.mp4")))

    total_found = len(video_files)
    print(f"[*] 掃描到總共 {total_found} 支影片。")

    # 檢查已完成進度 (斷點續跑機制：直接檢查各影片子資料夾內是否存在 *_color_lighting.json)
    tasks_to_run = []
    skipped_cnt = 0
    for vf in video_files:
        target_json = vf.parent / f"{vf.stem}_color_lighting.json"
        if target_json.exists() and target_json.stat().st_size > 100 and not args.overwrite:
            skipped_cnt += 1
        else:
            tasks_to_run.append(vf)

    print(f"[*] 進度狀態：已完成 {skipped_cnt} 支，待處理 {len(tasks_to_run)} 支。")

    if args.max_videos > 0:
        tasks_to_run = tasks_to_run[:args.max_videos]
        print(f"[*] 依照 --max_videos 設定，本次只處理前 {len(tasks_to_run)} 支待跑影片。")

    if not tasks_to_run:
        print("[+] 所有影片皆已處理完畢！無須重新執行。")
        return

    success_cnt = 0
    for idx, vf in enumerate(tasks_to_run, 1):
        print(f"[{idx}/{len(tasks_to_run)}] 正在分析: {vf.name} ...", end="", flush=True)
        res = process_single_video(vf)
        if res:
            # 儲存在該影片所在的子資料夾內
            out_file = vf.parent / f"{vf.stem}_color_lighting.json"
            tmp_file = out_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            tmp_file.replace(out_file)
            print(" [OK]")
            success_cnt += 1
        else:
            print(" [FAILED]")

    print("=" * 60)
    print(f"[+] 本次執行完成！成功產出 {success_cnt} 支影片特徵。")
    print(f"[+] 累計完成進度: {skipped_cnt + success_cnt}/{total_found}")
    print(f"[+] 檔案皆已儲存在各影片所屬的對應子目錄中 (*_color_lighting.json)。")
    print("=" * 60)


if __name__ == "__main__":
    main()