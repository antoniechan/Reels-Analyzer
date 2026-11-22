#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified QC Deformation Audit, Diagnosis, Rescue & Pruning Manager
----------------------------------------------------------------
整合功能：
1. 全盤掃描與健康度稽核 (Audit)：檢查指定目錄下所有影片的附屬 `*_qc.json` 完整性、數值合法性與有效過渡幀比例。
2. 根因診斷 (Diagnose)：針對缺失或異常的影片，自動檢查對應的 `*_face_aligned.json` / `*_face.json` 人臉偵測覆蓋率與畫面黑邊狀態，精準歸納異常原因。
3. 互動式決策與自動化執行 (Interactive Repair / Rescue / Prune)：
   診斷完成後，用戶可透過互動式選單（或命令列參數 `--action`）自由選擇：
   - [1] rescue : 啟動「自動去黑邊放大 (Auto-Crop Zoom) + 低門檻小臉偵測 + 容錯間隔 (max_gap)」救援重算異常影片，並自動重跑驗證。
   - [2] recalc : 刪除異常影片的舊快取（`*_qc.json` 與 `*_face_aligned.json`），調用標準計算管線重算。
   - [3] prune  : 將仍無法偵測到有效人臉的純圖卡/無露臉雜訊影片移入隔離區 (`_quarantined_no_face`)，並同步更新 `reelsData` JSON 與重新連號 (`video_001`, `video_002`...)。
   - [0] exit   : 僅匯出稽核與診斷報告，不更動任何檔案。
"""

import os
import json
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    VIDEO_PROC_DIR,
    NEW_REELS_DIR,
    REELS_DATA_DIR,
)
from compute_face_landmark import (
    MODEL_FILE_NAME,
    RIGID_ANCHOR_INDICES,
    ensure_model_exists,
    kabsch_algorithm,
    load_video_classification_map,
    DEFAULT_VIDEO_CLASS_JSON,
)
from compute_qc_deformation import (
    get_canonical_triangles,
    compute_beltrami_for_triangle,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


# ==============================================================================
# 1. 單一 _qc.json 健康度校驗
# ==============================================================================
def check_single_qc_json(qc_path: str, zero_eps: float = 1e-8) -> tuple[bool, list, dict]:
    issues = []

    if os.path.getsize(qc_path) == 0:
        return False, ["檔案大小為 0 位元組 (空檔案)"], {}

    try:
        with open(qc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"JSON 格式損毀無法解析: {e}"], {}

    if not isinstance(data, dict):
        return False, ["根節點非 JSON 物件 (Dictionary)"], {}

    transitions = data.get("consecutive_qc_metrics", [])
    if not transitions:
        return False, ["缺少 consecutive_qc_metrics 欄位或為空陣列"], {}

    total_transitions = len(transitions)
    valid_transitions_count = 0
    all_mu_values = []
    all_dil_values = []
    has_facet_data = False

    for t_idx, trans in enumerate(transitions):
        if not isinstance(trans, dict):
            issues.append(f"第 {t_idx} 幀過渡不是物件格式")
            continue

        if trans.get("valid_transition", False):
            valid_transitions_count += 1

        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            continue

        has_facet_data = True
        for fm in facet_metrics:
            mu = fm.get("abs_mu")
            dil = fm.get("dilation")
            verts = fm.get("vertices", [])

            if len(verts) != 3:
                issues.append(f"第 {t_idx} 幀存在非三角形面片 (vertices != 3)")

            if mu is not None:
                if np.isnan(mu) or np.isinf(mu):
                    issues.append(f"第 {t_idx} 幀發現 NaN 或 Inf abs_mu")
                else:
                    all_mu_values.append(float(mu))

            if dil is not None:
                if np.isnan(dil) or np.isinf(dil):
                    issues.append(f"第 {t_idx} 幀發現 NaN 或 Inf dilation")
                else:
                    all_dil_values.append(float(dil))

    if valid_transitions_count == 0:
        issues.append("整部影片完全沒有任何有效過渡幀 (valid_transition 全為 False)")

    if not has_facet_data:
        issues.append("整部影片沒有任何面片形變記錄 (facet_metrics 全部為空)")

    if not all_mu_values:
        issues.append("未擷取到任何有效的 abs_mu 數值")
    else:
        mu_arr = np.array(all_mu_values)
        if np.mean(mu_arr > zero_eps) == 0:
            issues.append("所有 abs_mu 數值皆為 0 (臉部完全未產生任何動態形變)")
        if np.any(mu_arr < 0):
            issues.append("存在小於 0 的 abs_mu (違反幾何定義)")

    if all_dil_values:
        dil_arr = np.array(all_dil_values)
        if np.any(dil_arr < 0.99):
            issues.append("存在小於 1.0 的 dilation (違反準共形膨脹定義)")

    summary = {
        "total_transitions": total_transitions,
        "valid_transitions": valid_transitions_count,
        "valid_transition_ratio": round(valid_transitions_count / max(1, total_transitions), 4),
        "mean_mu": round(float(np.mean(all_mu_values)), 6) if all_mu_values else 0.0,
        "max_mu": round(float(np.max(all_mu_values)), 6) if all_mu_values else 0.0,
        "mean_dilation": round(float(np.mean(all_dil_values)), 4) if all_dil_values else 0.0,
    }

    return len(issues) == 0, list(set(issues)), summary


# ==============================================================================
# 2. 異常項目深度診斷 (Diagnose Root Cause)
# ==============================================================================
def diagnose_abnormal_row(row: dict) -> dict:
    vid_path = Path(str(row["video_path"]))
    qc_path = Path(str(row["qc_path"]))
    aligned_path = vid_path.parent / f"{vid_path.stem}_face_aligned.json"
    legacy_face_path = vid_path.parent / f"{vid_path.stem}_face.json"

    diag = {
        "resolution": "N/A",
        "detected_face_frames": 0,
        "total_sampled_frames": 0,
        "face_detection_ratio": 0.0,
        "has_letterbox": False,
        "probable_cause": "",
    }

    if not vid_path.exists():
        diag["probable_cause"] = "實體 MP4 檔案遺失"
        return diag

    # 檢查解析度與是否有上下/左右黑邊
    cap = cv2.VideoCapture(str(vid_path))
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        diag["resolution"] = f"{w}x{h}"
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
            ret, sample_frame = cap.read()
            if ret and sample_frame is not None:
                x0, y0, x1, y1 = detect_active_content_box(sample_frame)
                active_area = (x1 - x0) * (y1 - y0)
                if active_area < 0.75 * (w * h):
                    diag["has_letterbox"] = True
    cap.release()

    # 檢查對應的 _face_aligned.json 或 _face.json
    face_file = aligned_path if aligned_path.exists() else (legacy_face_path if legacy_face_path.exists() else None)
    if face_file is not None:
        try:
            with open(face_file, "r", encoding="utf-8") as f:
                f_data = json.load(f)
            frames = f_data.get("frames") or f_data.get("frame_landmarks") or []
            diag["total_sampled_frames"] = len(frames)
            valid_f = sum(
                1 for fr in frames
                if isinstance(fr, dict) and (fr.get("face_detected") or fr.get("landmarks") or fr.get("has_face"))
            )
            diag["detected_face_frames"] = valid_f
            if len(frames) > 0:
                diag["face_detection_ratio"] = round(valid_f / len(frames), 4)
        except Exception as e:
            diag["probable_cause"] = f"特徵點 JSON 讀取失敗: {e}"
            return diag

    if not row["qc_exists"]:
        diag["probable_cause"] = "尚未計算 _qc.json"
    elif diag["detected_face_frames"] == 0:
        if diag["has_letterbox"]:
            diag["probable_cause"] = "橫屏黑邊 (Letterbox) 導致人臉過小漏檢，或全片無正臉"
        else:
            diag["probable_cause"] = "全片未偵測到正臉 (小臉信心度不足 / 純圖卡 / 食物藥材 / 遮擋)"
    elif diag["detected_face_frames"] == 1:
        diag["probable_cause"] = "全片僅 1 幀閃現人臉，無法構成連續兩幀過渡"
    else:
        diag["probable_cause"] = f"偵測到 {diag['detected_face_frames']} 幀人臉，但因穿插 B-roll 導致相鄰秒中斷"

    return diag


# ==============================================================================
# 3. 全盤稽核與診斷引擎 (Audit + Diagnose)
# ==============================================================================
def run_full_audit_and_diagnosis(reels_dir: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 90)
    print("🔍 【步驟 1：全盤 QC Deformation 完整性稽核與根因診斷】")
    print(f"  - 掃描目標目錄: {os.path.abspath(reels_dir)}")
    print("=" * 90)

    if not os.path.exists(reels_dir):
        raise FileNotFoundError(f"目錄不存在: {reels_dir}")

    all_video_files = []
    for root, dirs, files in os.walk(reels_dir):
        # 跳過隔離資料夾
        dirs[:] = [d for d in dirs if not d.startswith("_quarantined")]
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                all_video_files.append(os.path.join(root, f))

    all_video_files = sorted(all_video_files)
    total_videos = len(all_video_files)
    print(f"[*] 共檢索到 {total_videos} 支有效影片檔案，正在進行逐片校驗...\n")

    results = []
    missing_qc_count = 0
    corrupt_qc_count = 0
    healthy_qc_count = 0

    for idx, vid_path in enumerate(all_video_files, start=1):
        folder_dir, vid_name = os.path.split(vid_path)
        vid_stem = os.path.splitext(vid_name)[0]
        account_id = os.path.basename(folder_dir)
        qc_candidate = os.path.join(folder_dir, f"{vid_stem}_qc.json")

        status_rec = {
            "account_id": account_id,
            "video_stem": vid_stem,
            "video_path": vid_path,
            "qc_path": qc_candidate,
            "qc_exists": False,
            "is_healthy": False,
            "issue_details": "",
            "total_transitions": 0,
            "valid_transitions": 0,
            "valid_transition_ratio": 0.0,
            "mean_mu": 0.0,
            "max_mu": 0.0,
            "mean_dilation": 0.0,
        }

        if not os.path.exists(qc_candidate):
            missing_qc_count += 1
            status_rec["issue_details"] = "缺少 _qc.json 檔案"
        else:
            status_rec["qc_exists"] = True
            is_healthy, issues, summary = check_single_qc_json(qc_candidate)
            status_rec.update(summary)
            if is_healthy:
                healthy_qc_count += 1
                status_rec["is_healthy"] = True
                status_rec["issue_details"] = "正常"
            else:
                corrupt_qc_count += 1
                status_rec["issue_details"] = " | ".join(issues)

        # 若不健康，立即執行深度診斷
        if not status_rec["is_healthy"]:
            diag_info = diagnose_abnormal_row(status_rec)
            status_rec.update(diag_info)
        else:
            status_rec.update({
                "resolution": "",
                "detected_face_frames": -1,
                "total_sampled_frames": -1,
                "face_detection_ratio": -1.0,
                "has_letterbox": False,
                "probable_cause": "N/A",
            })

        results.append(status_rec)

        if idx % 500 == 0 or idx == total_videos:
            print(f"  進度: [{idx}/{total_videos}] (正常: {healthy_qc_count}, 異常: {corrupt_qc_count}, 缺失: {missing_qc_count})")

    df_all = pd.DataFrame(results)
    df_abnormal = df_all[~df_all["is_healthy"]].copy().reset_index(drop=True)

    print("\n" + "=" * 90)
    print("📊 【QC 稽核與診斷總結】")
    print("=" * 90)
    print(f"  • 掃描影片總數:                {total_videos:>5} 支")
    print(f"  • ✅ 健康且完整 _qc.json:      {healthy_qc_count:>5} 支 ({healthy_qc_count / max(1, total_videos) * 100:.2f}%)")
    print(f"  • ❌ 數據異常 _qc.json:        {corrupt_qc_count:>5} 支 ({corrupt_qc_count / max(1, total_videos) * 100:.2f}%)")
    print(f"  • ⚠️ 完全缺少 _qc.json:        {missing_qc_count:>5} 支 ({missing_qc_count / max(1, total_videos) * 100:.2f}%)")
    print("=" * 90)

    full_csv = os.path.join(output_dir, "qc_json_validation_full_report.csv")
    df_all.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"[*] 完整檢查報告已儲存至: {os.path.abspath(full_csv)}")

    abnormal_csv = os.path.join(output_dir, "qc_json_missing_or_corrupt_list.csv")
    if not df_abnormal.empty:
        df_abnormal.to_csv(abnormal_csv, index=False, encoding="utf-8-sig")
        print(f"[*] 異常與診斷清單已儲存至: {os.path.abspath(abnormal_csv)} (共 {len(df_abnormal)} 支)\n")
        print("📋 異常影片診斷明細：")
        print("-" * 90)
        for i, r in df_abnormal.iterrows():
            lb_tag = "[黑邊橫屏]" if r.get("has_letterbox") else "[標準畫面]"
            print(
                f"  {i+1:>2}. [{r['account_id']:<16}] {r['video_stem']} | "
                f"{lb_tag} ({r['resolution']}) | "
                f"人臉幀: {r['detected_face_frames']}/{r['total_sampled_frames']} | "
                f"診斷: {r['probable_cause']}"
            )
        print("-" * 90)
    else:
        if os.path.exists(abnormal_csv):
            os.remove(abnormal_csv)
        print("\n🎉 恭喜！全庫所有影片的 _qc.json 均 100% 健康無缺失！")

    return df_all, df_abnormal


# ==============================================================================
# 4. 救援引擎 (Auto-Crop Zoom + 低門檻小臉偵測 + 間隔容錯 QC)
# ==============================================================================
def detect_active_content_box(frame_bgr: np.ndarray, black_thresh: int = 12) -> tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)

    valid_rows = np.where(row_means > black_thresh)[0]
    valid_cols = np.where(col_means > black_thresh)[0]

    if len(valid_rows) < int(h * 0.15) or len(valid_cols) < int(w * 0.15):
        return 0, 0, w, h

    y0 = max(0, int(valid_rows[0]) - 8)
    y1 = min(h, int(valid_rows[-1]) + 9)
    x0 = max(0, int(valid_cols[0]) - 8)
    x1 = min(w, int(valid_cols[-1]) + 9)
    return x0, y0, x1, y1


def extract_and_align_single_video(
    video_path: str,
    landmarker,
    class_info: dict = None,
    use_autocrop: bool = True,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
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

    while True:
        target_frame_idx = int(round(current_sec * fps))
        if target_frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        used_crop = False
        x0, y0, x1, y1 = 0, 0, video_width, video_height
        crop_w, crop_h = float(video_width), float(video_height)

        if use_autocrop:
            x0, y0, x1, y1 = detect_active_content_box(frame)
            cropped_bgr = frame[y0:y1, x0:x1]
            crop_w = float(x1 - x0)
            crop_h = float(y1 - y0)

            if crop_w < 720:
                scale = 720.0 / crop_w
                detect_bgr = cv2.resize(
                    cropped_bgr,
                    (int(round(crop_w * scale)), int(round(crop_h * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                detect_bgr = cropped_bgr

            rgb_frame = cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB)
            detection_result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
            used_crop = True

            # 若裁切框未偵測到且確實有裁切過，退回全幅畫面再測一次
            if not detection_result.face_landmarks and (x0 > 0 or y0 > 0 or x1 < video_width or y1 < video_height):
                full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection_result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=full_rgb))
                used_crop = False
        else:
            full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=full_rgb))

        landmarks_list = None
        if detection_result.face_landmarks:
            detected_count += 1
            first_face = detection_result.face_landmarks[0]
            landmarks_list = []
            for idx, lm in enumerate(first_face):
                if used_crop:
                    orig_x = (x0 + lm.x * crop_w) / float(video_width)
                    orig_y = (y0 + lm.y * crop_h) / float(video_height)
                    orig_z = lm.z * (crop_w / float(video_width))
                else:
                    orig_x, orig_y, orig_z = float(lm.x), float(lm.y), float(lm.z)

                landmarks_list.append({
                    "id": idx,
                    "x": float(orig_x),
                    "y": float(orig_y),
                    "z": float(orig_z),
                    "px": int(round(orig_x * video_width)),
                    "py": int(round(orig_y * video_height)),
                })

        raw_frames.append({
            "second": current_sec,
            "frame_index": target_frame_idx,
            "face_detected": landmarks_list is not None,
            "landmarks": landmarks_list,
        })
        current_sec += 1

    cap.release()

    m0_landmarks = None
    m0_frame_sec = None
    for f in raw_frames:
        if f["face_detected"] and f["landmarks"]:
            m0_landmarks = np.array([[lm["x"], lm["y"], lm["z"]] for lm in f["landmarks"]])
            m0_frame_sec = f["second"]
            break

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
            "shares": class_info.get("shares"),
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
            "source": "first_detected_frame_autocrop" if use_autocrop else "first_detected_frame",
            "reference_second": m0_frame_sec,
            "anchor_indices": RIGID_ANCHOR_INDICES,
        },
        "frames": raw_frames,
    }


def compute_qc_from_aligned_data(
    aligned_data: dict,
    aligned_json_path: str,
    canonical_triangles: np.ndarray,
    max_gap_sec: int = 2,
) -> dict:
    frames = aligned_data.get("frames", [])
    num_canonical_triangles = len(canonical_triangles)
    transitions_results = []

    for i in range(len(frames) - 1):
        f_curr = frames[i]
        sec_curr = f_curr.get("second", i)
        valid_curr = f_curr.get("face_detected", False) and bool(f_curr.get("landmarks"))

        f_next = None
        if valid_curr:
            for step in range(1, max_gap_sec + 1):
                if i + step < len(frames):
                    cand = frames[i + step]
                    if cand.get("face_detected", False) and bool(cand.get("landmarks")):
                        f_next = cand
                        break

        if not (valid_curr and f_next is not None):
            sec_next_default = frames[i + 1].get("second", i + 1)
            transitions_results.append({
                "from_second": sec_curr,
                "to_second": sec_next_default,
                "valid_transition": False,
                "teichmuller_distance": 0.0,
                "mean_dilation": 0.0,
                "max_dilation": 0.0,
                "mean_abs_mu": 0.0,
                "max_abs_mu": 0.0,
                "num_triangles": num_canonical_triangles,
                "facet_metrics": [],
            })
            continue

        sec_next = f_next.get("second", i + 1)
        lms_curr = f_curr["landmarks"]
        lms_next = f_next["landmarks"]

        z_pts = np.array([complex(lm.get("aligned_x", lm["x"]), lm.get("aligned_y", lm["y"])) for lm in lms_curr[:468]])
        w_pts = np.array([complex(lm.get("aligned_x", lm["x"]), lm.get("aligned_y", lm["y"])) for lm in lms_next[:468]])

        mu_list, dilation_list, facet_data = [], [], []
        for tri_idx, tri in enumerate(canonical_triangles):
            mu = compute_beltrami_for_triangle(z_pts[tri], w_pts[tri])
            abs_mu = float(np.abs(mu))
            abs_mu_clipped = min(abs_mu, 0.999)
            dilation = (1.0 + abs_mu_clipped) / (1.0 - abs_mu_clipped)

            mu_list.append(abs_mu_clipped)
            dilation_list.append(dilation)
            facet_data.append({
                "triangle_id": int(tri_idx),
                "vertices": tri.tolist(),
                "mu_real": float(mu.real),
                "mu_imag": float(mu.imag),
                "abs_mu": abs_mu,
                "dilation": float(dilation),
                "arg_mu": float(np.angle(mu)),
            })

        mu_arr = np.array(mu_list)
        dil_arr = np.array(dilation_list)
        max_mu = float(np.max(mu_arr))
        mean_mu = float(np.mean(mu_arr))
        max_dil = float(np.max(dil_arr))
        mean_dil = float(np.mean(dil_arr))
        teichmuller_dist = float(0.5 * np.log((1.0 + max_mu) / (1.0 - max_mu)))

        transitions_results.append({
            "from_second": sec_curr,
            "to_second": sec_next,
            "valid_transition": True,
            "teichmuller_distance": teichmuller_dist,
            "mean_dilation": mean_dil,
            "max_dilation": max_dil,
            "mean_abs_mu": mean_mu,
            "max_abs_mu": max_mu,
            "num_triangles": num_canonical_triangles,
            "facet_metrics": facet_data,
        })

    return {
        "source_file": os.path.basename(aligned_json_path),
        "topology": {
            "type": "mediapipe_canonical_facemesh",
            "total_canonical_triangles": num_canonical_triangles,
        },
        "total_transitions": len(transitions_results),
        "consecutive_qc_metrics": transitions_results,
    }


def execute_recompute_batch(
    df_abnormal: pd.DataFrame,
    class_json_path: str,
    use_autocrop: bool = True,
    min_conf: float = 0.25,
    max_gap: int = 2,
):
    mode_label = "救援模式 (Auto-Crop去黑邊 + 低信心度 + 間隔容錯)" if use_autocrop else "標準重算模式 (清理舊快取並重新計算)"
    print("\n" + "=" * 90)
    print(f"🚑 【開始執行 {mode_label} — 共 {len(df_abnormal)} 支影片】")
    print("=" * 90)

    ensure_model_exists(MODEL_FILE_NAME)
    canonical_triangles = get_canonical_triangles()
    lookup_map = load_video_classification_map(class_json_path)

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE_NAME)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf,
    )

    rescued_ok = 0
    still_failed = 0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for idx, row in df_abnormal.iterrows():
            vid_path = str(row["video_path"])
            qc_path = str(row["qc_path"])
            acc_id = str(row["account_id"]).strip()
            vid_stem = str(row["video_stem"]).strip()
            aligned_path = os.path.join(os.path.dirname(vid_path), f"{vid_stem}_face_aligned.json")

            if not os.path.exists(vid_path):
                print(f"[{idx+1}/{len(df_abnormal)}] [缺失原片] {vid_path}")
                still_failed += 1
                continue

            print(f"[{idx+1}/{len(df_abnormal)}] 正在重算: {acc_id}/{vid_stem}.mp4 ...")
            class_info = lookup_map.get((acc_id, vid_stem), {})

            aligned_data = extract_and_align_single_video(
                vid_path, landmarker, class_info=class_info, use_autocrop=use_autocrop
            )
            if not aligned_data:
                print("    -> [失敗] 無法讀取視訊軌")
                still_failed += 1
                continue

            with open(aligned_path, "w", encoding="utf-8") as f:
                json.dump(aligned_data, f, ensure_ascii=False, indent=2)

            qc_data = compute_qc_from_aligned_data(
                aligned_data, aligned_path, canonical_triangles, max_gap_sec=max_gap
            )
            with open(qc_path, "w", encoding="utf-8") as f:
                json.dump(qc_data, f, ensure_ascii=False, indent=2)

            valid_tr = sum(1 for tr in qc_data["consecutive_qc_metrics"] if tr["valid_transition"])
            face_secs = aligned_data["face_detected_seconds"]
            total_secs = aligned_data["sampled_seconds"]

            if valid_tr > 0:
                rescued_ok += 1
                print(f"    -> ✅ 修復成功！偵測到人臉: {face_secs}/{total_secs} 秒 | 有效 QC 過渡幀: {valid_tr} 組")
            else:
                still_failed += 1
                print(f"    -> ⚠️ 仍無有效 QC 過渡 (人臉幀: {face_secs}/{total_secs} 秒)")

    print("\n" + "=" * 90)
    print(f"🎉 重算作業完成！成功修復: {rescued_ok}/{len(df_abnormal)} 支 | 仍異常: {still_failed}/{len(df_abnormal)} 支")
    print("=" * 90)


# ==============================================================================
# 5. 隔離與同步順號引擎 (Prune & Re-index)
# ==============================================================================
def execute_prune_and_sync(
    df_abnormal: pd.DataFrame,
    reels_dir: str,
    reels_data_dir: str,
):
    reels_root = Path(reels_dir)
    data_root = Path(reels_data_dir)
    quarantine_root = reels_root / "_quarantined_no_face"
    quarantine_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"🧹 【開始隔離無講者露臉雜訊影片並同步重新順號 — 共 {len(df_abnormal)} 支】")
    print(f"    隔離目錄: {quarantine_root}")
    print("=" * 90)

    affected_accounts = set()

    for _, row in df_abnormal.iterrows():
        acc = str(row["account_id"]).strip()
        stem = str(row["video_stem"]).strip()
        acc_dir = reels_root / acc
        if not acc_dir.exists():
            continue

        affected_accounts.add(acc)
        target_q_dir = quarantine_root / acc
        target_q_dir.mkdir(parents=True, exist_ok=True)

        for f in acc_dir.glob(f"{stem}*"):
            if f.is_file():
                shutil.move(str(f), str(target_q_dir / f.name))
                print(f"  [隔離] {acc}/{f.name} -> _quarantined_no_face/{acc}/")

    for acc in sorted(affected_accounts):
        acc_dir = reels_root / acc
        json_path = data_root / f"{acc}.json"

        remaining_mp4s = sorted(acc_dir.glob("video_*.mp4"))
        old_to_new_stem = {}
        for new_idx, mp4_f in enumerate(remaining_mp4s, start=1):
            old_stem = mp4_f.stem
            new_stem = f"video_{new_idx:03d}"
            old_to_new_stem[old_stem] = new_stem

        for old_stem, new_stem in old_to_new_stem.items():
            if old_stem == new_stem:
                continue
            for f in acc_dir.glob(f"{old_stem}*"):
                suffix_part = f.name[len(old_stem):]
                f.rename(acc_dir / f"__tmp_{new_stem}{suffix_part}")

        for f in acc_dir.glob("__tmp_video_*"):
            f.rename(acc_dir / f.name.replace("__tmp_", "", 1))

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            if isinstance(records, list):
                updated_records = []
                for item in records:
                    if not isinstance(item, dict):
                        continue
                    v_id = str(item.get("video_id", "")).strip()
                    if v_id in old_to_new_stem:
                        new_s = old_to_new_stem[v_id]
                        item["video_id"] = new_s
                        if "local_video_file" in item:
                            item["local_video_file"] = f"{new_s}.mp4"
                        if "local_audio_file" in item:
                            item["local_audio_file"] = f"{new_s}.mp3"
                        updated_records.append(item)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(updated_records, f, ensure_ascii=False, indent=2)
                print(f"  [同步更新 JSON] {acc}.json: 保留 {len(updated_records)} 筆有效紀錄並完成重新順號。")

    print("\n[+] 隔離與同步順號完成！建議隨後重跑 video_residual_classification-v2.py 更新分類總表。")


# ==============================================================================
# 6. 主程序入口 (支援互動式選單 + CLI 自動化參數)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="QC Deformation 一站式稽核、根因診斷、小臉救援與隔離修復管理器"
    )
    parser.add_argument("--dir", type=str, default=str(NEW_REELS_DIR), help=f"Reels 影片根目錄 (預設: {NEW_REELS_DIR})")
    parser.add_argument("--data_dir", type=str, default=str(REELS_DATA_DIR), help=f"reelsData JSON 目錄 (預設: {REELS_DATA_DIR})")
    parser.add_argument("--out_dir", type=str, default=str(VIDEO_PROC_DIR), help=f"稽核報告輸出目錄 (預設: {VIDEO_PROC_DIR})")
    parser.add_argument("--class_json", type=str, default=DEFAULT_VIDEO_CLASS_JSON, help="影片分類 JSON 路徑")
    parser.add_argument(
        "--action",
        type=str,
        default="interactive",
        choices=["interactive", "audit_only", "rescue", "recalc", "prune"],
        help="執行模式: interactive (診斷後互動選擇), audit_only (僅稽核診斷), rescue (去黑邊小臉救援), recalc (標準重算), prune (隔離無臉雜訊)",
    )
    parser.add_argument("--min_conf", type=float, default=0.25, help="救援模式 MediaPipe 最低信心度 (預設: 0.25)")
    parser.add_argument("--max_gap", type=int, default=2, help="B-roll 間隔容錯最大秒數 (預設: 2)")
    args = parser.parse_args()

    # 1. 執行全盤稽核與診斷
    _, df_abnormal = run_full_audit_and_diagnosis(args.dir, args.out_dir)

    if df_abnormal.empty:
        return

    action = args.action

    # 2. 若為互動模式，顯示選單讓用戶選擇維修或重試策略
    while True:
        if action == "interactive":
            print("\n" + "=" * 90)
            print(f"🛠️  偵測到 {len(df_abnormal)} 支異常/缺失影片，請選擇後續處理方式：")
            print("  [1] 救援重算 (Rescue)  — 自動去黑邊放大 + 降低小臉偵測門檻 + 容許 B-roll 間隔重算")
            print("  [2] 標準重算 (Recalc)  — 使用標準參數重新提取特徵點並計算 QC")
            print("  [3] 隔離清理 (Prune)   — 將無法修復的無露臉影片移至隔離區，並同步重排順號與 JSON")
            print("  [0] 結束離開 (Exit)    — 僅保留診斷報告，不做任何檔案修改")
            print("=" * 90)
            choice = input("👉 請輸入選項編號 [0 / 1 / 2 / 3]: ").strip()
            choice_map = {"1": "rescue", "2": "recalc", "3": "prune", "0": "audit_only"}
            action = choice_map.get(choice, "audit_only")

        if action == "audit_only":
            print("[*] 已結束作業，未更動任何影片或 JSON 檔案。")
            break

        elif action == "rescue":
            execute_recompute_batch(
                df_abnormal,
                class_json_path=args.class_json,
                use_autocrop=True,
                min_conf=args.min_conf,
                max_gap=args.max_gap,
            )
            # 修復後自動重跑一次全盤驗證
            print("\n🔄 正在自動重跑全盤驗證以確認修復結果...")
            _, df_abnormal = run_full_audit_and_diagnosis(args.dir, args.out_dir)
            if df_abnormal.empty or args.action != "interactive":
                break
            action = "interactive"  # 若仍有剩餘無法救回的檔案，讓用戶繼續選擇（例如按 3 隔離）

        elif action == "recalc":
            execute_recompute_batch(
                df_abnormal,
                class_json_path=args.class_json,
                use_autocrop=False,
                min_conf=0.50,
                max_gap=1,
            )
            print("\n🔄 正在自動重跑全盤驗證以確認重算結果...")
            _, df_abnormal = run_full_audit_and_diagnosis(args.dir, args.out_dir)
            if df_abnormal.empty or args.action != "interactive":
                break
            action = "interactive"

        elif action == "prune":
            confirm = "y"
            if args.action == "interactive":
                confirm = input(f"⚠️ 確定要將這 {len(df_abnormal)} 支影片移入隔離區並重排所屬帳號順號嗎？(y/N): ").strip().lower()
            if confirm == "y":
                execute_prune_and_sync(df_abnormal, reels_dir=args.dir, reels_data_dir=args.data_dir)
                print("\n🔄 正在自動重跑全盤驗證以確認資料集完整性...")
                run_full_audit_and_diagnosis(args.dir, args.out_dir)
            else:
                print("[*] 已取消隔離操作。")
            break


if __name__ == "__main__":
    main()
