#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Multimodal Time-Series Dataset Builder for Transformer (Fixed & Robust)
-------------------------------------------------------------------------------
核心功能：
1. 以 (account_id, video_stem) 雙鍵精準對齊四大特徵檔案。
2. 具備防越界重採樣 (Safe Resampling) 與異常時長自動校正 (防止影格數誤當秒數)。
3. 打包輸出為單一檔案: transformer_multimodal_dataset.pt
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. 路徑設定與環境載入
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import VIDEO_PROC_DIR, DATA_PROC_DIR, AUDIO_PROC_DIR, DEFAULT_REELS_DIR
except ImportError:
    VIDEO_PROC_DIR = CURRENT_DIR
    DATA_PROC_DIR = PROJECT_ROOT / "dataProcessing"
    AUDIO_PROC_DIR = PROJECT_ROOT / "audioProcessing"
    DEFAULT_REELS_DIR = Path(r"D:\reels")

DEFAULT_CLASS_JSON = DATA_PROC_DIR / "video_classification_summary.json"
DEFAULT_AUDIO_JSON = AUDIO_PROC_DIR / "local_transcript_audio_saliency_summary_all_tiers.json"
OUTPUT_DIR = CURRENT_DIR / "transformer_engine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 嘴角提肌、口輪匝肌、眼周眼角 MediaPipe 關鍵頂點集合
EXPRESSION_KEY_VERTICES = set([0, 13, 14, 17, 61, 291, 33, 263, 130, 359])


# ---------------------------------------------------------------------------
# 2. 磁碟檔案索引建置
# ---------------------------------------------------------------------------
def build_file_index(root_dir: Path, suffix: str) -> dict:
    print(f"[*] 正在索引 {root_dir} 下的 *{suffix} ...")
    pattern = str(root_dir / "**" / f"*{suffix}")
    all_files = glob.glob(pattern, recursive=True)
    lookup = {}
    for orig_p in all_files:
        norm_p = orig_p.replace("\\", "/")
        parts = norm_p.split("/")
        fname = parts[-1].lower()
        vid_stem = fname[:-len(suffix)] if fname.endswith(suffix) else os.path.splitext(fname)[0]
        for folder_part in parts[:-1]:
            acc_cand = folder_part.lower().strip()
            if acc_cand and acc_cand not in ["d:", "reels"]:
                lookup[(acc_cand, vid_stem)] = orig_p
                for sfx in ["_processed", "_face", "_aligned"]:
                    if sfx in vid_stem:
                        lookup[(acc_cand, vid_stem.replace(sfx, ""))] = orig_p
    return lookup


# ---------------------------------------------------------------------------
# 3. 模態時序提取工具函式
# ---------------------------------------------------------------------------
def extract_visual_timeseries(cl_series: list, target_t: int) -> np.ndarray:
    """安全地將 cl_series 對齊至 target_t 秒，徹底杜絕 IndexError"""
    vis_matrix = np.zeros((target_t, 4), dtype=np.float32)
    actual_points = len(cl_series)
    if actual_points == 0:
        return vis_matrix

    for t in range(target_t):
        # 計算第 t 秒對應在 cl_series 中的區間索引
        start_idx = int((t / target_t) * actual_points)
        end_idx = int(((t + 1) / target_t) * actual_points)

        # 嚴格邊界保護，確保必在 [0, actual_points-1] 內
        start_idx = min(max(0, start_idx), actual_points - 1)
        end_idx = min(max(start_idx + 1, end_idx), actual_points)

        chunk = cl_series[start_idx:end_idx]
        vis_matrix[t, 0] = np.mean([float(c.get("colorfulness", 0.0)) for c in chunk])
        vis_matrix[t, 1] = np.mean([float(c.get("face_rms_contrast", 0.0)) for c in chunk])
        vis_matrix[t, 2] = np.mean([float(c.get("skin_warmth_cbcr", 0.0)) for c in chunk])
        vis_matrix[t, 3] = np.mean([float(c.get("figure_ground_lum_contrast", 0.0)) for c in chunk])

    return vis_matrix


def extract_qc_timeseries(qc_path: str, max_t: int) -> tuple[np.ndarray, np.ndarray]:
    """從 _qc.json 提取每秒的 key_mus 與 all_mus"""
    key_ts = np.zeros(max_t, dtype=np.float32)
    all_ts = np.zeros(max_t, dtype=np.float32)
    try:
        with open(qc_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return key_ts, all_ts

    transitions = d.get("consecutive_qc_metrics", [])
    if not transitions:
        return key_ts, all_ts

    sec_key_collect = {s: [] for s in range(max_t)}
    sec_all_collect = {s: [] for s in range(max_t)}

    total_transitions = len(transitions)
    for idx, trans in enumerate(transitions):
        if not trans.get("valid_transition", False):
            continue
        sec_idx = int((idx / max(total_transitions, 1)) * max_t)
        sec_idx = min(max(0, sec_idx), max_t - 1)

        for fm in trans.get("facet_metrics", []):
            mu_val = float(fm.get("abs_mu", 0.0))
            if mu_val > 1e-6:
                sec_all_collect[sec_idx].append(mu_val)
                verts = fm.get("vertices", [])
                if any(v in EXPRESSION_KEY_VERTICES for v in verts):
                    sec_key_collect[sec_idx].append(mu_val)

    for s in range(max_t):
        key_ts[s] = np.mean(sec_key_collect[s]) if sec_key_collect[s] else (key_ts[s - 1] if s > 0 else 0.0)
        all_ts[s] = np.mean(sec_all_collect[s]) if sec_all_collect[s] else (all_ts[s - 1] if s > 0 else 0.0)

    return key_ts, all_ts


def extract_audio_timeseries(audio_info: dict, max_t: int) -> tuple[np.ndarray, np.ndarray]:
    """提取每秒的聲能 peak 脈衝與文本語意吸睛度"""
    audio_energy_ts = np.zeros(max_t, dtype=np.float32)
    semantic_ts = np.zeros(max_t, dtype=np.float32)

    mean_eng = float(audio_info.get("mean_engagement_score", 50.0)) / 100.0
    hook_5s = float(audio_info.get("hook_first_5s_score", mean_eng * 100.0)) / 100.0
    semantic_ts[:] = mean_eng
    semantic_ts[:min(5, max_t)] = hook_5s

    audio_energy_ts[:] = 0.2
    peaks = audio_info.get("top_peak_seconds", [])
    for p_sec in peaks:
        sec = int(round(float(p_sec)))
        if 0 <= sec < max_t:
            audio_energy_ts[sec] = 1.0
            if sec + 1 < max_t:
                audio_energy_ts[sec + 1] = max(audio_energy_ts[sec + 1], 0.6)
            if sec - 1 >= 0:
                audio_energy_ts[sec - 1] = max(audio_energy_ts[sec - 1], 0.6)

    return audio_energy_ts, semantic_ts


# ---------------------------------------------------------------------------
# 4. 主執行資料集裝配流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【啟動時序 Transformer 多模態資料集建置管線】")
    print("=" * 80)

    color_index = build_file_index(DEFAULT_REELS_DIR, "_color_lighting.json")
    qc_index = build_file_index(DEFAULT_REELS_DIR, "_qc.json")

    audio_dict = {}
    if DEFAULT_AUDIO_JSON.exists():
        with open(DEFAULT_AUDIO_JSON, "r", encoding="utf-8") as f:
            adata = json.load(f)
        for v in adata.get("videos", []):
            acc = str(v.get("account_id", "")).lower().strip()
            vid = str(v.get("video_id", "")).lower().strip()
            audio_dict[(acc, vid)] = v

    videos_meta = []
    if DEFAULT_CLASS_JSON.exists():
        with open(DEFAULT_CLASS_JSON, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        videos_meta = cdata.get("videos", cdata) if isinstance(cdata, dict) else cdata

    dataset_records = []
    tier_map = {"bad": 0, "neutral": 1, "good": 2}
    abnormal_len_records = []

    print("\n[*] 開始逐支影片時序對齊與特徵矩陣組裝...")
    for item in videos_meta:
        if not isinstance(item, dict):
            continue
        acc = str(item.get("account_id", "")).lower().strip()
        vid = str(item.get("video_id", "")).lower().strip()
        tier_str = str(item.get("video_classification", item.get("tier", ""))).lower().strip()
        if tier_str not in tier_map:
            continue

        cl_path = color_index.get((acc, vid))
        if not cl_path:
            continue

        try:
            with open(cl_path, "r", encoding="utf-8") as f:
                cl_data = json.load(f)
        except Exception:
            continue

        cl_series = cl_data.get("second_level_time_series", [])
        if len(cl_series) < 3:
            continue

        # 取得記錄的秒數
        raw_dur = float(cl_data.get("duration_sec", len(cl_series)))
        T = int(round(raw_dur)) if raw_dur > 0 else len(cl_series)

        # 診斷異常長度：若 T > 180 (超過3分鐘)，檢查是否誤將影格數當成秒數
        if T > 180:
            abnormal_len_records.append((acc, vid, T, len(cl_series)))
            # 若數值大於 180，但我們知道 Reels 不超過 3 分鐘，可能是 FPS 讀取失誤 (例如以 10~30 倍膨脹) 或長片截斷至 180
            T = min(T, 180)

        if T < 3:
            continue

        # 1. 安全提取 4 維視覺光影時序
        vis_matrix = extract_visual_timeseries(cl_series, target_t=T)

        # 2. 提取 2 維 QC 擬共形面部幾何時序
        qc_path = qc_index.get((acc, vid))
        if qc_path:
            key_mus, all_mus = extract_qc_timeseries(qc_path, max_t=T)
        else:
            key_mus = np.zeros(T, dtype=np.float32)
            all_mus = np.zeros(T, dtype=np.float32)

        # 3. 提取 2 維音訊與語意時序
        a_info = audio_dict.get((acc, vid), {})
        aud_energy, sem_saliency = extract_audio_timeseries(a_info, max_t=T)

        # 合併成 (T, 8) 多模態特徵張量
        feature_matrix = np.column_stack([
            vis_matrix,
            key_mus[:, None],
            all_mus[:, None],
            aud_energy[:, None],
            sem_saliency[:, None]
        ])

        q_score = float(item.get("video_quality_score", 50.0))

        record = {
            "account_id": acc,
            "video_id": vid,
            "tier_label": tier_map[tier_str],
            "quality_score": q_score,
            "duration": T,
            "time_series": torch.tensor(feature_matrix, dtype=torch.float32)
        }
        dataset_records.append(record)

    if abnormal_len_records:
        print(f"\n[!] 偵測到 {len(abnormal_len_records)} 支影片的 JSON 記錄時長超過 180 秒 (已自動校正上限至 180s)：")
        for acc, vid, orig_t, series_len in abnormal_len_records[:5]:
            print(f"    - 帳號: {acc} | 影片: {vid} | JSON duration_sec={orig_t}, series_len={series_len}")

    durations = [r["duration"] for r in dataset_records]
    print(f"\n[+] 成功打包多模態時序影片數: {len(dataset_records)} 支！")
    print(f"  - 影片時長分佈: 最短 {min(durations)} 秒 | 平均 {np.mean(durations):.1f} 秒 | 最長 {max(durations)} 秒")

    output_file = OUTPUT_DIR / "transformer_multimodal_dataset.pt"
    torch.save(dataset_records, output_file)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[+] 資料集已成功匯出至: {output_file} ({file_size_mb:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()