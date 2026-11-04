#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Diagnostic Tester for External New Videos (With Whisper Transcript Analysis)
---------------------------------------------------------------------------------------
功能：
1. 輸入外部新影片 (.mp4 / .mov)，自動提取：
   - 視覺與光影 (4 維): colorfulness, face_rms, skin_warmth, fg_bg_contrast
   - 臉部微表情形變 (2 維): key_mus, all_mus
   - 聲學與真實逐字稿語意 (2 維): audio_energy, transcript_semantic_saliency
2. 內建 Whisper 語音轉逐字稿引擎 (支援 openai-whisper / faster-whisper 或外部同名 .srt/.json)。
3. 結合語意與時序 Transformer，輸出包含「具體台詞修改與 B-roll 秒數定位」的後製指南。
"""

import os
import sys
import re
import json
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import TRANSFORMER_DIR, PYTHON_PROJECT_DIR

from train_transformer_engine import ReelsDiagnosticTransformer
from diagnose_reel_video import (
    compute_golden_benchmarks,
    find_continuous_intervals,
    plot_diagnostic_timeline,
)
from sensevoice_engine import (
    transcribe_cantonese_with_sensevoice,
    compute_sensevoice_semantic_saliency
)

ENGINE_DIR = TRANSFORMER_DIR / "transformer_engine"
DATASET_PATH = ENGINE_DIR / "transformer_multimodal_dataset.pt"
MODEL_PATH = ENGINE_DIR / "best_reels_transformer.pth"
STATS_PATH = ENGINE_DIR / "transformer_norm_stats.json"
EXTERNAL_REPORT_DIR = ENGINE_DIR / "external_test_reports"
EXTERNAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = PYTHON_PROJECT_DIR / ".env"
if not ENV_PATH.exists():
    ENV_PATH = PYTHON_PROJECT_DIR.parent / ".env"  # 防禦性檢查更上層

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()  # 預設自當前工作目錄向上搜尋


# ---------------------------------------------------------------------------
# 1. 語音轉逐字稿 (Whisper ASR) 與秒級台詞語意評分器
# ---------------------------------------------------------------------------
# 高吸睛度痛點詞庫 (涵蓋粵語/書面語常見爆款短影音關鍵詞)
HIGH_IMPACT_KEYWORDS = [
    "千萬", "千祈", "唔好", "不要", "點解", "為什麼", "為甚麼", "真相", "秘密", "後悔",
    "其實", "原來", "警告", "注意", "只要", "教你", "方法", "秘訣", "重點", "關鍵",
    "竟然", "居然", "小心", "解決", "改善", "效果", "90%", "99%", "第一", "最後",
    "到底", "立刻", "馬上", "免費", "必看", "停止", "錯誤", "陷阱", "萬萬"
]

# 慢熱寒暄扣分詞庫 (出現在前 3 秒會大幅拉低完播率)
WEAK_OPENING_PHRASES = [
    "大家好", "哈囉", "hello", "歡迎來到", "我是", "今天想", "今日想同大家", "好久不見"
]


def extract_audio_for_gemini(video_path: Path) -> Path:
    """使用 ffmpeg 從影片抽取低碼率 mp3，節省傳輸頻寬與延遲"""
    tmp_audio = Path(tempfile.gettempdir()) / f"gemini_audio_{video_path.stem}.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        str(tmp_audio)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_audio

def transcribe_video_audio(video_path: Path) -> list[dict]:
    """
    使用 Google Gemini API 提取精確香港廣東話逐字稿 (含時間戳記)
    回傳格式: [{"start": float, "end": float, "text": str}, ...]
    """
    # 1. 檢查是否已有現存快取檔案 (避免重複測試同支影片時浪費 Token)
    cache_json = video_path.with_name(f"{video_path.stem}_gemini_transcript.json")
    if cache_json.exists():
        print(f"[*] 發現現有 Gemini 逐字稿快取: {cache_json.name}，直接載入...")
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("segments", data) if isinstance(data, dict) else data
        except Exception:
            pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"[!] 找不到 GEMINI_API_KEY，請確認 {ENV_PATH} 中是否包含 GEMINI_API_KEY=xxx")
        return []

    print("[*] 成功讀取根目錄 .env 密鑰，正在透過 Gemini 進行轉錄...")

    # 抽取暫存音訊
    audio_path = extract_audio_for_gemini(video_path)
    if not audio_path.exists():
        print("[-] 音訊抽取失敗，無法使用 Gemini 轉錄。")
        return []

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # 上傳音訊檔案至 File API
        uploaded_audio = client.files.upload(file=str(audio_path))

        prompt = """
你是一位精通香港廣東話和普通話及口語書寫的語音辨識專家。
請將這段音訊轉錄為精確的文字（如為廣東話則包括香港地道口語字，如：唔、喺、咁、點解、邊度、千祈、成日、仲有，以及常見的中英夾雜詞彙）。

請輸出為合法的 JSON 陣列，每個元素包含：
- "start": 片段開始秒數 (float，精確至小數點後 1 位)
- "end": 片段結束秒數 (float，精確至小數點後 1 位)
- "text": 對應的廣東話逐字內容 (string)

範例格式：
[
  {"start": 0.0, "end": 2.4, "text": "如果你平時返工成日好攰"},
  {"start": 2.4, "end": 5.1, "text": "千祈唔好剩係飲咖啡提神"}
]

請只輸出純 JSON 陣列，不要加入任何額外解釋或 Markdown 標籤。
"""

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[uploaded_audio, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )

        raw_text = response.text.strip()
        segments = json.loads(raw_text)

        # 確保格式合法
        formatted_segments = []
        for s in segments:
            formatted_segments.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip()
            })

        # 儲存快取，下次秒讀
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(formatted_segments, f, ensure_ascii=False, indent=2)

        print(f"[+] Gemini 粵語辨識完成！共識別 {len(formatted_segments)} 個台詞片段，已快取至 {cache_json.name}")
        return formatted_segments

    except Exception as e:
        print(f"[!] Gemini 辨識發生異常: {e}")
        return []
    finally:
        # 清理暫存音訊檔
        if audio_path.exists():
            try:
                os.remove(audio_path)
            except Exception:
                pass

def compute_second_level_semantic_saliency(segments: list[dict], duration_sec: int, fallback_rms_norm: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    將 Whisper 逐字稿映射至每一秒，計算秒級語意吸睛度 semantic_ts (0.0 ~ 1.0)，
    並回傳逐秒台詞字典供診斷報告引用。
    """
    semantic_ts = np.full(duration_sec, 0.35, dtype=np.float32)
    sec_to_text = {s: "" for s in range(duration_sec)}
    has_weak_greeting_in_hook = False

    if not segments:
        # 若無逐字稿，退回使用聲能正規化值
        return 0.4 + 0.45 * fallback_rms_norm, sec_to_text

    for seg in segments:
        st_sec = max(0, int(np.floor(seg["start"])))
        ed_sec = min(duration_sec - 1, int(np.ceil(seg["end"])))
        text = seg.get("text", "").strip()
        if not text or ed_sec < st_sec:
            continue

        dur = max(0.5, seg["end"] - seg["start"])
        # 1. 計算語速資訊密度 (每秒字元數 CPS，理想短影音節奏約 4.5 ~ 7.5 字/秒)
        clean_len = len(re.sub(r"\s+", "", text))
        cps = clean_len / dur
        density_score = np.clip((cps - 1.5) / 5.5, 0.0, 1.0) * 0.35  # 最高貢獻 +0.35

        # 2. 關鍵痛點詞與數字加分
        keyword_hits = sum(1 for kw in HIGH_IMPACT_KEYWORDS if kw in text.lower())
        has_digits = 1 if re.search(r"\d+", text) else 0
        has_question = 1 if ("?" in text or "？" in text or "嗎" in text or "呢" in text or "點解" in text) else 0
        impact_bonus = min(0.35, keyword_hits * 0.14 + has_digits * 0.10 + has_question * 0.10)

        # 3. 檢查前 3 秒是否出現慢熱寒暄
        if st_sec < 3:
            if any(greet in text.lower() for greet in WEAK_OPENING_PHRASES):
                has_weak_greeting_in_hook = True
                impact_bonus -= 0.25  # 嚴重扣分

        seg_score = float(np.clip(0.35 + density_score + impact_bonus, 0.15, 0.98))

        for s in range(st_sec, ed_sec + 1):
            semantic_ts[s] = max(semantic_ts[s], seg_score)
            if text not in sec_to_text[s]:
                sec_to_text[s] = (sec_to_text[s] + " " + text).strip()

    # 若前 5 秒整體分數高，平滑強化 Hook 區段表現（對齊訓練集 hook_first_5s_score 特性）
    hook_mean = float(np.mean(semantic_ts[:min(5, duration_sec)]))
    semantic_ts[:min(5, duration_sec)] = 0.6 * semantic_ts[:min(5, duration_sec)] + 0.4 * hook_mean

    meta_transcript = {
        "sec_to_text": sec_to_text,
        "full_text": " | ".join([f"[{s['start']:.1f}s] {s['text']}" for s in segments]),
        "has_weak_greeting_in_hook": has_weak_greeting_in_hook,
        "hook_text": " ".join([s["text"] for s in segments if s["start"] < 3.5]).strip(),
    }
    return semantic_ts, meta_transcript


# ---------------------------------------------------------------------------
# 2. 視覺與光影特徵計算 (單幀)
# ---------------------------------------------------------------------------
def compute_frame_visual_features(frame_bgr: np.ndarray, face_bbox=None) -> tuple[float, float, float, float]:
    h, w, _ = frame_bgr.shape
    frame_float = frame_bgr.astype(np.float32)
    B, G, R = frame_float[:, :, 0], frame_float[:, :, 1], frame_float[:, :, 2]

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    colorfulness = float(std_root + 0.3 * mean_root)

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
    else:
        x1, y1, x2, y2 = int(w * 0.25), int(h * 0.2), int(w * 0.75), int(h * 0.65)

    face_roi = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if face_roi.size == 0:
        face_roi = frame_bgr

    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    face_rms = float(np.std(face_gray) / 255.0)

    face_ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Cr, Cb = face_ycrcb[:, :, 1], face_ycrcb[:, :, 2]
    skin_mask = (Cr > 133) & (Cr < 173) & (Cb > 77) & (Cb < 127)
    skin_warmth = float(np.mean(Cr[skin_mask] - Cb[skin_mask])) if np.any(skin_mask) else float(np.mean(Cr - Cb))

    full_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fg_mask = np.zeros((h, w), dtype=bool)
    fg_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = True
    fg_lum = np.mean(full_gray[fg_mask])
    bg_lum = np.mean(full_gray[~fg_mask])
    fg_bg_contrast = float(abs(fg_lum - bg_lum) / (fg_lum + bg_lum + 1e-5))

    return colorfulness, face_rms, skin_warmth, fg_bg_contrast


# ---------------------------------------------------------------------------
# 3. 聲學能量與 Whisper 逐字稿雙流提取 (1 Hz)
# ---------------------------------------------------------------------------
def extract_external_audio_and_transcript(video_path: Path, duration_sec: int) -> tuple[np.ndarray, np.ndarray, dict]:
    audio_energy_ts = np.full(duration_sec, 0.2, dtype=np.float32)
    norm_rms = np.full(duration_sec, 0.3, dtype=np.float32)

    # A. 提取音軌 RMS 聲能
    try:
        import librosa
        y, sr = librosa.load(str(video_path), sr=16000, mono=True)
    except Exception:
        try:
            from scipy.io import wavfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_name = tmp_wav.name
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", tmp_name]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sr, y = wavfile.read(tmp_name)
            y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-6)
            os.remove(tmp_name)
        except Exception:
            y, sr = None, 16000

    if y is not None and len(y) > 0:
        sec_rms = []
        for s in range(duration_sec):
            st, ed = s * sr, min((s + 1) * sr, len(y))
            sec_rms.append(float(np.sqrt(np.mean(y[st:ed] ** 2))) if ed > st else 0.0)
        sec_rms = np.array(sec_rms, dtype=np.float32)

        if np.max(sec_rms) > 1e-6:
            p75 = np.percentile(sec_rms, 75)
            for s in range(duration_sec):
                if sec_rms[s] >= p75:
                    audio_energy_ts[s] = 1.0
                    if s + 1 < duration_sec:
                        audio_energy_ts[s + 1] = max(audio_energy_ts[s + 1], 0.6)
                    if s - 1 >= 0:
                        audio_energy_ts[s - 1] = max(audio_energy_ts[s - 1], 0.6)
            norm_rms = (sec_rms - np.min(sec_rms)) / (np.max(sec_rms) - np.min(sec_rms) + 1e-6)

    # B. 執行 Whisper 逐字稿辨識與秒級語意計分
    segments = transcribe_video_audio(video_path)
    semantic_ts, transcript_meta = compute_second_level_semantic_saliency(segments, duration_sec, norm_rms)

    return audio_energy_ts, semantic_ts, transcript_meta

# ---------------------------------------------------------------------------
# 4. 整合提取：將外部影片轉為 (T, 8) 多模態張量 + 逐字稿元數據
# ---------------------------------------------------------------------------
def get_safe_cascade_classifier():
    """解決 Windows 中文路徑導致 OpenCV 無法讀取 haarcascade xml 的問題"""
    import shutil
    orig_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not orig_xml.exists():
        return None
    try:
        # 先複製到純英文路徑的系統 Temp 資料夾
        temp_xml = Path(tempfile.gettempdir()) / "haarcascade_frontalface_default.xml"
        if not temp_xml.exists():
            shutil.copyfile(str(orig_xml), str(temp_xml))
        clf = cv2.CascadeClassifier(str(temp_xml))
        if clf.empty():
            return None
        return clf
    except Exception:
        return None


def detect_face_bbox_robust(frame_bgr: np.ndarray, gray: np.ndarray, face_cascade) -> tuple[int, int, int, int] | None:
    """三重防禦人臉定位：1. OpenCV Safe Cascade -> 2. YCrCb 膚色連通域定位"""
    h, w = gray.shape

    # 策略 A: 使用修復路徑後的 CascadeClassifier
    if face_cascade is not None:
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                return (fx, fy, fx + fw, fy + fh)
        except Exception:
            pass

    # 策略 B: 利用 YCrCb 膚色遮罩尋找人臉重心區 (完全不依賴外部 xml 模型)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = ((cr > 135) & (cr < 175) & (cb > 80) & (cb < 125)).astype(np.uint8) * 255

    # 限制在畫面中上段 (避免手部干擾)
    skin_mask[int(h * 0.75):, :] = 0
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > (h * w * 0.015):
            fx, fy, fw, fh = cv2.boundingRect(largest_cnt)
            return (fx, fy, fx + fw, fy + fh)

    return None


def extract_multimodal_tensor_from_mp4(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片檔案: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = max(3, min(int(round(total_frames / fps)), 120))

    print(f"[*] 正在掃描外部影片視覺與表情: {video_path.name} (FPS: {fps:.1f}, 長度: {duration_sec} 秒)...")

    vis_matrix = np.zeros((duration_sec, 4), dtype=np.float32)
    key_mus = np.zeros(duration_sec, dtype=np.float32)
    all_mus = np.zeros(duration_sec, dtype=np.float32)

    # 安全載入分類器 (避開中文路徑問題)
    face_cascade = get_safe_cascade_classifier()
    prev_face_gray = None

    for s in range(duration_sec):
        frame_idx = min(int(s * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bbox = detect_face_bbox_robust(frame, gray, face_cascade)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size > 0:
                curr_face_gray = cv2.resize(roi, (128, 128))
            else:
                curr_face_gray = cv2.resize(gray[int(h*0.2):int(h*0.7), int(w*0.25):int(w*0.75)], (128, 128))
        else:
            curr_face_gray = cv2.resize(gray[int(h*0.2):int(h*0.7), int(w*0.25):int(w*0.75)], (128, 128))

        vis_matrix[s] = compute_frame_visual_features(frame, bbox)

        if prev_face_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_face_gray, curr_face_gray, None,
                pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            du_dx = np.gradient(flow[:, :, 0], axis=1)
            dv_dy = np.gradient(flow[:, :, 1], axis=0)
            du_dy = np.gradient(flow[:, :, 0], axis=0)
            dv_dx = np.gradient(flow[:, :, 1], axis=1)

            shear_mag = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2)
            all_mus[s] = float(np.clip(np.mean(shear_mag) * 0.45, 0.01, 0.45))
            eye_region = shear_mag[25:55, 20:108]
            mouth_region = shear_mag[75:115, 20:108]
            mouth_eye_region = np.concatenate([eye_region, mouth_region], axis=0)
            key_mus[s] = float(np.clip(np.mean(mouth_eye_region) * 0.48, 0.01, 0.48))

        prev_face_gray = curr_face_gray

    if duration_sec > 1:
        key_mus[0], all_mus[0] = key_mus[1], all_mus[1]

    cap.release()

    # 提取音訊聲能 + Whisper 逐字稿語意分數
    aud_energy, sem_saliency, transcript_meta = extract_external_audio_and_transcript(video_path, duration_sec)

    feature_matrix = np.column_stack([
        vis_matrix,
        key_mus[:, None],
        all_mus[:, None],
        aud_energy[:, None],
        sem_saliency[:, None]
    ])

    return {
        "account_id": "external_test",
        "video_id": video_path.stem,
        "tier_label": 1,
        "quality_score": 50.0,
        "duration": duration_sec,
        "time_series": torch.tensor(feature_matrix, dtype=torch.float32),
        "transcript_meta": transcript_meta
    }


# ---------------------------------------------------------------------------
# 5. 主程序：結合 Transformer 與逐字稿內容輸出具體建議
# ---------------------------------------------------------------------------
def main():
    print("=" * 85)
    print("【Reels 外部新影片 AI 後製前秒級體檢系統 (含 Whisper 逐字稿語意分析)】")
    print("=" * 85)

    if len(sys.argv) > 1:
        video_path_str = sys.argv[1].strip('"').strip("'")
    else:
        video_path_str = input("👉 請輸入或拖曳外部影片路徑 (.mp4 / .mov): ").strip('"').strip("'")

    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"[-] 找不到指定的影片檔案: {video_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(DATASET_PATH, weights_only=False)
    benchmarks = compute_golden_benchmarks(dataset)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32)

    model = ReelsDiagnosticTransformer(
        vis_dim=4, geom_dim=2, aud_dim=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=3
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 1. 提取多模態張量與逐字稿
    record = extract_multimodal_tensor_from_mp4(video_path)
    t_meta = record.get("transcript_meta", {})
    sec_to_text = t_meta.get("sec_to_text", {})

    # 2. 執行基礎模型診斷
    from diagnose_reel_video import diagnose_single_video
    diag = diagnose_single_video(record, model, norm_mean, norm_std, benchmarks, device)

    # 3. 附加【逐字稿專屬診斷建議】
    if t_meta.get("has_weak_greeting_in_hook"):
        hook_txt = t_meta.get("hook_text", "")
        diag["advice_list"].insert(
            0,
            f"✂️ 【首 3 秒台詞寒暄警告】：偵測到開頭台詞包含慢熱問候語（「{hook_txt}」）。"
            f"短影音觀眾在前 2 秒最容易滑走，強烈建議後製直接剪掉「大家好/我是...」這段寒暄，把後面的核心痛點句直接拉到第 0 秒破題！"
        )

    # 為每個建議插入 B-roll 的區間補上「當時正在講的台詞」，方便剪輯師對照
    enhanced_advice = []
    for adv in diag["advice_list"]:
        if "畫面動態疲乏區間" in adv:
            m = re.search(r"\((\d+)s~(\d+)s", adv)
            if m:
                st, ed = int(m.group(1)), int(m.group(2))
                spoken_in_window = []
                for s in range(st, ed + 1):
                    txt = sec_to_text.get(s, "")
                    if txt and txt not in spoken_in_window:
                        spoken_in_window.append(txt)
                if spoken_in_window:
                    quote = " / ".join(spoken_in_window)
                    adv += f"\n     💬 [對應台詞定位]：「{quote}」—— 建議根據這句台詞的內容配搭對應的 B-roll 畫面或字卡。"
        enhanced_advice.append(adv)

    # 4. 印出完整報告
    print("\n" + "=" * 85)
    print(f"🎬 外部測試影片: {video_path.name} (總長度: {diag['duration']} 秒)")
    if t_meta.get("full_text"):
        print(f"📝 辨識逐字稿摘要: {t_meta['full_text'][:120]}...")
    print(f"🤖 Transformer 預測評級: {diag['pred_tier']}")
    print(f"🔥 爆款潛力指數: {diag['good_prob']:.1f}% | 滑走流失風險: {diag['bad_prob']:.1f}%")
    print("-" * 85)
    print("🛠️ 【AI 後製剪輯具體修改建議 (結合視覺、聲學與逐字稿語意)】：")
    for i, adv in enumerate(enhanced_advice, 1):
        print(f"  {i}. {adv}")

    plot_file = EXTERNAL_REPORT_DIR / f"external_diagnosis_{video_path.stem}.png"
    plot_diagnostic_timeline(diag, benchmarks, plot_file)
    print("-" * 85)
    print(f"📈 專屬秒級體檢儀表板圖已儲存至:\n   {plot_file}")
    print("=" * 85)

    try:
        from interactive_video_inspector import launch_inspector
        print("\n🎬 正在啟動互動式審片播放器 (可邊看邊點擊圖表跳轉)...")
        launch_inspector(str(video_path), diag, benchmarks)
    except Exception as e:
        print(f"[!] 無法啟動圖形介面播放器: {e}")


if __name__ == "__main__":
    main()