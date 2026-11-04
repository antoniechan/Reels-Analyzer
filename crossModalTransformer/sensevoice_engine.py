#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice Cantonese ASR & Semantic Saliency Engine
--------------------------------------------------
1. 本地輕量載入 SenseVoice-Small (支援 CPU 推論)。
2. 針對香港廣東話 (language='yue') 轉錄，精準識別口語字。
3. 輸出含時間戳記 (Timestamp) 的逐字稿。
4. 結合健康科普痛點詞與語速資訊密度，動態計算 (T,) 秒級語意張力。
"""

import re
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from funasr import AutoModel

# ---------------------------------------------------------------------------
# 全域單例載入模型 (避免每次調用重複載入)
# ---------------------------------------------------------------------------
_SENSEVOICE_MODEL = None

def get_sensevoice_model():
    global _SENSEVOICE_MODEL
    if _SENSEVOICE_MODEL is None:
        print("[*] 正在載入 SenseVoice-Small 本地廣東話語音模型 (約 200MB)...")
        _SENSEVOICE_MODEL = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cpu",          # 筆電無 GPU 也可極速在 CPU 運行；有 CUDA 可改 "cuda:0"
            disable_update=True,
            log_level="ERROR"
        )
        print("[+] SenseVoice-Small 載入完成！")
    return _SENSEVOICE_MODEL


def extract_audio_wav(video_path: Path) -> Path:
    """使用 ffmpeg 抽取 16kHz 單聲道 wav 供 SenseVoice 辨識"""
    tmp_wav = Path(tempfile.gettempdir()) / f"sv_{video_path.stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(tmp_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav


def clean_sensevoice_text(raw_text: str) -> tuple[str, list[str]]:
    """過濾 SenseVoice 特殊標籤 (如 <|yue|>, <|NEUTRAL|>)，並抽取情緒標籤"""
    emotions = re.findall(r"<\|([A-Z_]+)\|>", raw_text)
    # 去除所有特殊標籤，留下純廣東話文字
    cleaned = re.sub(r"<\|.*?\|>", "", raw_text).strip()
    return cleaned, emotions


def transcribe_cantonese_with_sensevoice(video_path: Path) -> list[dict]:
    """
    執行 SenseVoice 廣東話轉錄，輸出相容 segments 格式：
    [{"start": 0.0, "end": 2.5, "text": "如果你成日好攰..."}, ...]
    """
    model = get_sensevoice_model()
    audio_path = extract_audio_wav(video_path)

    try:
        # 調用 SenseVoice，指定 language="yue" (粵語)
        res = model.generate(
            input=str(audio_path),
            language="yue",
            use_itn=True,
            batch_size_s=60
        )

        segments = []
        if res and len(res) > 0:
            for item in res:
                raw_text = item.get("text", "")
                cleaned_text, emotions = clean_sensevoice_text(raw_text)

                # SenseVoice 若有輸出 timestamp 則讀取，否則建立整體段落
                ts = item.get("timestamp")
                if ts and isinstance(ts, list) and len(ts) > 0:
                    for sub in ts:
                        # [start_ms, end_ms] 轉為秒數
                        st = sub[0] / 1000.0 if isinstance(sub, list) else 0.0
                        ed = sub[1] / 1000.0 if isinstance(sub, list) else 2.0
                        segments.append({"start": st, "end": ed, "text": cleaned_text})
                else:
                    # 無分段時間戳時預設依長度均勻鋪開
                    segments.append({"start": 0.0, "end": 5.0, "text": cleaned_text})

        return segments

    finally:
        if audio_path.exists():
            try:
                os.remove(audio_path)
            except Exception:
                pass


def compute_sensevoice_semantic_saliency(segments: list[dict], duration_sec: int) -> np.ndarray:
    """
    取代原本死板的純 dictionary 暴力匹配：
    1. 結合 SenseVoice 精確辨識的廣東話文字。
    2. 依據秒級時間戳計算「字數密度 (Speech Rate)」+「前 3 秒 Hook 痛點加權」。
    3. 輸出符合 Transformer 第 8 維度的 (T,) 語意張力序列。
    """
    saliency = np.zeros(duration_sec, dtype=np.float32)

    # 廣東話短影音高權重痛點詞彙 (針對健康、細胞營養、生理機能)
    HK_HEALTH_KEYWORDS = [
        "點解", "千祈", "唔好", "成日", "好攰", "發炎", "血管", "心臟", "營養",
        "蛋白質", "吸收", "猝死", "熬夜", "瞓唔著", "脂肪肝", "腸道", "細胞"
    ]

    for seg in segments:
        text = seg.get("text", "")
        if not text:
            continue

        st = max(0, int(seg.get("start", 0)))
        ed = min(duration_sec, int(np.ceil(seg.get("end", st + 1))))
        span = max(1, ed - st)

        # 1. 資訊密度 (每秒字數)
        cps = len(text) / span
        density_score = np.clip(cps / 4.0, 0.2, 1.0)  # 粵語每秒約 3~4 字為緊湊

        # 2. 痛點關鍵詞加乘
        keyword_hits = sum(1 for kw in HK_HEALTH_KEYWORDS if kw in text)
        keyword_bonus = min(0.4, keyword_hits * 0.15)

        base_score = density_score + keyword_bonus

        for t in range(st, ed):
            # 首 3 秒 Hook 區域賦予更高注意力權重
            time_weight = 1.3 if t < 3 else 1.0
            saliency[t] = max(saliency[t], base_score * time_weight)

    # 平滑濾波 (避免秒與秒之間階梯跳動過大)
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    smoothed = np.convolve(saliency, kernel, mode="same")
    return np.clip(smoothed, 0.0, 1.5)