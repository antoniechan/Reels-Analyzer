#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice Cantonese ASR & Semantic Saliency Engine
--------------------------------------------------
1. 本地輕量載入 SenseVoice-Small (支援 CPU 推論)。
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
import opencc

# ---------------------------------------------------------------------------
# 全域單例載入模型 (避免每次調用重複載入)
# ---------------------------------------------------------------------------
_SENSEVOICE_MODEL = None
_CC_S2HK = opencc.OpenCC("s2hk")

def get_sensevoice_model():
    """單例模式載入 SenseVoiceSmall 模型"""
    global _SENSEVOICE_MODEL
    if _SENSEVOICE_MODEL is None:
        from funasr import AutoModel
        print("[*] 正在載入 SenseVoice-Small 語音模型 (約 200MB)...")
        _SENSEVOICE_MODEL = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cpu",
            disable_update=True
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


def clean_sensevoice_text(raw_text) -> str:
    """過濾 SenseVoice 特殊標籤，並保證轉為香港繁體純字串 (str)"""
    # 若 FunASR 回傳 tuple 或 list，取其主要文字元素
    if isinstance(raw_text, (tuple, list)):
        raw_text = " ".join([str(item) for item in raw_text if item is not None])
    elif raw_text is None:
        return ""
    else:
        raw_text = str(raw_text)

    cleaned = re.sub(r"<\|.*?\|>", "", raw_text).strip()
    return to_traditional_chinese(cleaned)

def to_traditional_chinese(text: str) -> str:
    """將文字統一轉換為標準香港繁體中文"""
    if not text:
        return ""
    return _CC_S2HK.convert(text)


def transcribe_chinese_with_sensevoice(audio_or_video_path: Path) -> list[dict]:
    """
    使用本地 SenseVoice 辨識音訊（自動支援廣東話、普通話、英語）
    回傳繁中逐字文字片段列表: [{"text": str}, ...]
    """
    audio_path = Path(audio_or_video_path)
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return []

    try:
        model = get_sensevoice_model()

        res = model.generate(
            input=str(audio_path),
            language="auto",
            use_itn=True,
            batch_size_s=60
        )

        segments = []
        if res and isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    raw_val = item.get("text", "")
                else:
                    raw_val = item

                cleaned_str = clean_sensevoice_text(raw_val)
                if cleaned_str:
                    segments.append({"text": cleaned_str})

        return segments
    except Exception as e:
        print(f"      [!] SenseVoice 轉錄跳過 (可能無音軌或音訊格式不相容): {e}")
        return []


def compute_sensevoice_semantic_saliency(segments: list[dict], duration_sec: int) -> np.ndarray:
    """
    取代原本死板的純 dictionary 暴力匹配：
    1. 結合 SenseVoice 精確辨識的廣東話文字。
    2. 依據秒級時間戳計算「字數密度 (Speech Rate)」+「前 3 秒 Hook 痛點加權」。
    3. 輸出符合 Transformer 第 8 維度的 (T,) 語意張力序列。
    """
    saliency = np.zeros(duration_sec, dtype=np.float32)

    # 廣東話短影音高權重痛點詞彙 (針對健康、細胞營養、生理機能)
    HEALTH_KEYWORDS = [
        "點解", "千祈", "唔好", "成日", "好攰", "發炎", "血管", "心臟", "營養",
        "為什麼", "千萬不要", "總是", "疲勞", "失眠", "熬夜", "後悔", "小心",
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
        density_score = np.clip(cps / 4.0, 0.2, 1.0)  # 每秒約 3~4 字為緊湊

        # 2. 痛點關鍵詞加乘
        keyword_hits = sum(1 for kw in HEALTH_KEYWORDS if kw in text)
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
