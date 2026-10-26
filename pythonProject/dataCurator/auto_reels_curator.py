#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apify Metadata Direct Ingestion & Filter via Gemini
---------------------------------------------------
1. 先利用 caption 進行文字極速初篩 (節省下載時間)。
2. 若文案合格，自動下載音軌 (audioUrl, ~1MB) 並上傳至 Gemini。
3. 判定是否符合「健康知識/觀念為核心的單人口播短影音」。
"""

import os
import json
import tempfile
from pathlib import Path
import requests
from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from google import genai
from google.genai import types


def check_caption_first(client: genai.Client, caption: str) -> bool:
    """第一階段：用文案快速排除明顯無關的影片 (純文字，幾百毫秒完成)"""
    if not caption or len(caption.strip()) < 10:
        return True  # 沒寫文案的先不排除，留待聽音訊

    prompt = f"""
請判斷以下 Instagram Reels 文案是否與「健康、疾病預防、營養學、人體生理機能、飲食調理」相關：
文案：
\"\"\"
{caption[:400]}
\"\"\"
如果完全無關（如純娛樂搞笑、美妝穿搭、旅遊美食、投資詐騙），請回傳 NO。
如果包含健康/營養/醫學知識，請回傳 YES。
只輸出 YES 或 NO。
"""
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return "YES" in res.text.upper()


def download_stream_to_temp(url: str, suffix: str = ".mp4") -> Path:
    """下載帶簽名的 Instagram CDN 檔案至暫存目錄"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    tmp_path = Path(tempfile.gettempdir()) / f"apify_temp_{os.getpid()}{suffix}"

    with requests.get(url, headers=headers, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
    return tmp_path


def evaluate_apify_record(item: dict) -> dict:
    """主評估函式：處理單條 Apify 抓取到的 record"""
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    caption = item.get("caption", "")
    duration = float(item.get("videoDuration", 60.0))
    audio_url = item.get("audioUrl") or item.get("videoUrl")

    # 1. 文案快篩
    if not check_caption_first(client, caption):
        return {
            "qualified": False,
            "stage": "caption_filtered",
            "reason": "文案主題與健康營養無關"
        }

    # 2. 下載音訊小檔 (audioUrl 通常只有 1MB)
    tmp_file = None
    uploaded_file = None
    try:
        tmp_file = download_stream_to_temp(audio_url, suffix=".mp4")
        uploaded_file = client.files.upload(file=str(tmp_file))

        # 3. 呼叫 Gemini 進行專業健康知識口播評級
        system_instruction = """
你是一位嚴謹的短影音資料集審核專家。你的任務是評估一段 Reels 短影音是否符合「健康知識與營養觀念科普短影音」的收錄標準。

【收錄標準 (ACCEPT)】：
1. 形式：講者單人口播（Monologue）為主。
2. 主題：健康、營養素、預防醫學、人體機制、亞健康改善。
3. 商業容許度：允許推薦產品，但必須以「解釋該成分/營養素對身體的重要性與生理機制」為切入點。

【淘汰標準 (REJECT)】：
1. 純促銷/硬廣，缺乏營養機理解析。
2. 情境短劇、多人採訪、搞笑娛樂、純背景音樂跳舞。

請以純 JSON 格式輸出：
{
  "qualified": true / false,
  "confidence_score": 0.0 ~ 1.0,
  "is_monologue_presentation": true / false,
  "health_knowledge_focus": true / false,
  "product_pitch_acceptable": true / false,
  "primary_topic": "簡短主題 (例如：心血管與熬夜猝死預防)",
  "reasoning": "簡短繁體中文說明判定依據"
}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded_file, "請聽取這段語音並結合其傳遞的知識，判定是否符合收錄標準。"],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text.strip())

    finally:
        # 清理暫存檔與 Gemini 雲端空間
        if tmp_file and tmp_file.exists():
            os.remove(tmp_file)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 測試你的範例資料
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_item = {
        "videoUrl": "...",  # 貼上你剛剛提供的 videoUrl
        "audioUrl": "...",  # 貼上你剛剛提供的 audioUrl
        "caption": """當你的身體出現以下三大症狀，其實是心臟正在發出最後求救訊號！⚠️
全球每 5 秒就有 1 人因心血管疾病猝死！天天熬夜、壓力大...
想知道營養師怎麼正確保養心血管、降低猝死風險嗎？
👉 追蹤我 @33dietitian...""",
        "videoDuration": 84.3
    }

    # 進行實測
    result = evaluate_apify_record(sample_item)
    print(json.dumps(result, ensure_ascii=False, indent=2))