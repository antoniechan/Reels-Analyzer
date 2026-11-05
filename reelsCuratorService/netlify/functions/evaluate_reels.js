import { GoogleGenAI } from "@google/genai";

const SYSTEM_INSTRUCTION = `
你是一位嚴謹的短影音資料集審核專家。你的任務是評估一段 Reels 短影音逐字稿是否符合「健康知識與營養觀念科普短影音」的收錄標準。

【嚴格收錄標準 (qualified: true)】：
1. 形式：主要由單一講者對鏡頭進行口播（Monologue）傳遞知識、解答迷思。
2. 主題：健康、營養素作用、人體生理機能、亞健康改善、預防醫學。
3. 商業容許度：【可以推薦產品或保健成分】，但前提必須是「以解釋該成分/營養素對身體有何重要性與科學原理」為切入點。

【淘汰標準 (qualified: false)】：
1. 純促銷/硬廣：只強調折扣特價、開箱，無人體機理說明。
2. 純情境短劇、多人綜藝訪談、搞笑娛樂、純背景音樂跳舞。
3. 主題與人體健康完全無關（如純美妝穿搭、旅遊、理財、生活雜談）。
4. 食譜製作/美食烹調日常：主要鏡頭在展示煮菜過程、食材下鍋、餐廳開箱或吃播，即使有旁白配音解釋營養成分，若非「全程單人對鏡科普講解」亦必須淘汰。

請以純 JSON 格式輸出：
{
  "qualified": true / false,
  "confidence_score": 0.0 ~ 1.0,
  "primary_topic": "簡短主題（如：心血管硬化預防 / 蛋白質吸收迷思）",
  "reasoning": "簡短繁體中文判定依據"
}
`;

export default async (req, context) => {
  console.log("--> [Netlify] 收到請求 Method:", req.method);

  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method Not Allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    const rawKey = process.env.GEMINI_API_KEY;
    if (!rawKey) {
      console.error("[-] 錯誤：找不到 GEMINI_API_KEY 環境變數！");
      return new Response(
        JSON.stringify({
          qualified: false,
          reasoning: "Netlify 伺服器缺少 GEMINI_API_KEY 設定",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }

    const apiKey = rawKey.trim();
    const bodyText = await req.text();
    console.log("--> [Netlify] Payload 長度:", bodyText.length);
    const { transcript_text, duration_sec } = JSON.parse(bodyText);

    // 使用官方 @google/genai SDK 初始化
    const ai = new GoogleGenAI({ apiKey: apiKey });

    const prompt = `
影片時長: ${duration_sec || 0} 秒
台詞逐字稿內容:
"""
${transcript_text || ""}
"""
請嚴格根據收錄準則評判這段內容是否符合訓練集標準。
`;

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: prompt,
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        responseMimeType: "application/json",
        temperature: 0.1,
      },
    });

    const replyText = response.text.trim();
    console.log("--> [Netlify] Gemini 評估成功:", replyText);

    return new Response(replyText, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (err) {
    console.error("[-] [Netlify] 執行錯誤:", err);
    return new Response(
      JSON.stringify({
        qualified: false,
        confidence_score: 0.0,
        primary_topic: "審核異常",
        reasoning: `Server Error: ${err.message}`,
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }
};