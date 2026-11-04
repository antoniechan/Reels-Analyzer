import { GoogleGenerativeAI } from "@google/generative-ai";

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

請以純 JSON 格式輸出：
{
  "qualified": true / false,
  "confidence_score": 0.0 ~ 1.0,
  "primary_topic": "簡短主題（如：心血管硬化預防 / 蛋白質吸收迷思）",
  "reasoning": "簡短繁體中文判定依據"
}
`;

export default async (req, context) => {
  console.log("--> 收到請求 Method:", req.method);

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
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.error("[-] 錯誤：找不到 GEMINI_API_KEY 環境變數！");
      return new Response(
        JSON.stringify({
          qualified: false,
          reasoning: "Netlify 伺服器缺少 GEMINI_API_KEY 設定",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }

    const bodyText = await req.text();
    console.log("--> 接收到 Payload 長度:", bodyText.length);
    const { transcript_text, duration_sec } = JSON.parse(bodyText);

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: "gemini-3.8-flash",
      systemInstruction: SYSTEM_INSTRUCTION,
      generationConfig: {
        responseMimeType: "application/json",
        temperature: 0.1,
      },
    });

    const prompt = `
影片時長: ${duration_sec || 0} 秒
台詞逐字稿內容:
"""
${transcript_text || ""}
"""
請嚴格根據收錄準則評判這段內容是否符合訓練集標準。
`;

    const result = await model.generateContent(prompt);
    const replyText = result.response.text();
    console.log("--> Gemini 回應成功:", replyText);

    return new Response(replyText, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (err) {
    console.error("[-] 執行錯誤:", err);
    return new Response(
      JSON.stringify({
        qualified: false,
        confidence_score: 0.0,
        primary_topic: "審核異常",
        reasoning: err.message,
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