#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Reels Actionable Post-Production Diagnostic Tool
--------------------------------------------------------
功能：
1. 載入訓練好的 Transformer 模型與黃金基準常模 (Good Tier Benchmarks)。
2. 針對單支影片進行「秒級 (Second-by-Second)」掃描。
3. 自動定位異常時間段 (如第 6-8 秒)，並生成具體的繁體中文剪輯/後製建議：
   - 首 3 秒聲音與視覺 Hook 強化建議
   - 表情平淡區間自動標記 -> 建議插入 B-roll / 縮放特效
   - 局部秒數光影過硬或氣色偏冷 -> 建議後製調色與補光修正
   - 視聽重音脫節 -> 建議動態字幕補強
4. 繪製直觀的「秒級體檢儀表板圖 (Diagnostic Timeline Plot)」。
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import TRANSFORMER_DIR

# 匯入剛剛定義的模型結構
from train_transformer_engine import ReelsDiagnosticTransformer

# 設定中文字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ENGINE_DIR = TRANSFORMER_DIR / "transformer_engine"
DATASET_PATH = ENGINE_DIR / "transformer_multimodal_dataset.pt"
MODEL_PATH = ENGINE_DIR / "best_reels_transformer.pth"
STATS_PATH = ENGINE_DIR / "transformer_norm_stats.json"
REPORT_DIR = ENGINE_DIR / "diagnostic_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 建立 Good 組黃金基準線 (Golden Benchmarks)
# ---------------------------------------------------------------------------
def compute_golden_benchmarks(dataset: list) -> dict:
    """從所有 Good 級別影片中統計各項特徵的理想百分位數基準"""
    good_tensors = [r["time_series"] for r in dataset if r["tier_label"] == 2]
    all_good_frames = torch.cat(good_tensors, dim=0).numpy()

    # 欄位索引:
    # 0: colorfulness, 1: face_rms, 2: skin_warmth, 3: fg_bg_contrast,
    # 4: key_mus, 5: all_mus, 6: audio_energy, 7: semantic
    benchmarks = {
        "colorfulness_p40": float(np.percentile(all_good_frames[:, 0], 40)),
        "face_rms_p70": float(np.percentile(all_good_frames[:, 1], 70)),      # 越低越柔光，高於 P70 代表過硬
        "skin_warmth_p35": float(np.percentile(all_good_frames[:, 2], 35)),   # 低於 P35 代表氣色過冷慘白
        "key_mu_p30": float(np.percentile(all_good_frames[:, 4], 30)),        # 低於 P30 代表表情呆滯
        "semantic_hook_p50": float(np.percentile(all_good_frames[:, 7], 50)),
    }
    return benchmarks


# ---------------------------------------------------------------------------
# 2. 連續異常時間區間合併演算法 (Run-Length Interval Merger)
# ---------------------------------------------------------------------------
def find_continuous_intervals(mask_1d: np.ndarray, min_duration: int = 2) -> list[tuple[int, int]]:
    """
    找出布林陣列中連續為 True 且長度 >= min_duration 秒的區間
    回傳: [(start_sec, end_sec), ...]
    """
    intervals = []
    in_interval = False
    start = 0

    for t, val in enumerate(mask_1d):
        if val and not in_interval:
            in_interval = True
            start = t
        elif not val and in_interval:
            in_interval = False
            if (t - start) >= min_duration:
                intervals.append((start, t - 1))

    if in_interval and (len(mask_1d) - start) >= min_duration:
        intervals.append((start, len(mask_1d) - 1))

    return intervals


# ---------------------------------------------------------------------------
# 3. 核心：單支影片秒級診斷與後製建議生成引擎
# ---------------------------------------------------------------------------
def diagnose_single_video(record: dict, model: torch.nn.Module, norm_mean: torch.Tensor, norm_std: torch.Tensor, benchmarks: dict, device: torch.device) -> dict:
    raw_ts = record["time_series"][:120]  # (T, 8)
    T = raw_ts.size(0)
    raw_np = raw_ts.numpy()

    # 標準化並送入 Transformer
    norm_ts = ((raw_ts - norm_mean) / norm_std).unsqueeze(0).to(device)  # (1, T, 8)
    with torch.no_grad():
        logits, frame_risks, attn_weights = model(norm_ts, mask=None)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        # Attention 矩陣: (1, T, T) -> 計算每一秒被全片關注的活躍度 (Column Mean)
        attn_map = attn_weights.squeeze(0).cpu().numpy()
        temporal_importance = attn_map.mean(axis=0)  # (T,)

    pred_tier_idx = int(np.argmax(probs))
    tier_names = ["Bad (待加強)", "Neutral (中規中矩)", "Good (具備爆款潛力)"]
    good_prob = float(probs[2]) * 100.0
    bad_prob = float(probs[0]) * 100.0

    advice_list = []

    # =========================================================================
    # 診斷規則 A：首 3 秒黃金 Hook 診斷 (0 ~ 3s)
    # =========================================================================
    hook_end = min(3, T)
    hook_audio_max = np.max(raw_np[:hook_end, 6])
    hook_semantic_mean = np.mean(raw_np[:hook_end, 7])
    hook_color_mean = np.mean(raw_np[:hook_end, 0])

    if hook_audio_max < 0.6 and hook_semantic_mean < benchmarks["semantic_hook_p50"]:
        advice_list.append(
            f"🔊 【首 3 秒聲音 Hook 偏弱 (0s~{hook_end}s)】：開頭缺乏語音重音爆點且台詞吸睛度偏低（{hook_semantic_mean*100:.1f}分）。"
            f"建議在開頭前 2 秒加入轉場音效（如 Whoosh / 叮聲）、拉高第一句話的口播音量，或直接將最具爭議/痛點的金句剪到第 1 秒。"
        )
    elif hook_audio_max < 0.6:
        advice_list.append(
            f"🔊 【首 3 秒聽覺衝擊不足 (0s~{hook_end}s)】：台詞內容尚可，但聲音能量平淡。"
            f"建議後製在 0~2 秒處疊加背景重音特效（Sound Effect），抓住受眾聽覺注意力。"
        )

    if hook_color_mean < benchmarks["colorfulness_p40"]:
        advice_list.append(
            f"🎨 【首 3 秒視覺色彩單調 (0s~{hook_end}s)】：開頭畫面色彩豐富度（{hook_color_mean:.1f}）低於優質標準（{benchmarks['colorfulness_p40']:.1f}）。"
            f"建議在前 3 秒加上高對比醒目色塊標題字卡（如黃底黑字或紅框重點），防止受眾第一眼滑走。"
        )

    # =========================================================================
    # 診斷規則 B：中後段視覺疲勞與 B-roll 插入建議 (第 3 秒後)
    # 條件：微表情變化率 (key_mus) 低於 P30 且當秒無聲音爆點，持續 >= 3 秒
    # =========================================================================
    low_expression_mask = (raw_np[:, 4] < benchmarks["key_mu_p30"]) & (raw_np[:, 6] < 0.5)
    low_expression_mask[:3] = False  # 排除前 3 秒 (已由 Hook 診斷處理)
    broll_intervals = find_continuous_intervals(low_expression_mask, min_duration=3)

    for (st, ed) in broll_intervals[:3]:  # 最多列出前 3 個最需要 B-roll 的區間
        advice_list.append(
            f"🎬 【畫面動態疲乏區間 ({st}s~{ed}s，持續 {ed-st+1} 秒)】：偵測到此區段講者臉部微表情變化極小且無語音高潮，極易造成觀眾流失。"
            f"強烈建議在第 {st}～{ed} 秒覆蓋 B-roll 素材（如相關情境畫面、實操畫面、圖表字卡），或在剪輯時使用「微幅推鏡放大 (Zoom-in 10%~15%)」打破靜態沉悶感。"
        )

    # =========================================================================
    # 診斷規則 C：面部硬光陰影與膚色氣色異常掃描
    # =========================================================================
    harsh_light_mask = raw_np[:, 1] > benchmarks["face_rms_p70"]
    harsh_intervals = find_continuous_intervals(harsh_light_mask, min_duration=3)
    if harsh_intervals:
        st, ed = harsh_intervals[0]
        mean_rms = np.mean(raw_np[st:ed+1, 1])
        advice_list.append(
            f"💡 【面部光影過硬警告 ({st}s~{ed}s)】：臉部明暗對比度偏高（RMS={mean_rms:.3f} > 理想值 {benchmarks['face_rms_p70']:.3f}），出現陰陽臉或硬光陰影。"
            f"建議後製將該片段的「陰影 (Shadows)」拉亮 +15~25，下次拍攝時在正前方加裝柔光罩。"
        )

    cold_skin_mask = (raw_np[:, 2] < benchmarks["skin_warmth_p35"]) & (raw_np[:, 2] > 5.0)
    cold_intervals = find_continuous_intervals(cold_skin_mask, min_duration=4)
    if cold_intervals:
        st, ed = cold_intervals[0]
        mean_warmth = np.mean(raw_np[st:ed+1, 2])
        advice_list.append(
            f"🌡️ 【膚色氣色偏冷發灰 ({st}s~{ed}s)】：膚色紅潤指標（Cr-Cb={mean_warmth:.1f}）低於健康基準（{benchmarks['skin_warmth_p35']:.1f}）。"
            f"建議在剪輯軟體中將色溫（Temperature）往暖色微調 +5~10，提升講者的親和力與氣色血色感。"
        )

    # =========================================================================
    # 診斷規則 D：視聽重音與表情脫節檢查
    # =========================================================================
    audio_peak_secs = np.where(raw_np[:, 6] >= 0.8)[0]
    desync_secs = []
    for p_sec in audio_peak_secs:
        win_st = max(0, p_sec - 1)
        win_ed = min(T, p_sec + 2)
        if np.max(raw_np[win_st:win_ed, 4]) < benchmarks["key_mu_p30"]:
            desync_secs.append(int(p_sec))

    if desync_secs:
        sec_str = ", ".join([f"{s}s" for s in desync_secs[:4]])
        advice_list.append(
            f"⚡ 【視聽能量未同步 ({sec_str})】：系統在這些秒數偵測到語音重音爆點，但臉部眼神與嘴角缺乏對應的表情張力。"
            f"建議在 {sec_str} 的關鍵字字幕加上「彈跳放大特效」或高飽和變色，用視覺特效補足情緒張力。"
        )

    if not advice_list:
        advice_list.append("✅ 【整體表現優異】：全片光影柔和、氣色紅潤，且各秒數的視聽動態變化均維持在理想水準，可直接進入發布流程！")

    return {
        "account_id": record["account_id"],
        "video_id": record["video_id"],
        "duration": T,
        "true_tier": tier_names[record["tier_label"]],
        "pred_tier": tier_names[pred_tier_idx],
        "good_prob": good_prob,
        "bad_prob": bad_prob,
        "temporal_importance": temporal_importance,
        "raw_np": raw_np,
        "broll_intervals": broll_intervals,
        "advice_list": advice_list,
    }


# ---------------------------------------------------------------------------
# 4. 繪製「秒級 AI 診斷時間軸儀表板」
# ---------------------------------------------------------------------------
def plot_diagnostic_timeline(diag: dict, benchmarks: dict, save_path: Path):
    T = diag["duration"]
    secs = np.arange(T)
    raw_np = diag["raw_np"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), dpi=300, sharex=True)
    fig.suptitle(
        f"Reels 影片秒級後製 AI 診斷報告 | 帳號: {diag['account_id']} | 影片: {diag['video_id']}\n"
        f"AI 爆款潛力指數: {diag['good_prob']:.1f}% (劣質風險: {diag['bad_prob']:.1f}%)",
        fontsize=13, fontweight="bold", y=0.98
    )

    # 子圖 1: 臉部微表情張力 (QC Key Mu) & B-roll 建議區間標示
    ax1 = axes[0]
    ax1.plot(secs, raw_np[:, 4], color="#1f77b4", linewidth=2, marker="o", markersize=3, label="嘴角/眼周微表情形變 (QC Key Mu)")
    ax1.axhline(benchmarks["key_mu_p30"], color="red", linestyle="--", alpha=0.7, label="最低表情活躍門檻 (低於此線易沉悶)")
    for idx, (st, ed) in enumerate(diag["broll_intervals"]):
        label_str = "建議插入 B-roll / 縮放區間" if idx == 0 else None
        ax1.axvspan(st, ed, color="orange", alpha=0.35, label=label_str)
    ax1.set_ylabel("微表情動態強度", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_title("【維度一：微表情動態與 B-roll 切入點診斷】", fontsize=11, fontweight="bold", loc="left")

    # 子圖 2: 雙流音訊能量與 Transformer 時序注意力曲線
    ax2 = axes[1]
    ax2.bar(secs, raw_np[:, 6], color="#2ca02c", alpha=0.4, width=0.6, label="語音重音峰值脈衝 (Audio Peaks)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(secs, diag["temporal_importance"], color="#9467bd", linewidth=2, linestyle="-", label="Transformer 時序關注權重")
    ax2.axvspan(0, min(3, T-1), color="gold", alpha=0.2, label="黃金前 3 秒 Hook 區")
    ax2.set_ylabel("聲學能量脈衝", fontweight="bold", color="#2ca02c")
    ax2_twin.set_ylabel("AI 注意力權重", fontweight="bold", color="#9467bd")
    ax2.legend(loc="upper left", fontsize=9)
    ax2_twin.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_title("【維度二：黃金 3 秒 Hook 聲能與 Transformer 注意力曲線】", fontsize=11, fontweight="bold", loc="left")

    # 子圖 3: 面部柔光對比度 (RMS) 與膚色血色暖感 (Cr-Cb)
    ax3 = axes[2]
    ax3.plot(secs, raw_np[:, 1], color="#d62728", linewidth=1.8, label="面部光影 RMS 對比度 (越低越柔和)")
    ax3.axhline(benchmarks["face_rms_p70"], color="#d62728", linestyle="--", alpha=0.6, label="硬光陰影警戒線")
    ax3.set_ylabel("面部 RMS 對比", fontweight="bold", color="#d62728")
    ax3.set_xlabel("影片時間軸 (秒 Seconds)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.set_title("【維度三：逐秒打光柔和度監測 (高於紅虛線代表出現陰陽臉/頂光陰影)】", fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. 主程序：示範對影片進行完整秒級體檢
# ---------------------------------------------------------------------------
def main():
    print("=" * 85)
    print("【Reels 影片後製前 AI 秒級診斷工具 (Actionable Post-Production Advisor)】")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 載入資料集與正規化參數
    dataset = torch.load(DATASET_PATH, weights_only=False)
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32)

    # 2. 計算 Good 組黃金常模基準
    benchmarks = compute_golden_benchmarks(dataset)
    print("[*] 成功載入 1,339 支影片黃金基準常模 (Good Tier Benchmarks)：")
    for k, v in benchmarks.items():
        print(f"    - {k}: {v:.4f}")

    # 3. 載入訓練好的 Transformer 模型
    model = ReelsDiagnosticTransformer(
        vis_dim=4, geom_dim=2, aud_dim=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=3
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 4. 挑選 3 支不同特性的樣本影片進行實戰體檢示範
    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]

    for idx in sample_indices:
        record = dataset[idx]
        diag = diagnose_single_video(record, model, norm_mean, norm_std, benchmarks, device)

        print("\n" + "=" * 85)
        print(f"🎬 診斷影片: [{diag['account_id']}] - {diag['video_id']} (片長: {diag['duration']} 秒)")
        print(f"📊 歷史真實表現: {diag['true_tier']} | AI 潛力評估: 爆款機率 {diag['good_prob']:.1f}% / 劣質風險 {diag['bad_prob']:.1f}%")
        print("-" * 85)
        print("🛠️ 【AI 後製剪輯具體修改建議】：")
        for i, adv in enumerate(diag["advice_list"], 1):
            print(f"  {i}. {adv}")

        plot_file = REPORT_DIR / f"diagnosis_{diag['account_id']}_{diag['video_id']}.png"
        plot_diagnostic_timeline(diag, benchmarks, plot_file)
        print(f"\n📈 秒級視覺化診斷圖表已生成: {plot_file}")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    main()