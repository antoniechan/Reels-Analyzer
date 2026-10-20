import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_REELS_DIR = r"D:\reels"
DEFAULT_AUDIO_JSON = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\audioProcessing\local_transcript_audio_saliency_summary_all_tiers.json"
DEFAULT_CLASS_JSON = r"C:\Users\Anthony\Documents\GitHub\reelsAnalyzer\pythonProject\dataProcessing\video_classification_summary.json"
OUTPUT_DIR = r".\crossModalAnalysis"

# 口周與眼周表情關鍵點（嘴角提肌 61, 291；口輪匝肌 0, 13, 14, 17；眼角 33, 263, 130, 359）
EXPRESSION_KEY_VERTICES = set([0, 13, 14, 17, 61, 291, 33, 263, 130, 359])
TIER_ORDER = ["good", "neutral", "bad"]
TIER_COLORS = {"good": "#2ca02c", "neutral": "#ff7f0e", "bad": "#d62728"}


def build_qc_lookup_dict(root_dir: str) -> dict:
    """
    掃描 D:\reels 下所有的 *_qc.json，建立快速索引字典：
    Key: (account_id_lower, video_stem_lower) -> Value: 完整檔案路徑
    """
    print(f"正在全盤掃描 {root_dir} 下的所有 *_qc.json 檔案...")
    pattern = os.path.join(root_dir, "**", "*_qc.json")
    all_qc_files = glob.glob(pattern, recursive=True)
    print(f"磁碟中實際找到 {len(all_qc_files)} 個 QC 檔案。正在建立雙向索引表...")

    lookup = {}
    for orig_p in all_qc_files:
        norm_p = orig_p.replace("\\", "/")
        parts = norm_p.split("/")
        fname = parts[-1].lower()
        vid_stem = fname[:-8] if fname.endswith("_qc.json") else os.path.splitext(fname)[0]

        # 將路徑上所有上層資料夾名稱都註冊為潛在的 account_id，徹底免疫多層子目錄問題
        for folder_part in parts[:-1]:
            acc_cand = folder_part.lower().strip()
            if acc_cand and acc_cand not in ["d:", "reels"]:
                lookup[(acc_cand, vid_stem)] = orig_p
                # 同步註冊去除常見後綴的 stem
                for sfx in ["_processed", "_face", "_aligned"]:
                    if sfx in vid_stem:
                        lookup[(acc_cand, vid_stem.replace(sfx, ""))] = orig_p

    return lookup


def load_alias_map(class_json_path: str) -> dict:
    """
    讀取 video_classification_summary.json，建立 (account_id, video_id) -> local_index_name 的映射
    """
    alias_map = {}
    if not class_json_path or not os.path.exists(class_json_path):
        return alias_map
    try:
        with open(class_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        videos = data.get("videos", data) if isinstance(data, dict) else data
        for item in videos:
            if not isinstance(item, dict):
                continue
            acc = str(item.get("account_id", "")).lower().strip()
            vid = str(item.get("video_id", "")).lower().strip()
            loc = str(item.get("local_index_name", "")).lower().strip()
            if acc and vid and loc:
                alias_map[(acc, vid)] = loc
    except Exception as e:
        print(f"  [提示] 無法讀取分類別名表 ({e})，將直接使用 audio_json 內的名稱。")
    return alias_map


def extract_facial_deformation_timeseries(qc_json_path: str, target_duration_sec: int):
    """
    從 *_qc.json 提取每一轉場影格的臉部形變值，
    並透過線性插值對齊到 target_duration_sec (每秒 1 點)，
    解決不同抽幀率 (FPS) 導致除以 30 後秒數過短的問題！
    """
    try:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = json.load(f)
    except Exception:
        return None

    transitions = qc_data.get("consecutive_qc_metrics", [])
    if not transitions or len(transitions) < 3:
        return None

    raw_frame_series = []
    for trans in transitions:
        if not trans.get("valid_transition", False):
            raw_frame_series.append(np.nan)
            continue

        facet_metrics = trans.get("facet_metrics", [])
        if not facet_metrics:
            raw_frame_series.append(np.nan)
            continue

        # 優先提取表情關鍵點面片，若無則提取全臉有效面片均值
        key_mus = []
        all_mus = []
        for fm in facet_metrics:
            mu_val = float(fm.get("abs_mu", 0.0))
            if mu_val > 1e-6:
                all_mus.append(mu_val)
                verts = fm.get("vertices", [])
                if any(v in EXPRESSION_KEY_VERTICES for v in verts):
                    key_mus.append(mu_val)

        if key_mus:
            raw_frame_series.append(float(np.mean(key_mus)))
        elif all_mus:
            raw_frame_series.append(float(np.mean(all_mus)))
        else:
            raw_frame_series.append(np.nan)

    arr = np.array(raw_frame_series, dtype=np.float64)
    valid_mask = ~np.isnan(arr)
    if np.sum(valid_mask) < 3:
        return None

    # 填補中間缺失幀 (插值)
    x_all = np.arange(len(arr))
    arr_filled = np.interp(x_all, x_all[valid_mask], arr[valid_mask])

    # 將整個序列均勻重採樣至 target_duration_sec 長度 (1 Hz)
    target_x = np.linspace(0, len(arr_filled) - 1, num=target_duration_sec)
    resampled_ts = np.interp(target_x, x_all, arr_filled)

    return resampled_ts


def compute_cross_modal_metrics(audio_saliency_series: np.ndarray, face_series: np.ndarray):
    """
    計算單部影片的視聽協同對齊指標 (Pearson r, 最優延遲 Lag, 峰值重合率)
    """
    min_len = min(len(audio_saliency_series), len(face_series))
    if min_len < 5:
        return None

    a = audio_saliency_series[:min_len].astype(np.float64)
    f = face_series[:min_len].astype(np.float64)

    if np.std(a) < 1e-8 or np.std(f) < 1e-8:
        return None

    # 1. 基礎相關係數
    r0, p0 = stats.pearsonr(a, f)
    rho0, sp_p0 = stats.spearmanr(a, f)

    # 2. 互相關與最優延遲時間 (最佳平移窗口 ±5 秒)
    max_lag = min(5, min_len // 3)
    a_norm = (a - np.mean(a)) / (np.std(a) * min_len)
    f_norm = (f - np.mean(f)) / np.std(f)
    cc = signal.correlate(f_norm, a_norm, mode="full")
    lags = signal.correlation_lags(len(f_norm), len(a_norm), mode="full")

    valid_mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_cc = cc[valid_mask]
    valid_lags = lags[valid_mask]

    best_idx = int(np.argmax(valid_cc))
    max_corr = float(valid_cc[best_idx])
    best_lag = int(valid_lags[best_idx])

    # 3. 峰值時間重合率 (容許 ±1 秒誤差窗口)
    a_th = np.percentile(a, 75)
    f_th = np.percentile(f, 75)
    a_peaks = np.where(a >= a_th)[0]
    f_peaks = set(np.where(f >= f_th)[0])

    coincidence_count = sum(
        1 for ap in a_peaks if any((ap + delta) in f_peaks for delta in [-1, 0, 1])
    )
    peak_coincidence_rate = coincidence_count / len(a_peaks) if len(a_peaks) > 0 else 0.0

    return {
        "sync_pearson_r": float(r0),
        "sync_pearson_p": float(p0),
        "sync_spearman_rho": float(rho0),
        "optimal_lag_seconds": float(best_lag),
        "optimal_lag_max_corr": float(max_corr),
        "peak_coincidence_rate": float(peak_coincidence_rate),
    }


def run_cross_modal_experiment(
    audio_summary_json: str,
    root_dir: str = DEFAULT_REELS_DIR,
    output_dir: str = OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("【Cross-Modal Saliency: 視聽時空協同對齊分析系統】")
    print(f"  - 音訊彙總檔: {audio_summary_json}")
    print(f"  - Reels 目錄: {root_dir}")
    print("=" * 80 + "\n")

    if not os.path.exists(audio_summary_json):
        raise FileNotFoundError(f"找不到音訊摘要檔案: {audio_summary_json}")

    with open(audio_summary_json, "r", encoding="utf-8") as f:
        audio_data = json.load(f)

    # 嘗試載入 classification_source 以獲取 local_index_name 別名映射
    class_src = audio_data.get("classification_source", DEFAULT_CLASS_JSON)
    alias_map = load_alias_map(class_src)
    qc_lookup = build_qc_lookup_dict(root_dir)

    video_items = audio_data.get("videos", [])
    records = []

    # 診斷統計計數器
    cnt_total = len(video_items)
    cnt_path_found = 0
    cnt_face_ts_ok = 0
    cnt_metrics_ok = 0

    print(f"開始進行 {cnt_total} 支影片的視聽對齊配對...")

    for idx, v_info in enumerate(video_items, 1):
        tier = str(v_info.get("video_classification", "")).lower()
        if tier not in TIER_ORDER:
            continue

        acc_id = str(v_info.get("account_id", "")).strip()
        vid_id = str(v_info.get("video_id", "")).strip()
        acc_low, vid_low = acc_id.lower(), vid_id.lower()

        duration = int(v_info.get("duration_seconds") or 0)
        top_peaks = v_info.get("top_peak_seconds", [])

        if duration < 5:
            continue

        # 1. 查找 QC 路徑 (先查 video_id，再查 local_index_name 別名)
        qc_path = qc_lookup.get((acc_low, vid_low))
        if not qc_path and (acc_low, vid_low) in alias_map:
            alt_vid = alias_map[(acc_low, vid_low)]
            qc_path = qc_lookup.get((acc_low, alt_vid))

        if not qc_path:
            continue
        cnt_path_found += 1

        # 2. 提取並重採樣臉部形變時序至 duration 長度
        face_ts = extract_facial_deformation_timeseries(qc_path, target_duration_sec=duration)
        if face_ts is None:
            continue
        cnt_face_ts_ok += 1

        # 3. 構建音訊 Saliency 曲線
        base_score = float(v_info.get("mean_engagement_score") or 0.3)
        max_score = float(v_info.get("max_engagement_score") or 0.8)
        audio_ts = np.full(duration, base_score, dtype=np.float64)
        for p_sec in top_peaks:
            p_idx = int(p_sec) - 1 if int(p_sec) >= duration else int(p_sec)
            if 0 <= p_idx < duration:
                audio_ts[p_idx] = max_score
                for delta in [-1, 1]:
                    if 0 <= p_idx + delta < duration:
                        audio_ts[p_idx + delta] = max(audio_ts[p_idx + delta], base_score + 0.6 * (max_score - base_score))

        # 若 top_peaks 剛好為空，加入微小斜率避免 std == 0
        if np.std(audio_ts) < 1e-7:
            audio_ts[0] = float(v_info.get("hook_first_5s_score") or (base_score + 0.05))

        # 4. 計算跨模態同步指標
        cm_metrics = compute_cross_modal_metrics(audio_ts, face_ts)
        if cm_metrics is None:
            continue
        cnt_metrics_ok += 1

        records.append({
            "account_id": acc_id,
            "video_id": vid_id,
            "tier": tier,
            "video_quality_score": float(v_info.get("video_quality_score") or 0.0),
            **cm_metrics
        })

    print("\n" + "-" * 50)
    print("【配對診斷統計】")
    print(f"  - JSON 總影片數:         {cnt_total}")
    print(f"  - 成功找到 QC 檔案數:    {cnt_path_found}")
    print(f"  - 成功提取臉部時序數:    {cnt_face_ts_ok}")
    print(f"  - 成功計算跨模態指標數:  {cnt_metrics_ok}")
    print("-" * 50 + "\n")

    df = pd.DataFrame(records)
    if len(df) < 10:
        print(f"[錯誤] 有效樣本數過少 ({len(df)} 支)，無法進行統計檢定。")
        return

    tier_counts = {t: len(df[df["tier"] == t]) for t in TIER_ORDER}
    print(f"各 Tier 樣本數量: {tier_counts}\n")

    cm_fields = [
        ("sync_pearson_r", "視聽同步相關性 (Cross-Modal Correlation r)", "衡量聲音高潮與表情形變的同頻程度"),
        ("peak_coincidence_rate", "顯著性峰值重合率 (Peak Coincidence %)", "衡量重音/痛點時表情同步爆發的比例"),
        ("optimal_lag_max_corr", "最佳延遲下的最大相關性 (Max Lagged Corr)", "允許時間平移後的潛在耦合度"),
        ("optimal_lag_seconds", "最優視聽延遲時間 (Optimal Lag s)", "臉部表情落後(+)或超前(-)於聲音的秒數"),
    ]

    report_lines = []
    report_lines.append("=" * 85)
    report_lines.append("【Cross-Modal Saliency 跨模態視聽協同統計檢定報告】")
    report_lines.append("=" * 85)
    for t in TIER_ORDER:
        report_lines.append(f"  - {t.upper()} Tier: {tier_counts[t]} 支影片")
    report_lines.append("-" * 85 + "\n")

    for m_key, m_title, m_desc in cm_fields:
        report_lines.append(f"▶ 指標：{m_title} — {m_desc}")
        report_lines.append("-" * 75)

        g_vals = df[df["tier"] == "good"][m_key].dropna().values
        n_vals = df[df["tier"] == "neutral"][m_key].dropna().values
        b_vals = df[df["tier"] == "bad"][m_key].dropna().values

        for t, vals in [("good", g_vals), ("neutral", n_vals), ("bad", b_vals)]:
            m = np.mean(vals) if len(vals) > 0 else 0.0
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            med = np.median(vals) if len(vals) > 0 else 0.0
            report_lines.append(f"  [{t.upper():<7}] Mean = {m:.4f} ± {s:.4f} | Median = {med:.4f}")

        if min(len(g_vals), len(n_vals), len(b_vals)) >= 2:
            f_stat, p_anova = stats.f_oneway(g_vals, n_vals, b_vals)
            t_gb, p_welch = stats.ttest_ind(g_vals, b_vals, equal_var=False)
            u_gb, p_mann = stats.mannwhitneyu(g_vals, b_vals, alternative="two-sided")

            nx, ny = len(g_vals), len(b_vals)
            pooled_std = np.sqrt(((nx - 1) * np.var(g_vals, ddof=1) + (ny - 1) * np.var(b_vals, ddof=1)) / (nx + ny - 2))
            d_val = (np.mean(g_vals) - np.mean(b_vals)) / pooled_std if pooled_std > 0 else 0.0

            report_lines.append(f"  [跨組 ANOVA]     F = {f_stat:.4f}, p = {p_anova:.4e}")
            report_lines.append(f"  [GOOD vs. BAD]   Welch's t = {t_gb:.4f}, p = {p_welch:.4e} | Cohen's d = {d_val:.4f}")
            report_lines.append(f"                   Mann-Whitney U = {u_gb:.1f}, p = {p_mann:.4e}")
        else:
            report_lines.append("  [提示] 某組樣本數不足，已跳過假設檢定。")

        report_lines.append("")

    report_path = os.path.join(output_dir, "cross_modal_saliency_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    df_csv = os.path.join(output_dir, "cross_modal_dataset.csv")
    df.to_csv(df_csv, index=False, encoding="utf-8-sig")

    print("\n".join(report_lines))
    print(f"報告已存至: {os.path.abspath(report_path)}")
    print(f"資料集已存至: {os.path.abspath(df_csv)}")

    # 繪製品質分數與對齊指標視覺化圖表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    sns.boxplot(
        data=df,
        x="tier",
        y="peak_coincidence_rate",
        order=TIER_ORDER,
        palette=TIER_COLORS,
        width=0.45,
        fliersize=0,
        boxprops=dict(alpha=0.75),
        ax=axes[0],
    )
    sns.stripplot(
        data=df,
        x="tier",
        y="peak_coincidence_rate",
        order=TIER_ORDER,
        color="black",
        alpha=0.2,
        jitter=0.2,
        size=4,
        ax=axes[0],
    )
    axes[0].set_title("視聽顯著性峰值重合率\n(Audio & Facial Peak Coincidence)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Coincidence Rate [0 ~ 1]", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    sns.scatterplot(
        data=df,
        x="video_quality_score",
        y="sync_pearson_r",
        hue="tier",
        palette=TIER_COLORS,
        alpha=0.5,
        s=30,
        ax=axes[1],
    )
    sns.regplot(
        data=df,
        x="video_quality_score",
        y="sync_pearson_r",
        scatter=False,
        color="darkblue",
        line_kws={"linewidth": 2.0, "linestyle": "--"},
        ax=axes[1],
    )
    r_val, p_val = stats.pearsonr(df["video_quality_score"], df["sync_pearson_r"])
    axes[1].set_title(f"影片品質得分 vs 視聽同步相關性\nPearson r = {r_val:.4f} (p = {p_val:.4e})", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Sync Pearson r", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_png = os.path.join(output_dir, "cross_modal_alignment_comparison.png")
    plt.savefig(plot_png, bbox_inches="tight")
    plt.close()
    print(f"對比圖表已生成至: {os.path.abspath(plot_png)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 Cross-Modal Saliency 視聽協同對齊分析")
    parser.add_argument(
        "--audio_json",
        type=str,
        default=DEFAULT_AUDIO_JSON,
        help="音訊顯著性 JSON 檔路徑",
    )
    parser.add_argument("--dir", type=str, default=DEFAULT_REELS_DIR, help="Reels 根目錄路徑")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR, help="結果輸出目錄")
    args = parser.parse_args()

    run_cross_modal_experiment(
        audio_summary_json=args.audio_json,
        root_dir=args.dir,
        output_dir=args.out_dir,
    )