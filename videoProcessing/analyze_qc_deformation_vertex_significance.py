import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from config import VIDEO_PROC_DIR

DEFAULT_JSON_PATH = str(VIDEO_PROC_DIR / "qc_deformation_symmetric_stats.json")
OUTPUT_CSV_DIR = str(VIDEO_PROC_DIR / "vertex_significance_reports")

METRICS = ["abs_mu", "dilation", "teichmuller_distance"]
FIELDS = ["mean", "max", "min", "std"]

# 關鍵肌肉區域解剖對照表
ANATOMICAL_LABELS = {
    1: "鼻尖", 6: "鼻樑中央", 168: "印堂", 9: "額頭中央", 152: "下巴下緣",
    70: "右皺眉肌 (Corrugator)", 63: "右眉中", 105: "右眉峰",
    300: "左皺眉肌 (Corrugator)", 293: "左眉中", 334: "左眉峰",
    130: "右外眼角 (Duchenne微笑)", 246: "右眼外周緣",
    359: "左外眼角 (Duchenne微笑)", 466: "左眼外周緣",
    0: "上唇正中 (說話)", 17: "下唇正中 (說話)",
    61: "右嘴角 (Zygomaticus微笑)", 291: "左嘴角 (Zygomaticus微笑)",
    13: "上唇內咬合緣", 14: "下唇內咬合緣",
    234: "右臉頰外側", 454: "左臉頰外側",
    172: "右下頜角 (咬合肌)", 397: "左下頜角 (咬合肌)",
}


def test_vertices_significance(json_path: str, metric: str = "abs_mu", field: str = "mean") -> dict:
    if not os.path.exists(json_path):
        print(f"[錯誤] 找不到檔案 {json_path}，請先執行 analyze_qc_deformation_with_heatmap_symmetric.py。")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tiers = data.get("tiers", {})
    if "good" not in tiers or "bad" not in tiers:
        print("[錯誤] JSON 中未找到 'good' 或 'bad' 的專屬資料塊！")
        return None

    n_good = data["metadata"]["tier_video_counts"]["good"]
    n_bad = data["metadata"]["tier_video_counts"]["bad"]

    results = []
    for v_idx in range(468):
        v_str = str(v_idx)
        g_info = tiers["good"]["per_vertex_details"][v_str][metric][field]
        b_info = tiers["bad"]["per_vertex_details"][v_str][metric][field]

        m1, s1 = g_info["mean"], g_info["std"]
        m2, s2 = b_info["mean"], b_info["std"]

        se_diff = np.sqrt((s1 ** 2 / n_good) + (s2 ** 2 / n_bad))
        if se_diff < 1e-10:
            continue

        t_val = (m1 - m2) / se_diff
        df_num = ((s1 ** 2 / n_good) + (s2 ** 2 / n_bad)) ** 2
        df_denom = ((s1 ** 2 / n_good) ** 2 / (n_good - 1)) + ((s2 ** 2 / n_bad) ** 2 / (n_bad - 1))
        df_val = df_num / df_denom if df_denom > 0 else 1.0

        p_val = 2.0 * stats.t.sf(np.abs(t_val), df=df_val)

        s_pooled = np.sqrt(((n_good - 1) * s1 ** 2 + (n_bad - 1) * s2 ** 2) / (n_good + n_bad - 2))
        d_val = (m1 - m2) / s_pooled if s_pooled > 0 else 0.0

        results.append({
            "vertex_id": v_idx,
            "anatomy": ANATOMICAL_LABELS.get(v_idx, "一般皮膚網格點"),
            "good_mean": m1,
            "good_std": s1,
            "bad_mean": m2,
            "bad_std": s2,
            "diff (G-B)": m1 - m2,
            "t_stat": t_val,
            "p_value": p_val,
            "cohens_d": d_val,
        })

    df = pd.DataFrame(results).sort_values(by="p_value").reset_index(drop=True)
    # Benjamini-Hochberg FDR 多重檢定校正
    m_tests = len(df)
    df["rank"] = np.arange(1, m_tests + 1)
    df["fdr_q"] = np.minimum(1.0, df["p_value"] * m_tests / df["rank"])
    df["fdr_q"] = np.minimum.accumulate(df["fdr_q"][::-1])[::-1]

    sig_raw = df[df["p_value"] < 0.05]
    sig_fdr = df[df["fdr_q"] < 0.05]

    print("\n" + "=" * 85)
    print(f"【全臉 468 點 Good (N={n_good}) vs Bad (N={n_bad}) 幾何顯著性檢定】")
    print(f"  - 檢定指標: [{metric}] | 統計量: [{field}]")
    print(f"  - 原始 p < 0.05 顯著頂點數:       {len(sig_raw)} / {m_tests} 個")
    print(f"  - FDR 校正後 (q < 0.05) 顯著頂點: {len(sig_fdr)} / {m_tests} 個")
    print("=" * 85)

    if not df.empty:
        print("\nTop 10 差異最顯著的頂點：")
        cols = ["vertex_id", "anatomy", "good_mean", "bad_mean", "diff (G-B)", "t_stat", "p_value", "fdr_q", "cohens_d"]
        print(df.head(10)[cols].to_string(index=False))

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_CSV_DIR, f"vertex_significance_{metric}_{field}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[*] 完整 468 點統計檢定表已匯出至: {os.path.abspath(out_csv)}")

    top_row = df.iloc[0] if not df.empty else None
    return {
        "metric": metric,
        "field": field,
        "total_tested_vertices": m_tests,
        "sig_raw_p05_count": len(sig_raw),
        "sig_fdr_q05_count": len(sig_fdr),
        "top_vertex_id": int(top_row["vertex_id"]) if top_row is not None else None,
        "top_anatomy": top_row["anatomy"] if top_row is not None else "",
        "top_diff_G_minus_B": round(float(top_row["diff (G-B)"]), 6) if top_row is not None else 0.0,
        "top_p_value": float(top_row["p_value"]) if top_row is not None else 1.0,
        "top_fdr_q": float(top_row["fdr_q"]) if top_row is not None else 1.0,
        "top_cohens_d": round(float(top_row["cohens_d"]), 4) if top_row is not None else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 468 頂點 Good vs Bad 顯著性檢定與 FDR 校正（預設分析全部組合）")
    parser.add_argument("--json", type=str, default=DEFAULT_JSON_PATH, help="對稱統計 JSON 路徑")
    parser.add_argument("--metric", type=str, default="abs_mu", choices=METRICS, help="指定單一幾何指標（需配合 --single）")
    parser.add_argument("--field", type=str, default="mean", choices=FIELDS, help="指定單一統計量（需配合 --single）")
    parser.add_argument("--all", action="store_true", default=True, help="預設分析全部 3 指標 x 4 統計量 = 12 種組合")
    parser.add_argument("--single", action="store_true", help="僅分析由 --metric 與 --field 指定的單一組合")
    args = parser.parse_args()

    if args.all and not args.single:
        print("=" * 85)
        print("🚀 開始批次執行全部 12 種組合 (3 指標 x 4 統計量) 的全臉 468 點顯著性檢定...")
        print("=" * 85)
        overview_rows = []
        for m_name in METRICS:
            for f_name in FIELDS:
                res = test_vertices_significance(args.json, metric=m_name, field=f_name)
                if res:
                    overview_rows.append(res)

        if overview_rows:
            os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
            overview_df = pd.DataFrame(overview_rows)
            overview_csv = os.path.join(OUTPUT_CSV_DIR, "all_combinations_significance_overview.csv")
            overview_df.to_csv(overview_csv, index=False, encoding="utf-8-sig")

            print("\n" + "=" * 95)
            print("📊 【全部 12 種組合顯著性檢定總覽表】")
            print("=" * 95)
            print(overview_df[[
                "metric", "field", "sig_raw_p05_count", "sig_fdr_q05_count",
                "top_vertex_id", "top_anatomy", "top_fdr_q", "top_cohens_d"
            ]].to_string(index=False))
            print(f"\n[*] 總覽彙總表已儲存至: {os.path.abspath(overview_csv)}")
    else:
        test_vertices_significance(args.json, metric=args.metric, field=args.field)
