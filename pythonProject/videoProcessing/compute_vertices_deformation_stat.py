import json
import numpy as np
import pandas as pd
from scipy import stats

JSON_PATH = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\videoProcessing\all_vertices_deformation_stats_symmetric.json"


def test_vertices_significance(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tiers = data.get("tiers", {})
    if "good" not in tiers or "bad" not in tiers:
        print("[錯誤] JSON 中未找到 'good' 或 'bad' 的專屬資料塊，請確認檔案內容是否完整！")
        return

    n_good = data["metadata"]["tier_video_counts"]["good"]
    n_bad = data["metadata"]["tier_video_counts"]["bad"]

    metric = "abs_mu"
    field = "max"  # 檢視最大形變峰值

    results = []
    for v_idx in range(468):
        v_str = str(v_idx)

        # 讀取 Good 與 Bad 的 mean 與 std
        g_info = tiers["good"]["per_vertex_details"][v_str][metric][field]
        b_info = tiers["bad"]["per_vertex_details"][v_str][metric][field]

        m1, s1 = g_info["mean"], g_info["std"]
        m2, s2 = b_info["mean"], b_info["std"]

        # 計算 Welch's t-test
        se_diff = np.sqrt((s1 ** 2 / n_good) + (s2 ** 2 / n_bad))
        if se_diff < 1e-8:
            continue

        t_val = (m1 - m2) / se_diff
        df_num = ((s1 ** 2 / n_good) + (s2 ** 2 / n_bad)) ** 2
        df_denom = ((s1 ** 2 / n_good) ** 2 / (n_good - 1)) + ((s2 ** 2 / n_bad) ** 2 / (n_bad - 1))
        df_val = df_num / df_denom if df_denom > 0 else 1.0

        p_val = 2.0 * stats.t.sf(np.abs(t_val), df=df_val)

        # Cohen's d
        s_pooled = np.sqrt(((n_good - 1) * s1 ** 2 + (n_bad - 1) * s2 ** 2) / (n_good + n_bad - 2))
        d_val = (m1 - m2) / s_pooled if s_pooled > 0 else 0.0

        results.append({
            "vertex_id": v_idx,
            "good_mean": m1,
            "bad_mean": m2,
            "diff (G-B)": m1 - m2,
            "t_stat": t_val,
            "p_value": p_val,
            "cohens_d": d_val
        })

    df = pd.DataFrame(results)
    # FDR (Benjamini-Hochberg) 校正
    df["rank"] = df["p_value"].rank()
    df["fdr_q"] = df["p_value"] * len(df) / df["rank"]
    df["fdr_q"] = np.minimum.accumulate(df["fdr_q"][::-1])[::-1]

    sig_raw = df[df["p_value"] < 0.05]
    sig_fdr = df[df["fdr_q"] < 0.05]

    print("=" * 70)
    print("【全臉 468 點 Good vs Bad 幾何顯著性檢定結果】")
    print(f"  - 原始 p < 0.05 的頂點數量:  {len(sig_raw)} / 468 個")
    print(f"  - 經 FDR 校正後顯著的頂點數量: {len(sig_fdr)} / 468 個")
    print("=" * 70)

    if not sig_raw.empty:
        print("\n前 10 個差異最顯著的頂點：")
        print(df.sort_values(by="p_value").head(10)[
                  ["vertex_id", "good_mean", "bad_mean", "diff (G-B)", "p_value", "cohens_d"]].to_string(index=False))


if __name__ == "__main__":
    test_vertices_significance(JSON_PATH)