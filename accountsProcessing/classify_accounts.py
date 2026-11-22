import glob
import json
import os
import pandas as pd
from config import ACCOUNT_DATA_DIR, ACCOUNT_PROC_DIR

ACCOUNTS_DATA_DIR = ACCOUNT_DATA_DIR
OUTPUT_SUMMARY_JSON = str(ACCOUNT_PROC_DIR / "account_classification_summary.json")
OUTPUT_SUMMARY_CSV = str(ACCOUNT_PROC_DIR / "account_classification_summary.csv")


def classify_accounts_data():
  json_pattern = os.path.join(ACCOUNTS_DATA_DIR, "*.json")
  json_files = glob.glob(json_pattern)

  if not json_files:
    print(f"錯誤：在 {ACCOUNTS_DATA_DIR} 中未找到任何 JSON 檔案。")
    return

  records = []

  for fpath in json_files:
    fname = os.path.basename(fpath)
    if fname in [
        os.path.basename(OUTPUT_SUMMARY_JSON),
        "accounts_followers.json",
        "account_followers.json",
    ]:
      continue

    try:
      with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as e:
      print(f"讀取失敗 {fname}: {e}")
      continue

    account_id = data.get("account_id", os.path.splitext(fname)[0])
    metrics = data.get("metrics", {})
    med_stats = data.get("median_stats", {})

    vfr = metrics.get("VFR")
    erv = metrics.get("ERV")
    clr = metrics.get("CLR")
    srv = metrics.get("SRV")

    # 若舊 JSON 尚未計算 SRV，自動從 median_stats 補算
    if srv is None:
      med_v = med_stats.get("median_views", 0.0)
      med_s = med_stats.get("median_shares", 0.0)
      srv = float(med_s / (med_v + 1e-6)) if med_v > 0 else 0.0

    records.append({
        "file_path": fpath,
        "account_id": account_id,
        "followers": data.get("followers"),
        "total_videos": data.get("total_videos_analyzed", 0),
        "VFR": vfr if vfr is not None else 0.0,
        "ERV": erv if erv is not None else 0.0,
        "CLR": clr if clr is not None else 0.0,
        "SRV": srv if srv is not None else 0.0,
        "SLR": metrics.get("SLR", 0.0),
        "CV_views": metrics.get("CV_views", 0.0),
        "raw_data": data,
    })

  if not records:
    print("未讀取到有效的帳號數據。")
    return

  df = pd.DataFrame(records)
  print(f"成功載入 {len(df)} 個帳號，開始執行四維度（VFR / ERV / CLR / SRV）評級...\n" + "-" * 65)

  # 計算四維百分位排名 (0 ~ 100)
  df["P_VFR"] = df["VFR"].rank(pct=True) * 100
  df["P_ERV"] = df["ERV"].rank(pct=True) * 100
  df["P_CLR"] = df["CLR"].rank(pct=True) * 100
  df["P_SRV"] = df["SRV"].rank(pct=True) * 100

  # 四維度加權綜合得分：ERV (30%) + VFR (25%) + SRV (25%) + CLR (20%)
  df["quality_score"] = (
      0.30 * df["P_ERV"]
      + 0.25 * df["P_VFR"]
      + 0.25 * df["P_SRV"]
      + 0.20 * df["P_CLR"]
  )

  def assign_label(score):
    if score >= 70.0:
      return "good"
    elif score >= 30.0:
      return "neutral"
    else:
      return "bad"

  df["classification"] = df["quality_score"].apply(assign_label)

  for _, row in df.iterrows():
    acc_json = row["raw_data"]
    acc_json.setdefault("metrics", {})["SRV"] = row["SRV"]
    acc_json["quality_evaluation"] = {
        "classification": row["classification"],
        "quality_score": round(float(row["quality_score"]), 2),
        "weights": {"ERV": 0.30, "VFR": 0.25, "SRV": 0.25, "CLR": 0.20},
        "percentiles": {
            "P_ERV": round(float(row["P_ERV"]), 2),
            "P_VFR": round(float(row["P_VFR"]), 2),
            "P_SRV": round(float(row["P_SRV"]), 2),
            "P_CLR": round(float(row["P_CLR"]), 2),
        },
    }

    with open(row["file_path"], "w", encoding="utf-8") as f:
      json.dump(acc_json, f, ensure_ascii=False, indent=2)

  summary_cols = [
      "account_id",
      "classification",
      "quality_score",
      "VFR",
      "ERV",
      "SRV",
      "CLR",
      "SLR",
      "followers",
      "total_videos",
      "CV_views",
  ]
  summary_df = df[summary_cols].sort_values(by="quality_score", ascending=False)

  summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
  with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

  counts = summary_df["classification"].value_counts()
  print("【四維度帳號分級統計結果】")
  print(f"  - Good   (優質): {counts.get('good', 0)} 個")
  print(f"  - Neutral(普通): {counts.get('neutral', 0)} 個")
  print(f"  - Bad    (不佳): {counts.get('bad', 0)} 個")
  print("-" * 65)
  print("\nTop 5 帳號預覽：")
  print(
      summary_df[["account_id", "classification", "quality_score", "ERV", "SRV", "VFR"]]
      .head()
      .to_string(index=False)
  )


if __name__ == "__main__":
  classify_accounts_data()