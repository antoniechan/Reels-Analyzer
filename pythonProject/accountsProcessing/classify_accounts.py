import glob
import json
import os
import pandas as pd

ACCOUNTS_DATA_DIR = (
    r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\accountsData"
)
OUTPUT_SUMMARY_JSON = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\accountsProcessing\account_classification_summary.json"
OUTPUT_SUMMARY_CSV = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\accountsProcessing\account_classification_summary.csv"


def classify_accounts_data():
  json_pattern = os.path.join(ACCOUNTS_DATA_DIR, "*.json")
  json_files = glob.glob(json_pattern)

  if not json_files:
    print(f"錯誤：在 {ACCOUNTS_DATA_DIR} 中未找到任何 JSON 檔案。")
    return

  records = []

  # 1. 讀取所有帳號資料
  for fpath in json_files:
    fname = os.path.basename(fpath)
    if fname in [
        OUTPUT_SUMMARY_JSON,
        "accounts_followers.json",
    ]:  # 避免讀取到總表
      continue

    try:
      with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as e:
      print(f"讀取失敗 {fname}: {e}")
      continue

    account_id = data.get("account_id", os.path.splitext(fname)[0])
    metrics = data.get("metrics", {})

    vfr = metrics.get("VFR")
    erv = metrics.get("ERV")
    clr = metrics.get("CLR")
    cv_views = metrics.get("CV_views")

    records.append({
        "file_path": fpath,
        "account_id": account_id,
        "followers": data.get("followers"),
        "total_videos": data.get("total_videos_analyzed", 0),
        "VFR": vfr if vfr is not None else 0.0,
        "ERV": erv if erv is not None else 0.0,
        "CLR": clr if clr is not None else 0.0,
        "CV_views": cv_views if cv_views is not None else 0.0,
        "raw_data": data,
    })

  if not records:
    print("未讀取到有效的帳號數據。")
    return

  df = pd.DataFrame(records)
  total_accounts = len(df)
  print(f"成功載入 {total_accounts} 個帳號，開始進行評級計算...\n" + "-" * 60)

  # 2. 計算百分位排名 (Percentile Rank: 0 ~ 100)
  df["P_VFR"] = df["VFR"].rank(pct=True) * 100
  df["P_ERV"] = df["ERV"].rank(pct=True) * 100
  df["P_CLR"] = df["CLR"].rank(pct=True) * 100

  # 3. 加權計算綜合得分
  # ERV (40%): 觀看轉化互動是核心
  # VFR (35%): 破圈能力
  # CLR (25%): 深度信任度與討論意願
  df["quality_score"] = (
      0.40 * df["P_ERV"] + 0.35 * df["P_VFR"] + 0.25 * df["P_CLR"]
  )

  # 4. 判定級別 (Good / Neutral / Bad)
  def assign_label(score):
    if score >= 70.0:
      return "good"
    elif score >= 30.0:
      return "neutral"
    else:
      return "bad"

  df["classification"] = df["quality_score"].apply(assign_label)

  # 5. 回寫至各帳號原有的 JSON 檔案
  for _, row in df.iterrows():
    acc_json = row["raw_data"]
    acc_json["quality_evaluation"] = {
        "classification": row["classification"],
        "quality_score": round(float(row["quality_score"]), 2),
        "percentiles": {
            "P_ERV": round(float(row["P_ERV"]), 2),
            "P_VFR": round(float(row["P_VFR"]), 2),
            "P_CLR": round(float(row["P_CLR"]), 2),
        },
    }

    with open(row["file_path"], "w", encoding="utf-8") as f:
      json.dump(acc_json, f, ensure_ascii=False, indent=2)

  # 6. 生成彙總表
  summary_cols = [
      "account_id",
      "classification",
      "quality_score",
      "VFR",
      "ERV",
      "CLR",
      "followers",
      "total_videos",
      "CV_views",
  ]
  summary_df = df[summary_cols].sort_values(
      by="quality_score", ascending=False
  )

  # 輸出 CSV
  summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

  # 輸出 JSON
  summary_list = summary_df.to_dict(orient="records")
  with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_list, f, ensure_ascii=False, indent=2)

  # 7. 控制台印出統計結果
  counts = summary_df["classification"].value_counts()
  print("【帳號分級統計結果】")
  print(f"  - Good   (優質): {counts.get('good', 0)} 個")
  print(f"  - Neutral(普通): {counts.get('neutral', 0)} 個")
  print(f"  - Bad    (不佳): {counts.get('bad', 0)} 個")
  print("-" * 60)
  print("\nTop 5 帳號預覽：")
  print(
      summary_df[["account_id", "classification", "quality_score", "ERV", "VFR"]]
      .head()
      .to_string(index=False)
  )
  print(f"\n已更新所有單獨帳號的 JSON 檔案！")
  print(f"總結摘要已儲存至: {OUTPUT_SUMMARY_CSV} 及 {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
  classify_accounts_data()