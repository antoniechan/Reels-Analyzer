import os
import glob
import json
import argparse
import numpy as np
from config import REELS_DATA_DIR, DEFAULT_REELS_DIR, ACCOUNT_PROC_DIR, DATA_PROC_DIR

# 預設路徑設定
ACCOUNTS_DATA_DIR = str(REELS_DATA_DIR / "reelsData")
REELS_DIR = DEFAULT_REELS_DIR
ACCOUNT_SUMMARY_JSON = str(ACCOUNT_PROC_DIR / "account_classification_summary.json")
FOLLOWERS_JSON = str(ACCOUNT_PROC_DIR / "accounts_followers.json")
OUTPUT_VIDEO_JSON = str(DATA_PROC_DIR / "video_classification_summary.json")


# ==============================================================================
# 1. 載入帳號級調查數據與所有影片的 Play / Like / Comment
# ==============================================================================

def load_account_metadata(summary_path=ACCOUNT_SUMMARY_JSON, followers_path=FOLLOWERS_JSON):
    """整合帳號粉絲數與帳號調查指標"""
    account_meta = {}

    if os.path.exists(followers_path):
        with open(followers_path, "r", encoding="utf-8") as f:
            f_data = json.load(f).get("accounts", {})
            for acc_id, info in f_data.items():
                cnt = info.get("followersCount", 0) if isinstance(info, dict) else info
                account_meta[acc_id] = {"followers": max(1, int(cnt)), "account_quality_score": 50.0}

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            s_data = json.load(f)
            for item in s_data:
                acc_id = item.get("account_id")
                if not acc_id:
                    continue
                if acc_id not in account_meta:
                    account_meta[acc_id] = {}
                account_meta[acc_id]["followers"] = max(1, int(item.get("followers", 1)))
                account_meta[acc_id]["account_quality_score"] = float(item.get("quality_score", 50.0))
                account_meta[acc_id]["account_tier"] = item.get("classification", "neutral")

    return account_meta


def extract_videos_from_json_dir(accounts_dir, account_meta):
    """
    從 accountsData 目錄下的各帳號 JSON 檔案中提取每支影片的 play / like / comment。
    自動相容常見欄位名稱 (如 videoPlayCount / playCount / views, likesCount / likes 等)
    """
    video_records = []
    json_files = sorted(glob.glob(os.path.join(accounts_dir, "*.json")))

    for fpath in json_files:
        fname = os.path.basename(fpath)
        if fname in [ACCOUNT_SUMMARY_JSON, FOLLOWERS_JSON, OUTPUT_VIDEO_JSON, "merged_accounts_data.json"]:
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        acc_id = os.path.splitext(fname)[0]
        if isinstance(data, dict):
            acc_id = data.get("account_id", acc_id)
            # 尋找影片陣列欄位
            posts = (
                data.get("videos")
                or data.get("posts")
                or data.get("reels")
                or data.get("items")
                or data.get("latestPosts")
                or []
            )
            # 若 followers 寫在個別帳號 JSON 中，同步更新
            f_cnt = data.get("followersCount") or data.get("followers")
            if f_cnt and acc_id not in account_meta:
                account_meta[acc_id] = {"followers": max(1, int(f_cnt)), "account_quality_score": 50.0}
        elif isinstance(data, list):
            posts = data
        else:
            continue

        meta = account_meta.get(acc_id, {"followers": 1000, "account_quality_score": 50.0, "account_tier": "neutral"})
        followers = meta["followers"]

        for idx, post in enumerate(posts, start=1):
            if not isinstance(post, dict):
                continue

            # 自動對應影片 ID (如 video_001)
            vid_id = (
                post.get("video_id")
                or post.get("local_id")
                or post.get("id")
                or post.get("shortCode")
                or f"video_{idx:03d}"
            )
            local_video_name = f"video_{idx:03d}"

            # 讀取播放量 (Plays / Views)
            plays = (
                post.get("videoPlayCount")
                or post.get("playCount")
                or post.get("views")
                or post.get("plays")
                or 0
            )
            # 讀取按讚數 (Likes)
            likes = post.get("likesCount") or post.get("diggCount") or post.get("likes") or 0
            # 讀取留言數 (Comments)
            comments = post.get("commentsCount") or post.get("commentCount") or post.get("comments") or 0

            plays = max(0, int(plays))
            likes = max(0, int(likes))
            comments = max(0, int(comments))

            # 略過無效或未抓到觀看數的異常紀錄
            if plays <= 0 and likes <= 0:
                continue

            video_records.append({
                "account_id": acc_id,
                "video_id": str(vid_id),
                "local_index_name": local_video_name,
                "followers": followers,
                "account_quality_score": meta.get("account_quality_score", 50.0),
                "account_tier": meta.get("account_tier", "neutral"),
                "plays": plays,
                "likes": likes,
                "comments": comments,
                "VFR": plays / float(followers),
                "ERV": (likes + comments) / float(max(1, plays)),
                "CLR": comments / float(max(1, likes)),
            })

    return video_records


# ==============================================================================
# 2. 策略三：計量經濟學殘差法 (OLS Econometric Residual Valuation)
# ==============================================================================

def z_score(arr):
    std = np.std(arr)
    return (arr - np.mean(arr)) / (std if std > 1e-8 else 1.0)


def compute_econometric_residual_scores(
    video_records,
    comment_weight: float = 3.0,
    reach_vs_eng_ratio: float = 0.5,
    account_prior_blend: float = 0.2,
    good_percentile: float = 75.0,
    bad_percentile: float = 25.0,
):
    """
    1. 構建對數綜合互動指標 Y_iv
    2. 對 ln(Followers), [ln(Followers)]^2 與帳號基礎特徵進行 OLS 迴歸
    3. 提取殘差 e_iv = Y_iv - Y_hat_iv 作為剔除粉絲規模偏誤後的「純內容表現分」
    4. 選擇性融合小比例的帳號品質先驗 (account_prior_blend)，轉為 0~100 分並分級
    """
    n = len(video_records)
    if n == 0:
        raise ValueError("未找到任何有效的影片數據，請確認 accountsData 目錄結構。")

    plays = np.array([v["plays"] for v in video_records], dtype=np.float64)
    likes = np.array([v["likes"] for v in video_records], dtype=np.float64)
    comments = np.array([v["comments"] for v in video_records], dtype=np.float64)
    followers = np.array([v["followers"] for v in video_records], dtype=np.float64)
    acc_q_scores = np.array([v["account_quality_score"] for v in video_records], dtype=np.float64)

    # Step 1: 對數轉換與綜合互動指標 Y_iv
    log_plays = np.log1p(plays)
    log_eng = np.log1p(likes + comment_weight * comments)

    # 綜合廣度 (Plays) 與深度 (Likes + 3*Comments)
    y_composite = reach_vs_eng_ratio * z_score(log_plays) + (1.0 - reach_vs_eng_ratio) * z_score(log_eng)

    # Step 2: 建立設計矩陣 X = [1, ln(Followers), (ln(Followers))^2]
    # 控制粉絲基數的一階線性效應與二階邊際遞減效應
    log_followers = np.log1p(followers)
    log_f_norm = z_score(log_followers)
    X = np.column_stack([
        np.ones(n),
        log_f_norm,
        log_f_norm ** 2,
    ])

    # 使用最小平方法求解 Y_hat = X * beta
    beta, _, _, _ = np.linalg.lstsq(X, y_composite, rcond=None)
    y_hat = X @ beta

    # Step 3: 提取計量殘差 (Econometric Residual)
    residuals = y_composite - y_hat
    res_z = z_score(residuals)

    # Step 4: 結合帳號調查權重進行微調平衡 (預設 80% 單片純內容殘差 + 20% 帳號歷史品質權重)
    acc_q_z = z_score(acc_q_scores)
    final_latent_score = (1.0 - account_prior_blend) * res_z + account_prior_blend * acc_q_z

    # 將潛在分數透過百分位數排序 (Percentile Rank) 映射至 [0, 100] 區間
    order = np.argsort(final_latent_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 100.0, n)

    # 切分 Good / Neutral / Bad
    for i, rec in enumerate(video_records):
        score_100 = float(ranks[i])
        if score_100 >= good_percentile:
            clf = "good"
        elif score_100 <= bad_percentile:
            clf = "bad"
        else:
            clf = "neutral"

        rec["expected_baseline"] = round(float(y_hat[i]), 5)
        rec["raw_residual"] = round(float(residuals[i]), 5)
        rec["video_quality_score"] = round(score_100, 4)
        rec["video_classification"] = clf

    # 依 video_quality_score 由高至低排序
    video_records_sorted = sorted(video_records, key=lambda x: x["video_quality_score"], reverse=True)

    reg_stats = {
        "total_videos": n,
        "ols_intercept": float(beta[0]),
        "ols_beta_log_followers": float(beta[1]),
        "ols_beta_log_followers_sq": float(beta[2]),
        "r_squared_follower_effect": float(1.0 - np.var(residuals) / np.var(y_composite)),
    }

    return video_records_sorted, reg_stats


# ==============================================================================
# 3. 主程序與結果輸出
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="策略三：以計量殘差法進行 Video-level 分類與評分")
    parser.add_argument("--accounts_dir", type=str, default=ACCOUNTS_DATA_DIR, help="accountsData 資料夾路徑")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_JSON, help="輸出的影片分類 JSON 路徑")
    parser.add_argument("--good_pct", type=float, default=75.0, help="Good 影片的百分位門檻 (預設 Top 25%, 即 >=75)")
    parser.add_argument("--bad_pct", type=float, default=25.0, help="Bad 影片的百分位門檻 (預設 Bottom 25%, 即 <=25)")
    parser.add_argument("--prior_weight", type=float, default=0.20, help="帳號調查品質分所佔的調節權重 (預設 0.20)")
    args = parser.parse_args()

    account_meta = load_account_metadata()
    videos = extract_videos_from_json_dir(args.accounts_dir, account_meta)

    if not videos:
        print(f"錯誤：在 {args.accounts_dir} 中找不到任何影片紀錄。")
        return

    scored_videos, reg_stats = compute_econometric_residual_scores(
        videos,
        account_prior_blend=args.prior_weight,
        good_percentile=args.good_pct,
        bad_percentile=args.bad_pct,
    )

    counts = {
        "good": sum(1 for v in scored_videos if v["video_classification"] == "good"),
        "neutral": sum(1 for v in scored_videos if v["video_classification"] == "neutral"),
        "bad": sum(1 for v in scored_videos if v["video_classification"] == "bad"),
    }

    output_payload = {
        "methodology": "Econometric Residual Valuation (OLS Follower-Scale Debiased)",
        "regression_diagnostics": reg_stats,
        "classification_counts": counts,
        "videos": scored_videos,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("【殘差法 — 影片級分類統計結果】")
    print(f"  - 分析影片總數 (Total Videos):         {reg_stats['total_videos']}")
    print(f"  - 粉絲規模彈性係數 (Beta_1 Followers):  {reg_stats['ols_beta_log_followers']:.4f}")
    print(f"  - 規模邊際遞減係數 (Beta_2 Squared):    {reg_stats['ols_beta_log_followers_sq']:.4f}")
    print(f"  - 粉絲基數解釋變異比例 (R-squared):     {reg_stats['r_squared_follower_effect']*100:.2f}%")
    print("-" * 70)
    print(f"  - Good 影片數量 (Top {100-args.good_pct:.0f}%):            {counts['good']} 支")
    print(f"  - Neutral 影片數量 (Middle):          {counts['neutral']} 支")
    print(f"  - Bad 影片數量 (Bottom {args.bad_pct:.0f}%):          {counts['bad']} 支")
    print(f"  - 輸出檔案路徑:                        {os.path.abspath(args.output)}")
    print("=" * 70)

    print("\n【Top 5 最高評分影片 (超越粉絲規模預期的爆款)】")
    for idx, v in enumerate(scored_videos[:5], 1):
        print(
            f"  [{idx}] {v['account_id']} / {v['local_index_name']} ({v['video_id']}) | "
            f"Score: {v['video_quality_score']:.2f} | Residual: +{v['raw_residual']:.2f} | "
            f"Plays: {v['plays']:,} (Followers: {v['followers']:,})"
        )


if __name__ == "__main__":
    main()