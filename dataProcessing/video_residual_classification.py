import os
import glob
import json
import argparse
import numpy as np
from config import REELS_DATA_DIR, DEFAULT_REELS_DIR, ACCOUNT_PROC_DIR, DATA_PROC_DIR

# 預設路徑設定
ACCOUNTS_DATA_DIR = REELS_DATA_DIR
REELS_DIR = DEFAULT_REELS_DIR
ACCOUNT_SUMMARY_JSON = str(ACCOUNT_PROC_DIR / "account_classification_summary.json")
FOLLOWERS_JSON = str(ACCOUNT_PROC_DIR / "accounts_followers.json")
OUTPUT_VIDEO_JSON = str(DATA_PROC_DIR / "video_classification_summary.json")


# ==============================================================================
# 1. 載入帳號級調查數據與所有影片的 Play / Like / Comment / Share
# ==============================================================================

def load_account_metadata(summary_path=ACCOUNT_SUMMARY_JSON, followers_path=FOLLOWERS_JSON):
    """
    整合帳號粉絲數與帳號調查指標。
    相容新版 list 格式（[{"username": "...", "followersCount": 123, ...}]）與舊版 dict 格式。
    """
    account_meta = {}

    if os.path.exists(followers_path):
        with open(followers_path, "r", encoding="utf-8") as f:
            raw_f_data = json.load(f)

        if isinstance(raw_f_data, list):
            for item in raw_f_data:
                if not isinstance(item, dict):
                    continue
                acc_id = str(item.get("username") or item.get("account_id") or item.get("ownerUsername") or "").strip()
                if not acc_id:
                    continue
                cnt = item.get("followersCount")
                if cnt is None:
                    cnt = item.get("followers")
                if cnt is not None:
                    account_meta[acc_id] = {
                        "followers": max(1, int(cnt)),
                        "account_quality_score": 50.0,
                    }
        elif isinstance(raw_f_data, dict):
            f_data = raw_f_data.get("accounts") if isinstance(raw_f_data.get("accounts"), dict) else raw_f_data
            for acc_id, info in f_data.items():
                if isinstance(info, dict):
                    cnt = info.get("followersCount") if info.get("followersCount") is not None else info.get("followers")
                else:
                    cnt = info
                if cnt is not None:
                    account_meta[str(acc_id).strip()] = {
                        "followers": max(1, int(cnt)),
                        "account_quality_score": 50.0,
                    }

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            s_data = json.load(f)
            if isinstance(s_data, list):
                for item in s_data:
                    acc_id = item.get("account_id")
                    if not acc_id:
                        continue
                    account_meta.setdefault(acc_id, {})
                    if item.get("followers") is not None:
                        account_meta[acc_id]["followers"] = max(1, int(item.get("followers", 1)))
                    account_meta[acc_id]["account_quality_score"] = float(item.get("quality_score", 50.0))
                    account_meta[acc_id]["account_tier"] = item.get("classification", "neutral")

    return account_meta


def extract_videos_from_json_dir(accounts_dir, account_meta):
    """
    從目錄下的各帳號 JSON 檔案中提取每支影片的 play / like / comment / share。
    自動相容常見欄位名稱。
    """
    video_records = []
    json_files = sorted(glob.glob(os.path.join(accounts_dir, "*.json")))

    excluded_names = {
        os.path.basename(ACCOUNT_SUMMARY_JSON),
        "accounts_followers.json",
        "account_followers.json",
        os.path.basename(OUTPUT_VIDEO_JSON),
        "merged_accounts_data.json",
    }

    for fpath in json_files:
        fname = os.path.basename(fpath)
        if fname in excluded_names:
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        acc_id = os.path.splitext(fname)[0]
        if isinstance(data, dict):
            acc_id = data.get("account_id", acc_id)
            posts = (
                data.get("videos")
                or data.get("posts")
                or data.get("reels")
                or data.get("items")
                or data.get("latestPosts")
                or []
            )
            f_cnt = data.get("followersCount") if data.get("followersCount") is not None else data.get("followers")
            if f_cnt is not None and acc_id not in account_meta:
                account_meta[acc_id] = {"followers": max(1, int(f_cnt)), "account_quality_score": 50.0}
        elif isinstance(data, list):
            posts = data
        else:
            continue

        meta = account_meta.get(acc_id, {"followers": 1000, "account_quality_score": 50.0, "account_tier": "neutral"})
        followers = meta.get("followers", 1000)

        for idx, post in enumerate(posts, start=1):
            if not isinstance(post, dict):
                continue

            vid_id = (
                post.get("video_id")
                or post.get("local_id")
                or post.get("id")
                or post.get("shortCode")
                or f"video_{idx:03d}"
            )
            local_video_name = f"video_{idx:03d}"

            plays = (
                post.get("videoPlayCount")
                or post.get("videoViewCount")
                or post.get("playCount")
                or post.get("views")
                or post.get("plays")
                or 0
            )
            likes = post.get("likesCount") or post.get("diggCount") or post.get("likeCount") or post.get("likes") or 0
            comments = post.get("commentsCount") or post.get("commentCount") or post.get("comments") or 0
            shares = post.get("sharesCount") or post.get("shareCount") or post.get("shares") or 0

            plays = max(0, int(plays))
            likes = max(0, int(likes))
            comments = max(0, int(comments))
            shares = max(0, int(shares))

            if plays <= 0 and likes <= 0:
                continue

            safe_plays = float(max(1, plays))
            safe_likes = float(max(1, likes))

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
                "shares": shares,
                "VFR": plays / float(followers),
                "ERV": (likes + comments + shares) / safe_plays,
                "CLR": comments / safe_likes,
                "SRV": shares / safe_plays,
                "SLR": shares / safe_likes,
            })

    return video_records


# ==============================================================================
# 2. 計量經濟學殘差法 (OLS Econometric Residual Valuation, 納入 sharesCount)
# ==============================================================================

def z_score(arr):
    std = np.std(arr)
    return (arr - np.mean(arr)) / (std if std > 1e-8 else 1.0)


def compute_econometric_residual_scores(
    video_records,
    comment_weight: float = 3.0,
    share_weight: float = 5.0,
    reach_vs_eng_ratio: float = 0.45,
    account_prior_blend: float = 0.20,
    good_percentile: float = 75.0,
    bad_percentile: float = 25.0,
):
    """
    1. 構建包含高信任社交背書 (sharesCount) 的對數綜合互動指標 Y_iv:
       log_eng = ln(1 + likes + 3 * comments + 5 * shares)
    2. 對 ln(Followers) 與 [ln(Followers)]^2 進行 OLS 迴歸，控制粉絲基數規模偏誤
    3. 提取殘差 e_iv = Y_iv - Y_hat_iv 作為「純內容破圈與高信任表現分」
    4. 融合帳號歷史品質先驗，轉為 0~100 百分位分並分級 (good / neutral / bad)
    """
    n = len(video_records)
    if n == 0:
        raise ValueError("未找到任何有效的影片數據，請確認輸入目錄結構。")

    plays = np.array([v["plays"] for v in video_records], dtype=np.float64)
    likes = np.array([v["likes"] for v in video_records], dtype=np.float64)
    comments = np.array([v["comments"] for v in video_records], dtype=np.float64)
    shares = np.array([v["shares"] for v in video_records], dtype=np.float64)
    followers = np.array([v["followers"] for v in video_records], dtype=np.float64)
    acc_q_scores = np.array([v["account_quality_score"] for v in video_records], dtype=np.float64)

    # Step 1: 對數轉換與綜合互動指標 Y_iv（納入 sharesCount 高信任權重）
    log_plays = np.log1p(plays)
    log_eng = np.log1p(likes + comment_weight * comments + share_weight * shares)

    y_composite = reach_vs_eng_ratio * z_score(log_plays) + (1.0 - reach_vs_eng_ratio) * z_score(log_eng)

    # Step 2: 建立設計矩陣 X = [1, ln(Followers), (ln(Followers))^2]
    log_followers = np.log1p(followers)
    log_f_norm = z_score(log_followers)
    X = np.column_stack([
        np.ones(n),
        log_f_norm,
        log_f_norm ** 2,
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y_composite, rcond=None)
    y_hat = X @ beta

    # Step 3: 提取計量殘差
    residuals = y_composite - y_hat
    res_z = z_score(residuals)

    # Step 4: 結合帳號品質先驗進行微調平衡
    acc_q_z = z_score(acc_q_scores)
    final_latent_score = (1.0 - account_prior_blend) * res_z + account_prior_blend * acc_q_z

    order = np.argsort(final_latent_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 100.0, n)

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

    video_records_sorted = sorted(video_records, key=lambda x: x["video_quality_score"], reverse=True)

    reg_stats = {
        "total_videos": n,
        "comment_weight": comment_weight,
        "share_weight": share_weight,
        "reach_vs_eng_ratio": reach_vs_eng_ratio,
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
    parser = argparse.ArgumentParser(description="以計量殘差法進行 Video-level 分類與評分")
    parser.add_argument("--accounts_dir", type=str, default=ACCOUNTS_DATA_DIR, help="reelsData 資料夾路徑")
    parser.add_argument("--followers_json", type=str, default=FOLLOWERS_JSON, help="新版 account_followers.json 路徑")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_JSON, help="輸出的影片分類 JSON 路徑")
    parser.add_argument("--good_pct", type=float, default=75.0, help="Good 影片的百分位門檻 (預設 Top 25%, 即 >=75)")
    parser.add_argument("--bad_pct", type=float, default=25.0, help="Bad 影片的百分位門檻 (預設 Bottom 25%, 即 <=25)")
    parser.add_argument("--prior_weight", type=float, default=0.20, help="帳號調查品質分所佔的調節權重 (預設 0.20)")
    parser.add_argument("--comment_weight", type=float, default=3.0, help="留言權重 (預設 3.0)")
    parser.add_argument("--share_weight", type=float, default=5.0, help="分享背書權重 (預設 5.0)")
    args = parser.parse_args()

    account_meta = load_account_metadata(followers_path=args.followers_json)
    videos = extract_videos_from_json_dir(args.accounts_dir, account_meta)

    if not videos:
        print(f"錯誤：在 {args.accounts_dir} 中找不到任何影片紀錄。")
        return

    scored_videos, reg_stats = compute_econometric_residual_scores(
        videos,
        comment_weight=args.comment_weight,
        share_weight=args.share_weight,
        account_prior_blend=args.prior_weight,
        good_percentile=args.good_pct,
        bad_percentile=args.bad_pct,
    )

    counts = {
        "good": sum(1 for v in scored_videos if v["video_classification"] == "good"),
        "neutral": sum(1 for v in scored_videos if v["video_classification"] == "neutral"),
        "bad": sum(1 for v in scored_videos if v["video_classification"] == "bad"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_payload = {
        "methodology": "Econometric Residual Valuation (OLS Follower-Scale Debiased with Shares Endorsement)",
        "regression_diagnostics": reg_stats,
        "classification_counts": counts,
        "videos": scored_videos,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("【殘差法 — 影片級分類統計結果 (含 sharesCount 高信任背書權重)】")
    print(f"  - 分析影片總數 (Total Videos):         {reg_stats['total_videos']}")
    print(f"  - 粉絲規模彈性係數 (Beta_1 Followers):  {reg_stats['ols_beta_log_followers']:.4f}")
    print(f"  - 規模邊際遞減係數 (Beta_2 Squared):    {reg_stats['ols_beta_log_followers_sq']:.4f}")
    print(f"  - 粉絲基數解釋變異比例 (R-squared):     {reg_stats['r_squared_follower_effect']*100:.2f}%")
    print("-" * 80)
    print(f"  - Good 影片數量 (Top {100-args.good_pct:.0f}%):            {counts['good']} 支")
    print(f"  - Neutral 影片數量 (Middle):          {counts['neutral']} 支")
    print(f"  - Bad 影片數量 (Bottom {args.bad_pct:.0f}%):          {counts['bad']} 支")
    print(f"  - 輸出檔案路徑:                        {os.path.abspath(args.output)}")
    print("=" * 80)

    print("\n【Top 5 最高評分影片 (超越粉絲規模預期的爆款)】")
    for idx, v in enumerate(scored_videos[:5], 1):
        print(
            f"  [{idx}] {v['account_id']} / {v['local_index_name']} ({v['video_id']}) | "
            f"Score: {v['video_quality_score']:.2f} | Residual: +{v['raw_residual']:.2f} | "
            f"Plays: {v['plays']:,}, Likes: {v['likes']:,}, Comments: {v['comments']:,}, Shares: {v['shares']:,} "
            f"(Followers: {v['followers']:,})"
        )


if __name__ == "__main__":
    main()
