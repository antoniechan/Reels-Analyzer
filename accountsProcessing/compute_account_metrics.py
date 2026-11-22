import os
import json
import numpy as np
from config import ACCOUNT_PROC_DIR, REELS_DATA_DIR, ACCOUNT_DATA_DIR

# 路徑設定（同時相容 account_followers.json 與 accounts_followers.json）
FOLLOWERS_JSON = str(ACCOUNT_PROC_DIR / "accounts_followers.json")
ACCOUNTS_DATA_DIR = str(ACCOUNT_DATA_DIR)


def safe_extract_metric(post, keys):
    """
    依序嘗試多個欄位 key，取得第一個非 None 且可轉為 float 的數值
    """
    for k in keys:
        if k in post and post[k] is not None:
            try:
                return float(post[k])
            except (ValueError, TypeError):
                continue
    return 0.0


def calculate_cv(arr):
    """
    計算樣本變異係數 CV = std / mean (ddof=1 樣本標準差)
    若 mean <= 0 或樣本數 <= 1 則回傳 0.0
    """
    if len(arr) <= 1:
        return 0.0
    mean_val = float(np.mean(arr))
    if mean_val <= 0:
        return 0.0
    std_val = float(np.std(arr, ddof=1))
    return float(std_val / mean_val)


def normalize_accounts_data(raw_data):
    """
    相容新版 list 格式（來自 scrape_account_followers.py）與舊版 dict 格式：
    - 新版 list: [{"username": "xxx", "followersCount": 123, ...}, ...]
    - 舊版 dict: {"accounts": {"xxx": {"followersCount": 123}}} 或 {"xxx": 123}
    統一轉換為 {account_name: acc_info} 的字典結構。
    """
    normalized = {}
    if isinstance(raw_data, list):
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            acc_name = str(item.get("username") or item.get("account_id") or item.get("ownerUsername") or "").strip()
            if acc_name:
                normalized[acc_name] = item
    elif isinstance(raw_data, dict):
        source_dict = raw_data.get("accounts") if isinstance(raw_data.get("accounts"), dict) else raw_data
        for acc_name, acc_info in source_dict.items():
            normalized[str(acc_name).strip()] = acc_info
    return normalized


def process_accounts_metrics():
    os.makedirs(ACCOUNTS_DATA_DIR, exist_ok=True)

    # 1. 讀取最新版 account_followers.json / accounts_followers.json
    if not os.path.exists(FOLLOWERS_JSON):
        print(f"錯誤：找不到設定檔 {FOLLOWERS_JSON}")
        return

    with open(FOLLOWERS_JSON, "r", encoding="utf-8") as f:
        followers_data = json.load(f)

    accounts_dict = normalize_accounts_data(followers_data)

    print(f"開始處理帳號指標計算，共 {len(accounts_dict)} 個帳號登錄於清單 ({os.path.basename(FOLLOWERS_JSON)})。\n" + "-" * 60)

    success_count = 0
    skipped_count = 0

    for account_name, acc_info in accounts_dict.items():
        # 取得 followers 數量（注意：followersCount 可能為 0，不能直接用 or 判斷）
        followers_count = None
        if isinstance(acc_info, dict):
            if acc_info.get("followersCount") is not None:
                followers_count = acc_info.get("followersCount")
            elif acc_info.get("followers") is not None:
                followers_count = acc_info.get("followers")
        elif isinstance(acc_info, (int, float)):
            followers_count = acc_info

        # 檢查該帳號在 reelsData 目錄下對應的 json 檔案
        reels_file_path = os.path.join(REELS_DATA_DIR, f"{account_name}.json")
        if not os.path.exists(reels_file_path):
            print(f"  [略過] 找不到影片資料: {account_name}.json")
            skipped_count += 1
            continue

        # 2. 讀取 reelsData 檔案內容
        try:
            with open(reels_file_path, "r", encoding="utf-8") as f:
                reels_content = json.load(f)
        except Exception as e:
            print(f"  [錯誤] 讀取 {account_name}.json 失敗: {e}")
            skipped_count += 1
            continue

        # 取得影片物件清單
        posts = []
        if isinstance(reels_content, list):
            posts = reels_content
        elif isinstance(reels_content, dict):
            for k in ["videos", "reels", "posts", "data", "items"]:
                if k in reels_content and isinstance(reels_content[k], list):
                    posts = reels_content[k]
                    break

        if not posts:
            print(f"  [略過] {account_name}.json 中無任何影片資料項目。")
            skipped_count += 1
            continue

        # 3. 提取觀看數、點讚數、留言數與分享數
        views_list = []
        likes_list = []
        comments_list = []
        shares_list = []

        for p in posts:
            if not isinstance(p, dict):
                continue

            # 優先採用 videoPlayCount，次選 videoViewCount
            v = safe_extract_metric(p, [
                "videoPlayCount",
                "videoViewCount",
                "playCount",
                "viewCount",
                "views"
            ])

            l = safe_extract_metric(p, [
                "likesCount",
                "likeCount",
                "likes"
            ])

            c = safe_extract_metric(p, [
                "commentsCount",
                "commentCount",
                "comments"
            ])

            s = safe_extract_metric(p, [
                "sharesCount",
                "shareCount",
                "shares"
            ])

            views_list.append(v)
            likes_list.append(l)
            comments_list.append(c)
            shares_list.append(s)

        if not views_list:
            print(f"  [略過] {account_name} 提取不到有效數據。")
            skipped_count += 1
            continue

        # 4. 計算中位數、平均數與標準差
        med_views = float(np.median(views_list))
        med_likes = float(np.median(likes_list))
        med_comments = float(np.median(comments_list))
        med_shares = float(np.median(shares_list))

        mean_views = float(np.mean(views_list))
        mean_likes = float(np.mean(likes_list))
        mean_comments = float(np.mean(comments_list))
        mean_shares = float(np.mean(shares_list))

        std_views = float(np.std(views_list, ddof=1)) if len(views_list) > 1 else 0.0
        std_likes = float(np.std(likes_list, ddof=1)) if len(likes_list) > 1 else 0.0
        std_comments = float(np.std(comments_list, ddof=1)) if len(comments_list) > 1 else 0.0
        std_shares = float(np.std(shares_list, ddof=1)) if len(shares_list) > 1 else 0.0

        eps = 1e-6

        # 計算 VFR, ERV, CLR
        if followers_count is not None and followers_count > 0:
            vfr = float(med_views / followers_count)
            vfr_mean = float(mean_views / followers_count)
        else:
            vfr = None
            vfr_mean = None

        erv = float((med_likes + med_comments) / (med_views + eps)) if med_views > 0 else 0.0
        clr = float(med_comments / (med_likes + eps)) if med_likes > 0 else 0.0
        srv = float(med_shares / (med_views + eps)) if med_views > 0 else 0.0
        slr = float(med_shares / (med_likes + eps)) if med_likes > 0 else 0.0

        # 計算 CV (變異係數)
        cv_views = calculate_cv(views_list)
        cv_likes = calculate_cv(likes_list)
        cv_comments = calculate_cv(comments_list)
        cv_shares = calculate_cv(shares_list)

        # 5. 組合輸出結構（保留新版主頁元數據）
        result_payload = {
            "account_id": account_name,
            "full_name": acc_info.get("fullName", "") if isinstance(acc_info, dict) else "",
            "followers": followers_count,
            "follows": acc_info.get("followsCount") if isinstance(acc_info, dict) else None,
            "posts_count": acc_info.get("postsCount") if isinstance(acc_info, dict) else None,
            "verified": acc_info.get("verified") if isinstance(acc_info, dict) else None,
            "total_videos_analyzed": len(views_list),
            "metrics": {
                "VFR": vfr,
                "ERV": erv,
                "CLR": clr,
                "SRV": srv,
                "SLR": slr,
                "VFR_mean": vfr_mean,
                "CV_views": cv_views,
                "CV_likes": cv_likes,
                "CV_comments": cv_comments,
                "CV_shares": cv_shares
            },
            "median_stats": {
                "median_views": med_views,
                "median_likes": med_likes,
                "median_comments": med_comments,
                "median_shares": med_shares
            },
            "mean_stats": {
                "mean_views": mean_views,
                "mean_likes": mean_likes,
                "mean_comments": mean_comments,
                "mean_shares": mean_shares
            },
            "std_stats": {
                "std_views": std_views,
                "std_likes": std_likes,
                "std_comments": std_comments,
                "std_shares": std_shares
            }
        }

        # 6. 輸出至 accountsData/<account_name>.json
        out_path = os.path.join(ACCOUNTS_DATA_DIR, f"{account_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

        vfr_str = f"{round(vfr, 4)}" if vfr is not None else "N/A"
        print(
            f"  -> 已儲存: {account_name}.json (VFR: {vfr_str}, ERV: {round(erv * 100, 2)}%, CV_views: {round(cv_views, 2)})")
        success_count += 1

    print("\n" + "=" * 60)
    print(f"處理完成！成功輸出: {success_count} 個帳號，略過: {skipped_count} 個帳號。")
    print(f"儲存目錄: {ACCOUNTS_DATA_DIR}")


if __name__ == "__main__":
    process_accounts_metrics()
