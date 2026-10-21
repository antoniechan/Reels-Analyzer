import os
import json
import numpy as np

# 路徑設定
FOLLOWERS_JSON = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\accountsProcessing\accounts_followers.json"
REELS_DATA_DIR = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\reelsData"
ACCOUNTS_DATA_DIR = r"C:\Users\anton\OneDrive\文件\GitHub\Reels-Analyzer\pythonProject\accountsData"


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


def process_accounts_metrics():
    os.makedirs(ACCOUNTS_DATA_DIR, exist_ok=True)

    # 1. 讀取 accounts_followers.json
    if not os.path.exists(FOLLOWERS_JSON):
        print(f"錯誤：找不到設定檔 {FOLLOWERS_JSON}")
        return

    with open(FOLLOWERS_JSON, "r", encoding="utf-8") as f:
        followers_data = json.load(f)

    accounts_dict = followers_data.get("accounts", followers_data)

    print(f"開始處理帳號指標計算，共 {len(accounts_dict)} 個帳號登錄於清單。\n" + "-" * 60)

    success_count = 0
    skipped_count = 0

    for account_name, acc_info in accounts_dict.items():
        # 取得 followers 數量
        followers_count = None
        if isinstance(acc_info, dict):
            followers_count = acc_info.get("followersCount") or acc_info.get("followers")
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

        # 3. 提取觀看數、點讚數與留言數
        views_list = []
        likes_list = []
        comments_list = []

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

            views_list.append(v)
            likes_list.append(l)
            comments_list.append(c)

        if not views_list:
            print(f"  [略過] {account_name} 提取不到有效數據。")
            skipped_count += 1
            continue

        # 4. 計算中位數、平均數與標準差
        med_views = float(np.median(views_list))
        med_likes = float(np.median(likes_list))
        med_comments = float(np.median(comments_list))

        mean_views = float(np.mean(views_list))
        mean_likes = float(np.mean(likes_list))
        mean_comments = float(np.mean(comments_list))

        std_views = float(np.std(views_list, ddof=1)) if len(views_list) > 1 else 0.0
        std_likes = float(np.std(likes_list, ddof=1)) if len(likes_list) > 1 else 0.0
        std_comments = float(np.std(comments_list, ddof=1)) if len(comments_list) > 1 else 0.0

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

        # 計算 CV (變異係數)
        cv_views = calculate_cv(views_list)
        cv_likes = calculate_cv(likes_list)
        cv_comments = calculate_cv(comments_list)

        # 5. 組合輸出結構
        result_payload = {
            "account_id": account_name,
            "followers": followers_count,
            "total_videos_analyzed": len(views_list),
            "metrics": {
                "VFR": vfr,
                "ERV": erv,
                "CLR": clr,
                "VFR_mean": vfr_mean,
                "CV_views": cv_views,
                "CV_likes": cv_likes,
                "CV_comments": cv_comments
            },
            "median_stats": {
                "median_views": med_views,
                "median_likes": med_likes,
                "median_comments": med_comments
            },
            "mean_stats": {
                "mean_views": mean_views,
                "mean_likes": mean_likes,
                "mean_comments": mean_comments
            },
            "std_stats": {
                "std_views": std_views,
                "std_likes": std_likes,
                "std_comments": std_comments
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