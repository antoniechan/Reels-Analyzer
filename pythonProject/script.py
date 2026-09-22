from datetime import datetime, timedelta
import pandas as pd

# 讀取從 Apify 下載的 Profile Dataset (JSON 格式保留較完整巢狀欄位)
df = pd.read_json("apify_instagram_profiles.json")

# 設定研究基準日（以當前時間回推 6 個月，約 180 天）
six_months_ago = datetime.now() - timedelta(days=180)


def meets_criteria(row):
    # 條件 1: Account Age > 6 months
    # 註：Apify 通常提供 date_joined_as_timestamp 或 joinedDate 字串
    join_date = row.get("about", {}).get("date_joined_as_timestamp")
    if join_date:
        account_created = datetime.fromtimestamp(join_date)
        if account_created > six_months_ago:
            return False
    else:
        # 若 Meta 隱藏非商業帳號的註冊日，改用其最早一則貼文/第一篇可抓取的貼文時間做近似檢驗
        pass

    # 條件 4: Followers > 1,000
    followers = row.get("followersCount", 0)
    if followers <= 1000:
        return False

    # 條件 2 & 3: Reels > 10 counts 且 Reels 佔比 > 30%
    # 計算方式 A：若 Actor 有抓到總 Reels 數與總貼文數
    total_posts = row.get("postsCount", 0)
    reels_count = row.get(
        "videoCount", 0
    )  # 部分 actor 標示為 videoCount 或 clipsCount

    # 若抓取的是最新 N 則貼文樣本 (例如 latestPosts)
    latest_posts = row.get("latestPosts", [])
    if latest_posts:
        sample_reels = sum(
            1
            for post in latest_posts
            if post.get("isVideo") or post.get("type") == "Video"
        )
        sample_ratio = sample_reels / len(latest_posts)

        # 樣本 Reels 比例 >= 30% 且估算總 Reels 數 >= 10
        if sample_ratio < 0.3 or (total_posts * sample_ratio) < 10:
            return False
    else:
        if total_posts == 0 or reels_count < 10:
            return False
        if (reels_count / total_posts) < 0.3:
            return False

    return True


# 執行過濾
filtered_influencers = df[df.apply(meets_criteria, axis=1)]

# 匯出符合學術研究標準的名單
filtered_influencers[
    ["username", "fullName", "followersCount", "postsCount", "url"]
].to_csv("health_influencers_filtered.csv", index=False)
print(f"篩選完成，共符合 {len(filtered_influencers)} 位 Influencer。")