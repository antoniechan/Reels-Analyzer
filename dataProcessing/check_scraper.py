import os
import glob
import json
import argparse
from config import REELS_DATA_DIR, DATA_PROC_DIR

DEFAULT_REELS_DATA_DIR = str(REELS_DATA_DIR)
OUTPUT_REPORT_JSON = str(DATA_PROC_DIR / "zero_engagement_videos_report.json")

# 忽略系統彙總類檔案
EXCLUDE_JSON_NAMES = {
    "account_classification_summary.json",
    "accounts_followers.json",
    "video_classification_summary.json",
    "merged_accounts_data.json",
    "zero_engagement_videos_report.json",
}


def scan_zero_engagement_videos(data_dir=DEFAULT_REELS_DATA_DIR, output_json=OUTPUT_REPORT_JSON):
    if not os.path.exists(data_dir):
        print(f"錯誤：找不到目錄 -> {data_dir}")
        return []

    json_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))

    print("=" * 80)
    print(f"正在掃描目錄: {data_dir}")
    print(f"找到 {len(json_files)} 個 JSON 檔案，開始檢查 videoPlayCount == 0 且 likesCount == 0 的影片...")
    print("=" * 80 + "\n")

    zero_videos = []
    total_scanned_videos = 0

    for fpath in json_files:
        fname = os.path.basename(fpath)
        if fname in EXCLUDE_JSON_NAMES or fname.endswith("_audio_saliency.json") or fname.endswith(
                "_qc.json") or fname.endswith("_face.json"):
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            print(f"[警告] 無法讀取檔案 {fname}: {e}")
            continue

        account_id = os.path.splitext(fname)[0]

        posts = []
        if isinstance(content, list):
            posts = content
        elif isinstance(content, dict):
            account_id = content.get("account_id", account_id)
            posts = (
                    content.get("videos")
                    or content.get("posts")
                    or content.get("reels")
                    or content.get("items")
                    or content.get("latestPosts")
                    or []
            )

        for idx, post in enumerate(posts, start=1):
            if not isinstance(post, dict):
                continue

            total_scanned_videos += 1

            # 抓取播放數與讚數 (相容各種常見命名，若為 None 則視為 0)
            plays_raw = (
                post.get("videoPlayCount")
                if post.get("videoPlayCount") is not None
                else post.get("playCount")
                if post.get("playCount") is not None
                else post.get("views")
                if post.get("views") is not None
                else post.get("videoViewCount")
            )
            likes_raw = (
                post.get("likesCount")
                if post.get("likesCount") is not None
                else post.get("likes")
                if post.get("likes") is not None
                else post.get("diggCount")
            )
            comments_raw = (
                post.get("commentsCount")
                if post.get("commentsCount") is not None
                else post.get("commentCount")
                if post.get("commentCount") is not None
                else post.get("comments")
            )

            plays = int(plays_raw) if plays_raw is not None else 0
            likes = int(likes_raw) if likes_raw is not None else 0
            comments = int(comments_raw) if comments_raw is not None else 0

            # 判斷條件：播放量 == 0 且 按讚量 == 0
            if plays == 0 and likes == 0:
                local_vid = post.get("local_video_file") or post.get("local_audio_file") or f"video_{idx:03d}"
                video_stem = os.path.splitext(os.path.basename(str(local_vid)))[0]

                caption = str(post.get("caption") or "").replace("\n", " ").strip()
                caption_snippet = (caption[:45] + "...") if len(caption) > 45 else caption

                zero_videos.append({
                    "account_id": account_id,
                    "video_id": video_stem,
                    "json_file": os.path.relpath(fpath, data_dir),
                    "file_full_path": os.path.abspath(fpath),
                    "video_index_in_file": idx,
                    "videoPlayCount": plays,
                    "likesCount": likes,
                    "commentsCount": comments,
                    "videoDuration": post.get("videoDuration"),
                    "videoUrl": post.get("videoUrl"),
                    "caption": caption_snippet,
                    "timestamp": post.get("timestamp"),
                })

    # 輸出終端機報表
    print(f"掃描完成！總共檢查了 {total_scanned_videos} 條影片記錄。")
    print(f"共發現 {len(zero_videos)} 條影片播放數與按讚數皆為 0：\n")

    if zero_videos:
        print(f"{'No.':<4} | {'Account':<20} | {'Video ID':<10} | {'Comments':<8} | {'Caption (預覽)':<40}")
        print("-" * 90)
        for i, item in enumerate(zero_videos, start=1):
            print(
                f"{i:<4} | {item['account_id']:<20} | {item['video_id']:<10} | {item['commentsCount']:<8} | {item['caption']:<40}")
            print(f"     -> 檔案來源: {item['json_file']} (第 {item['video_index_in_file']} 筆)")
            if item.get("videoUrl"):
                print(f"     -> IG 連結: {item['videoUrl'][:80]}...")
            print()

        # 儲存報告為 JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "total_scanned_videos": total_scanned_videos,
                "zero_engagement_count": len(zero_videos),
                "zero_engagement_videos": zero_videos
            }, f, ensure_ascii=False, indent=2)
        print(f"詳細清單已儲存至: {os.path.abspath(output_json)}")
    else:
        print("未發現任何 videoPlayCount == 0 且 likesCount == 0 的影片。")

    return zero_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="搜尋 reelsData 目錄下所有 videoPlayCount=0 且 likesCount=0 的影片")
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_REELS_DATA_DIR,
        help=f"存放 JSON 的目錄路徑 (預設: {DEFAULT_REELS_DATA_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_REPORT_JSON,
        help=f"輸出報表 JSON 檔名 (預設: {OUTPUT_REPORT_JSON})"
    )
    args = parser.parse_args()

    scan_zero_engagement_videos(data_dir=args.dir, output_json=args.output)