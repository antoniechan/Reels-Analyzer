import os
import re
import json
import shutil
import argparse

# 匹配 local_video_file 與 local_audio_file 的正則表達式
# 例如: video_005.mp4 -> group(1)="video_", group(2)="005", group(3)=".mp4"
FILENAME_PATTERN = re.compile(r"^(video_)(\d+)(\.[a-zA-Z0-9]+)$")


def update_target_filename(orig_filename: str, threshold: int):
    """
    若檔名格式為 video_xxx.ext 且數字 > threshold，則將數字減 1 並保持原補零長度。
    回傳: (更新後的檔名, 是否有異動)
    """
    if not orig_filename or not isinstance(orig_filename, str):
        return orig_filename, False

    match = FILENAME_PATTERN.match(orig_filename.strip())
    if not match:
        return orig_filename, False

    prefix, num_str, ext = match.groups()
    val = int(num_str)

    if val > threshold:
        new_val = val - 1
        pad_len = len(num_str)
        new_num_str = f"{new_val:0{pad_len}d}"
        new_filename = f"{prefix}{new_num_str}{ext}"
        return new_filename, True

    return orig_filename, False


def process_reels_json(json_path: str, threshold: int, execute: bool = False):
    if not os.path.exists(json_path):
        print(f"[錯誤] 找不到 JSON 檔案: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[錯誤] JSON 格式解析失敗: {e}")
            return

    if not isinstance(data, list):
        print("[錯誤] JSON 頂層不是列表 (List) 格式。")
        return

    changes = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        vid_file = item.get("local_video_file")
        aud_file = item.get("local_audio_file")

        new_vid, vid_changed = update_target_filename(vid_file, threshold)
        new_aud, aud_changed = update_target_filename(aud_file, threshold)

        if vid_changed or aud_changed:
            changes.append({
                "index": idx,
                "old_video": vid_file,
                "new_video": new_vid,
                "old_audio": aud_file,
                "new_audio": new_aud
            })
            item["local_video_file"] = new_vid
            item["local_audio_file"] = new_aud

    print("=" * 75)
    print("【reelsData JSON 影音記錄編號前移 (減 1) 工具】")
    print(f"  - 目標檔案: {os.path.abspath(json_path)}")
    print(f"  - 觸發門檻: 編號 > {threshold}")
    print(f"  - 預計修改筆數: {len(changes)} 筆記錄")
    print("=" * 75)

    if not changes:
        print(f"[提示] 沒有找到大於 {threshold} 的 local_video_file 或 local_audio_file 記錄。")
        return

    print("\n即將進行以下更動預覽：")
    for ch in changes:
        print(f"  [Item {ch['index']:>2}]")
        if ch["old_video"] != ch["new_video"]:
            print(f"    Video: {ch['old_video']:<18} -> {ch['new_video']}")
        if ch["old_audio"] != ch["new_audio"]:
            print(f"    Audio: {ch['old_audio']:<18} -> {ch['new_audio']}")

    if not execute:
        print("\n" + "-" * 75)
        confirm = input("確認將上述變更寫入 JSON 檔案？(y/N): ").strip().lower()
        if confirm != "y":
            print("[取消] 操作已中止，未覆寫檔案。")
            return

    # 寫入前先建立備份
    backup_path = f"{json_path}.bak"
    shutil.copy2(json_path, backup_path)
    print(f"\n[備份] 已建立原始檔案備份: {backup_path}")

    # 寫入更新後資料
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"【成功】已將變更寫入: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將 reelsData JSON 裡大於指定編號的 local_video_file 與 local_audio_file 減 1")
    parser.add_argument("--json", type=str, required=True, help="目標 JSON 檔案路徑 (例如: anminclinic.json)")
    parser.add_argument("--above", type=int, required=True, help="門檻數字 (例如: 4，則所有 > 004 的記錄都會減 1)")
    parser.add_argument("--yes", action="store_true", help="略過人工確認直接寫入")
    args = parser.parse_args()

    process_reels_json(
        json_path=args.json,
        threshold=args.above,
        execute=args.yes
    )