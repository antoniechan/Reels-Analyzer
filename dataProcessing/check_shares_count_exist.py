#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:\\reelsData 欄位完整性檢查腳本 (Check "sharesCount" Field)
----------------------------------------------------------
功能：
1. 遍歷 D:\\reelsData 下的所有 .json 檔案（支援遞迴搜尋子資料夾或單層目錄）。
2. 檢查每筆資料（dict）是否包含鍵值 "sharesCount"。
3. 區分統計：
   - 正常包含且有有效數值（例如 >= 0 或非 None）
   - 包含該鍵但值為 None（null）
   - 完全缺少 "sharesCount" 鍵（Missing Key）
4. 在終端機輸出各檔案異常明細與整體統計總結，並匯出 JSON 稽核報告。
"""

import re
import json
from pathlib import Path

# 目標目錄
TARGET_DIR = Path(r"D:\reelsData")
REPORT_PATH = Path(__file__).resolve().parent / "reelsdata_sharescount_audit.json"

# 目標檢查欄位名稱
TARGET_FIELD = "sharesCount"


def load_json_safe(json_path: Path):
    """安全讀取 JSON 檔案（容錯控制字元）"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f, strict=False)
    except Exception:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content_cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
        return json.loads(content_cleaned, strict=False)


def audit_shares_count(root_dir: Path):
    print("=" * 80)
    print(f"🔍 開始檢查目錄下所有 JSON 的 '{TARGET_FIELD}' 欄位: {root_dir}")
    print("=" * 80)

    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[-] 錯誤：找不到指定目錄 {root_dir}")
        return

    # 搜尋所有 .json 檔案（依字母排序）
    json_files = sorted(root_dir.rglob("*.json"), key=lambda p: str(p).lower())
    if not json_files:
        print(f"[-] 在 {root_dir} 下找不到任何 .json 檔案。")
        return

    total_files = len(json_files)
    total_records = 0
    valid_count = 0
    null_value_count = 0
    missing_key_count = 0

    problem_files = []
    unreadable_files = []

    for jf in json_files:
        rel_name = str(jf.relative_to(root_dir))
        try:
            data = load_json_safe(jf)
        except Exception as e:
            unreadable_files.append({"file": rel_name, "error": str(e)})
            continue

        # 統一轉換為 list 處理（相容單筆 dict 或 list[dict]）
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            unreadable_files.append({"file": rel_name, "error": f"非預期的 JSON 根結構: {type(data).__name__}"})
            continue

        file_total = len(records)
        file_missing = []
        file_null = []

        for idx, item in enumerate(records, 1):
            if not isinstance(item, dict):
                continue

            total_records += 1
            item_id = item.get("video_id") or item.get("shortCode") or item.get("id") or f"index_{idx}"

            if TARGET_FIELD not in item:
                missing_key_count += 1
                file_missing.append(str(item_id))
            elif item[TARGET_FIELD] is None:
                null_value_count += 1
                file_null.append(str(item_id))
            else:
                valid_count += 1

        if file_missing or file_null:
            problem_files.append({
                "file": rel_name,
                "total_in_file": file_total,
                "missing_key_count": len(file_missing),
                "null_value_count": len(file_null),
                "missing_items": file_missing,
                "null_items": file_null,
            })

    # 輸出有缺漏的檔案明細
    if problem_files:
        print(f"\n⚠️ 發現 {len(problem_files)} 個檔案存在缺少 '{TARGET_FIELD}' 或值為 null 的資料：")
        print("-" * 80)
        for pf in problem_files:
            print(f"📄 {pf['file']} (共 {pf['total_in_file']} 筆)")
            if pf["missing_key_count"] > 0:
                sample_missing = ", ".join(pf["missing_items"][:10])
                more_m = f" ...等共 {pf['missing_key_count']} 筆" if pf["missing_key_count"] > 10 else ""
                print(f"    ❌ 缺少 '{TARGET_FIELD}' 鍵 ({pf['missing_key_count']} 筆): {sample_missing}{more_m}")
            if pf["null_value_count"] > 0:
                sample_null = ", ".join(pf["null_items"][:10])
                more_n = f" ...等共 {pf['null_value_count']} 筆" if pf["null_value_count"] > 10 else ""
                print(f"    🔸 '{TARGET_FIELD}' 為 null ({pf['null_value_count']} 筆): {sample_null}{more_n}")
        print("-" * 80)
    else:
        print(f"\n✅ 太棒了！所有檔案的每筆資料均完整包含 '{TARGET_FIELD}' 欄位且無 null 值！")

    if unreadable_files:
        print(f"\n[-] 有 {len(unreadable_files)} 個檔案無法解析：")
        for uf in unreadable_files:
            print(f"    - {uf['file']}: {uf['error']}")

    # 輸出總結統計
    print("\n" + "=" * 80)
    print(f"📊 '{TARGET_FIELD}' 欄位檢查總結報告")
    print("=" * 80)
    print(f"  • 掃描 JSON 檔案總數:                 {total_files:>6} 個")
    print(f"  • 檢查資料筆數總計 (Total Records):   {total_records:>6} 筆")
    print(f"  • ✅ 完整包含且有數值 (Valid):        {valid_count:>6} 筆")
    print(f"  • 🔸 包含欄位但值為 null (None):      {null_value_count:>6} 筆")
    print(f"  • ❌ 完全缺少 '{TARGET_FIELD}' 欄位:   {missing_key_count:>6} 筆")
    print("=" * 80)

    # 匯出 JSON 稽核報告
    report_data = {
        "target_dir": str(root_dir),
        "checked_field": TARGET_FIELD,
        "summary": {
            "total_files": total_files,
            "total_records": total_records,
            "valid_count": valid_count,
            "null_value_count": null_value_count,
            "missing_key_count": missing_key_count,
            "problem_files_count": len(problem_files),
            "unreadable_files_count": len(unreadable_files),
        },
        "problem_files": problem_files,
        "unreadable_files": unreadable_files,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"[*] 詳細稽核報告已儲存至: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    audit_shares_count(TARGET_DIR)
