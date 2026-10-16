import argparse
import glob
import json
import os
import numpy as np

# ---------------------------------------------------------
# 定義 MediaPipe Face Mesh 中不受表情拉扯的解剖剛性錨點 (Rigid Anchors)
# 包含：鼻樑骨中央、鼻根、外眼眶骨緣、耳前相對穩定點
# ---------------------------------------------------------
RIGID_ANCHOR_INDICES = [
    6,  # 鼻樑中心 (Nasion / Mid-nose)
    168,  # 雙眼眉間骨 (Glabella)
    197,  # 鼻樑上方骨骼點
    195,  # 鼻樑骨上部
    5,  # 鼻下緣上方骨質處
    33,  # 右眼內眼角骨質邊緣
    133,  # 右眼外側眼眶骨
    263,  # 左眼內眼角骨質邊緣
    362,  # 左眼外側眼眶骨
]


def kabsch_algorithm(source_pts, target_pts):
    """
    使用 Kabsch Algorithm (SVD) 求解最佳剛體旋轉 R 與平移 t
    滿足: R @ source_pts + t ≈ target_pts
    :param source_pts: 形狀為 (M, 3) 的原始點集 (當前幀錨點)
    :param target_pts: 形狀為 (M, 3) 的目標點集 (基準 M_0 錨點)
    :return: 旋轉矩陣 R (3, 3), 平移向量 t (3,)
    """
    # 1. 計算質心
    centroid_src = np.mean(source_pts, axis=0)
    centroid_tgt = np.mean(target_pts, axis=0)

    # 2. 去中心化
    p = source_pts - centroid_src
    q = target_pts - centroid_tgt

    # 3. 計算共變異數矩陣 H
    H = p.T @ q

    # 4. 奇異值分解 (SVD)
    U, S, Vt = np.linalg.svd(H)

    # 5. 計算旋轉矩陣 R
    R = Vt.T @ U.T

    # 6. 特殊情況處理：防止鏡像反射 (Reflection Correction, det(R) 必須為 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 7. 計算平移向量 t
    t = centroid_tgt - R @ centroid_src

    return R, t


def align_single_json(json_path, output_path=None, custom_m0=None):
    """
    讀取一個 _face.json，對每幀執行剛體對齊並儲存新 JSON
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        print(f"  [略過] 檔案中無幀資料: {json_path}")
        return False

    # -----------------------------------------------------
    # 1. 決定基準模型 M_0 (478 x 3 的坐標矩陣)
    # -----------------------------------------------------
    m0_landmarks = custom_m0
    m0_frame_sec = None

    if m0_landmarks is None:
        # 若未指定外部 M_0，自動尋找影片第一幀成功檢測到人臉的幀作為 M_0
        for frame in frames:
            if frame.get("face_detected") and frame.get("landmarks"):
                m0_landmarks = np.array(
                    [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]]
                )
                m0_frame_sec = frame["second"]
                break

    if m0_landmarks is None:
        print(f"  [錯誤] 影片中無任何有效人臉幀，無法建立 M_0: {json_path}")
        return False

    # 提取 M_0 的剛性錨點座標 (M, 3)
    target_anchors = m0_landmarks[RIGID_ANCHOR_INDICES]

    aligned_frames_count = 0

    # -----------------------------------------------------
    # 2. 遍歷每一幀，對齊到 M_0
    # -----------------------------------------------------
    for frame in frames:
        if not frame.get("face_detected") or not frame.get("landmarks"):
            frame["rigid_alignment"] = {"aligned": False, "error": "No face detected"}
            continue

        # 當前幀所有點座標 (478, 3)
        current_pts = np.array(
            [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]]
        )

        # 當前幀剛性錨點 (M, 3)
        source_anchors = current_pts[RIGID_ANCHOR_INDICES]

        # 求解剛體運動 R, t
        R, t = kabsch_algorithm(source_anchors, target_anchors)

        # 將整張臉的所有 478 個特徵點套用剛體變換: P_aligned = R @ P + t
        aligned_pts = (R @ current_pts.T).T + t

        # 計算錨點對齊後的殘差 (RMSE) 作為評估指標
        aligned_anchors = (R @ source_anchors.T).T + t
        rmse = float(np.sqrt(np.mean(np.sum((aligned_anchors - target_anchors) ** 2, axis=1))))

        # 將對齊後座標寫回結構中
        for idx, lm in enumerate(frame["landmarks"]):
            lm["aligned_x"] = float(aligned_pts[idx, 0])
            lm["aligned_y"] = float(aligned_pts[idx, 1])
            lm["aligned_z"] = float(aligned_pts[idx, 2])

        # 記錄轉換參數 (可選，供下游幾何分析與可解釋性驗證)
        frame["rigid_alignment"] = {
            "aligned": True,
            "rotation_matrix": R.tolist(),
            "translation_vector": t.tolist(),
            "anchor_rmse": rmse,
        }
        aligned_frames_count += 1

    # -----------------------------------------------------
    # 3. 儲存結果
    # -----------------------------------------------------
    data["m0_reference"] = {
        "source": "first_detected_frame" if custom_m0 is None else "custom_template",
        "reference_second": m0_frame_sec,
        "anchor_indices": RIGID_ANCHOR_INDICES,
    }

    if output_path is None:
        dir_name, file_name = os.path.split(json_path)
        base_name, _ = os.path.splitext(file_name)
        if base_name.endswith("_face"):
            output_name = f"{base_name}_aligned.json"
        else:
            output_name = f"{base_name}_face_aligned.json"
        output_path = os.path.join(dir_name, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  -> 對齊完成: {os.path.basename(output_path)} (成功對齊 {aligned_frames_count}/{len(frames)} 幀)")
    return True


def batch_align_reels(root_dir=r"C:\Users\Anthony\Download"):
    """
    搜尋目錄下所有 *_face.json (排除已對齊的 *_aligned.json) 並進行剛體對齊
    """
    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        return

    json_pattern = os.path.join(root_dir, "**", "*_face.json")
    json_files = glob.glob(json_pattern, recursive=True)

    # 過濾掉已經是 aligned 的檔案
    json_files = [f for f in json_files if not f.endswith("_face_aligned.json")]

    print(f"掃描目錄: {root_dir}")
    print(f"找到 {len(json_files)} 個面部特徵點 JSON 檔案待對齊。\n" + "-" * 60)

    for idx, json_path in enumerate(json_files, 1):
        print(f"[{idx}/{len(json_files)}] 剛體對齊中: {os.path.basename(json_path)}")
        align_single_json(json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Kabsch 剛體對齊演算法將 JSON 中的每幀特徵點對齊至 M_0 基準"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=r"C:\Users\Anthony\Downloads",
        help="單一 JSON 檔案路徑或包含多個 JSON 的資料夾 (預設: D:\\reels)",
    )
    args = parser.parse_args()

    if os.path.isfile(args.path):
        align_single_json(args.path)
    else:
        batch_align_reels(args.path)
