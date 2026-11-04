import argparse
import glob
import json
import os
import numpy as np
from config import DEFAULT_REELS_DIR, VIDEO_PROC_DIR

DEFAULT_DIR = DEFAULT_REELS_DIR

# -------------------------------------------------------------------------
# 官方標準 Canonical Face Mesh 三角剖分索引 (468 點皮膚網格)
# 透過此快取載入，徹底避免引用 mediapipe.python 或 mediapipe.solutions 舊語法
# -------------------------------------------------------------------------
CANONICAL_TRIANGLES_FILE = str(VIDEO_PROC_DIR / "canonical_face_triangles.npy")
_CANONICAL_TRIANGLES_CACHE = None


def get_canonical_triangles():
    """
    取得標準 Canonical 三角形面片拓撲 (F, 3)。
    完全不依賴 mediapipe.python 或 mediapipe.solutions。
    若本機沒有 npy 檔案，提供自動下載或以基準拓撲初始化。
    """
    global _CANONICAL_TRIANGLES_CACHE
    if _CANONICAL_TRIANGLES_CACHE is not None:
        return _CANONICAL_TRIANGLES_CACHE

    if os.path.exists(CANONICAL_TRIANGLES_FILE):
        _CANONICAL_TRIANGLES_CACHE = np.load(CANONICAL_TRIANGLES_FILE)
        return _CANONICAL_TRIANGLES_CACHE

    # 若無本地快取，嘗試從 mediapipe.tasks 的底層數據取得，或回退至標準定義
    try:
        # 新版若有對應 connections 常數
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    except (ImportError, AttributeError):
        connections = None

    if connections is not None:
        adjacency = {}
        for edge in connections:
            u, v = edge.start, edge.end
            if u < 468 and v < 468:
                adjacency.setdefault(u, set()).add(v)
                adjacency.setdefault(v, set()).add(u)

        triangles = []
        for u in sorted(adjacency.keys()):
            for v in sorted(adjacency[u]):
                if v > u:
                    common = adjacency[u].intersection(adjacency[v])
                    for w in sorted(common):
                        if w > v:
                            triangles.append([u, v, w])

        _CANONICAL_TRIANGLES_CACHE = np.array(triangles, dtype=np.int32)
        np.save(CANONICAL_TRIANGLES_FILE, _CANONICAL_TRIANGLES_CACHE)
        return _CANONICAL_TRIANGLES_CACHE

    raise RuntimeError(
        "無法取得標準三角拓撲。請確保環境中具備 canonical_face_triangles.npy 檔案。"
    )


def compute_beltrami_for_triangle(z_tri, w_tri):
    """
    計算單個三角形從 z (Frame t) 映射到 w (Frame t+1) 的 Beltrami 係數 mu
    :param z_tri: (3,) 複數向量
    :param w_tri: (3,) 複數向量
    :return: 複數 mu (若退化或數值異常回傳 0.0 + 0.0j)
    """
    dz1 = z_tri[1] - z_tri[0]
    dz2 = z_tri[2] - z_tri[0]
    dw1 = w_tri[1] - w_tri[0]
    dw2 = w_tri[2] - w_tri[0]

    det = dz1 * np.conj(dz2) - np.conj(dz1) * dz2
    if np.abs(det) < 1e-10:
        return 0.0 + 0.0j

    alpha = (dw1 * np.conj(dz2) - dw2 * np.conj(dz1)) / det
    beta = (dz1 * dw2 - dz2 * dw1) / det

    if np.abs(alpha) < 1e-10:
        return 0.0 + 0.0j

    return beta / alpha


def process_video_quasi_conformal(aligned_json_path, canonical_triangles, output_json_path=None):
    # 若外層未指明 output_json_path，在此自行推導並再次確認
    if output_json_path is None:
        dir_name, file_name = os.path.split(aligned_json_path)
        base_name, _ = os.path.splitext(file_name)
        output_name = base_name.replace("_face_aligned", "_qc").replace("_aligned", "_qc") + ".json"
        if not output_name.endswith("_qc.json"):
            output_name = f"{base_name}_qc.json"
        output_json_path = os.path.join(dir_name, output_name)

    if os.path.exists(output_json_path):
        return True

    try:
        with open(aligned_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [錯誤] JSON 損毀無法解析: {aligned_json_path} -> {e}")
        return False

    frames = data.get("frames", [])
    if len(frames) < 2:
        print(f"  [略過] 幀數少於 2 幀: {aligned_json_path}")
        return False

    transitions_results = []
    num_canonical_triangles = len(canonical_triangles)

    for i in range(len(frames) - 1):
        f_curr = frames[i]
        f_next = frames[i + 1]

        sec_curr = f_curr.get("second", i)
        sec_next = f_next.get("second", i + 1)

        valid_curr = f_curr.get("face_detected", False) and bool(f_curr.get("landmarks"))
        valid_next = f_next.get("face_detected", False) and bool(f_next.get("landmarks"))

        # 只要其中一個 frame 沒有 face landmark，相關數值全部設為 0
        if not (valid_curr and valid_next):
            transitions_results.append({
                "from_second": sec_curr,
                "to_second": sec_next,
                "valid_transition": False,
                "teichmuller_distance": 0.0,
                "mean_dilation": 0.0,
                "max_dilation": 0.0,
                "mean_abs_mu": 0.0,
                "max_abs_mu": 0.0,
                "num_triangles": num_canonical_triangles,
                "facet_metrics": []
            })
            continue

        lms_curr = f_curr["landmarks"]
        lms_next = f_next["landmarks"]

        z_pts = np.array([
            complex(lm.get("aligned_x", lm["x"]), lm.get("aligned_y", lm["y"]))
            for lm in lms_curr[:468]
        ])
        w_pts = np.array([
            complex(lm.get("aligned_x", lm["x"]), lm.get("aligned_y", lm["y"]))
            for lm in lms_next[:468]
        ])

        mu_list = []
        dilation_list = []
        facet_data = []

        for tri_idx, tri in enumerate(canonical_triangles):
            z_tri = z_pts[tri]
            w_tri = w_pts[tri]

            mu = compute_beltrami_for_triangle(z_tri, w_tri)
            abs_mu = float(np.abs(mu))

            # 數值穩定性防護: 截斷 |mu| <= 0.999
            abs_mu_clipped = min(abs_mu, 0.999)
            dilation = (1.0 + abs_mu_clipped) / (1.0 - abs_mu_clipped)

            mu_list.append(abs_mu_clipped)
            dilation_list.append(dilation)

            facet_data.append({
                "triangle_id": int(tri_idx),
                "vertices": tri.tolist(),
                "mu_real": float(mu.real),
                "mu_imag": float(mu.imag),
                "abs_mu": abs_mu,
                "dilation": float(dilation),
                "arg_mu": float(np.angle(mu))
            })

        mu_arr = np.array(mu_list)
        dil_arr = np.array(dilation_list)

        max_mu = float(np.max(mu_arr))
        mean_mu = float(np.mean(mu_arr))
        max_dil = float(np.max(dil_arr))
        mean_dil = float(np.mean(dil_arr))

        # Teichmüller Distance
        teichmuller_dist = float(0.5 * np.log((1.0 + max_mu) / (1.0 - max_mu)))

        transitions_results.append({
            "from_second": sec_curr,
            "to_second": sec_next,
            "valid_transition": True,
            "teichmuller_distance": teichmuller_dist,
            "mean_dilation": mean_dil,
            "max_dilation": max_dil,
            "mean_abs_mu": mean_mu,
            "max_abs_mu": max_mu,
            "num_triangles": num_canonical_triangles,
            "facet_metrics": facet_data
        })

    output_data = {
        "source_file": os.path.basename(aligned_json_path),
        "topology": {
            "type": "mediapipe_canonical_facemesh",
            "total_canonical_triangles": num_canonical_triangles
        },
        "total_transitions": len(transitions_results),
        "consecutive_qc_metrics": transitions_results
    }

    if output_json_path is None:
        dir_name, file_name = os.path.split(aligned_json_path)
        base_name, _ = os.path.splitext(file_name)
        output_name = base_name.replace("_face_aligned", "_qc").replace("_aligned", "_qc") + ".json"
        if not output_name.endswith("_qc.json"):
            output_name = f"{base_name}_qc.json"
        output_json_path = os.path.join(dir_name, output_name)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  -> QC 計算完成: {os.path.basename(output_json_path)}")
    return True


def batch_compute_quasi_conformal(root_dir=DEFAULT_DIR):
    if not os.path.exists(root_dir):
        print(f"錯誤：目錄不存在 -> {root_dir}")
        return

    print("正在載入標準 Canonical 三角剖分拓撲...")
    canonical_triangles = get_canonical_triangles()
    print(f"Canonical 拓撲載入成功，共包含 {len(canonical_triangles)} 個固定面片。\n" + "-" * 60)

    pattern = os.path.join(root_dir, "**", "*_face_aligned.json")
    aligned_files = glob.glob(pattern, recursive=True)

    print(f"掃描目錄: {root_dir}")
    print(f"找到 {len(aligned_files)} 個對齊特徵點檔案。\n" + "-" * 60)

    skipped_count = 0
    processed_count = 0

    for idx, path in enumerate(aligned_files, 1):
        folder_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)

        # 預先計算預期的 QC 輸出路徑
        dir_name = os.path.dirname(path)
        base_name, _ = os.path.splitext(file_name)
        qc_output_name = base_name.replace("_face_aligned", "_qc").replace("_aligned", "_qc") + ".json"
        if not qc_output_name.endswith("_qc.json"):
            qc_output_name = f"{base_name}_qc.json"
        qc_output_path = os.path.join(dir_name, qc_output_name)

        # 檢查是否已計算過
        if os.path.exists(qc_output_path):
            print(f"[{idx}/{len(aligned_files)}] [{folder_name}] 已存在，略過: {qc_output_name}")
            skipped_count += 1
            continue

        print(f"[{idx}/{len(aligned_files)}] [{folder_name}] 正在計算 QC 特徵: {file_name}")
        success = process_video_quasi_conformal(path, canonical_triangles, output_json_path=qc_output_path)
        if success:
            processed_count += 1
        else:
            print(f"  -> 計算失敗或檔案損毀: {file_name}")

    print("\n" + "=" * 60)
    print("【執行統計】")
    print(f"總檔案數: {len(aligned_files)}")
    print(f"本次新計算: {processed_count}")
    print(f"已存在略過: {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="計算連續幀之 Beltrami 係數與 Teichmuller 距離 (支援 MediaPipe Tasks 新架構)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_DIR,
        help=f"單一 *_face_aligned.json 檔案路徑或包含多個檔案的目錄 (預設: {DEFAULT_DIR})",
    )
    args = parser.parse_args()

    # 確保先載入 Canonical 拓撲
    triangles = get_canonical_triangles()

    if os.path.isfile(args.path):
        process_video_quasi_conformal(args.path, triangles)
    else:
        batch_compute_quasi_conformal(args.path)