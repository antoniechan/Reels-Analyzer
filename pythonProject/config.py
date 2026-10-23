# pythonProject/config.py
from pathlib import Path


def get_project_root() -> Path:
    """從目前檔案往上遍歷，找到包含 .git 或特定資料夾的專案根目錄"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pythonProject").exists():
            return parent
    return current.parent


# 1. 專案根目錄 (例如 C:\...\Reels-Analyzer)
ROOT_DIR = get_project_root()

# 2. 定義各子資料夾目錄 (Path 物件)
PYTHON_PROJECT_DIR = ROOT_DIR / "pythonProject"
ACCOUNT_DATA_DIR = PYTHON_PROJECT_DIR / "accountsData"
REELS_DATA_DIR = PYTHON_PROJECT_DIR / "reelsData"
ACCOUNT_PROC_DIR = PYTHON_PROJECT_DIR / "accountsProcessing"
VIDEO_PROC_DIR = PYTHON_PROJECT_DIR / "videoProcessing"
AUDIO_PROC_DIR = PYTHON_PROJECT_DIR / "audioProcessing"
DATA_PROC_DIR = PYTHON_PROJECT_DIR / "dataProcessing"
CROSS_MODAL_PROC_DIR = PYTHON_PROJECT_DIR / "crossModalAnalysis"
TRANSFORMER_DIR = PYTHON_PROJECT_DIR / "crossModalTransformer"

# 3. 常用檔案路徑常數
DEFAULT_REELS_DIR = Path(r"D:\reels")
DEFAULT_VIDEO_CLASS_JSON = DATA_PROC_DIR / "video_classification_summary.json"
MODEL_FILE_NAME = VIDEO_PROC_DIR / "face_landmarker.task"
CANONICAL_TRIANGLES_FILE = VIDEO_PROC_DIR / "canonical_face_triangles.npy"
