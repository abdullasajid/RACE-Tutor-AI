from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_A_DIR = MODELS_DIR / "model_a"
MODEL_B_DIR = MODELS_DIR / "model_b"

REQUIRED_COLUMNS = ["id", "article", "question", "A", "B", "C", "D", "answer"]
OPTION_LABELS = ["A", "B", "C", "D"]
