from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Liver Patient Dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "liver_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
TARGET_COLUMN = "Result"
TEST_SIZE = 0.2
RANDOM_STATE = 42