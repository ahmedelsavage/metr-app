from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
PROCESSED_DIR = ROOT / "data/processed"

DATASET = "coastalcph/lex_glue"
CONFIG = "unfair_tos"
SPLITS = {"train": "train", "val": "validation", "test": "test"}

LABEL_NAMES_FILE = RAW_DIR / "label_names.json"
REPORT_FILE = PROCESSED_DIR / "cleaning_report.json"
