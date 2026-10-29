"""Load and format actual evaluation results created by training scripts."""
import json
from pathlib import Path
from config import OUTPUT_DIR

def load_evaluation():
    path = OUTPUT_DIR / "evaluation.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
