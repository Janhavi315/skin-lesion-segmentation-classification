"""Verify the HAM10000 metadata and image files."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import dataset_report

if __name__ == "__main__":
    report = dataset_report(check_readable=True)
    print(json.dumps(report, indent=2))
    if not report["metadata_exists"] or report["error"]:
        raise SystemExit(1)
