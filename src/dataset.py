
"""Dataset discovery and streaming image loading helpers."""
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from config import (
    METADATA_CSV,
    IMAGE_DIRS,
    REQUIRED_COLUMNS,
    SUPPORTED_EXTENSIONS,
)


def find_image(image_id, image_dirs=None):
    """Find an image by metadata ID across configured folders."""
    dirs = image_dirs if image_dirs is not None else IMAGE_DIRS

    for folder in dirs:
        folder = Path(folder)

        if not folder.exists():
            continue

        for ext in SUPPORTED_EXTENSIONS:
            candidate = folder / f"{image_id}{ext}"

            if candidate.is_file():
                return candidate

    return None


def load_metadata(csv_path=METADATA_CSV):
    """Load metadata and normalize supported column names."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {path}. "
            "Put HAM10000_metadata.csv in the configured dataset folder."
        )

    df = pd.read_csv(path)

    # Support both the original HAM10000 columns and your CSV format.
    if "image_id" not in df.columns and "isic_id" in df.columns:
        df = df.rename(columns={"isic_id": "image_id"})

    if "dx" not in df.columns and "diagnosis_1" in df.columns:
        df = df.rename(columns={"diagnosis_1": "dx"})

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["image_id", "dx"]).copy()

    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["dx"] = df["dx"].astype(str).str.strip()

    # Remove empty IDs or labels.
    df = df[
        (df["image_id"] != "")
        & (df["dx"] != "")
    ].copy()

    return df


def dataset_report(
    csv_path=METADATA_CSV,
    image_dirs=None,
    check_readable=False,
):
    """Report metadata, image matches, and optional readability checks."""
    selected_dirs = (
        image_dirs if image_dirs is not None else IMAGE_DIRS
    )

    result = {
        "metadata_path": str(csv_path),
        "metadata_exists": Path(csv_path).is_file(),
        "image_dirs": [str(p) for p in selected_dirs],
        "existing_image_dirs": [],
        "required_columns_present": False,
        "metadata_rows": 0,
        "usable_images": 0,
        "missing_images": [],
        "unreadable_images": [],
        "class_counts": {},
        "error": None,
    }

    result["existing_image_dirs"] = [
        str(p)
        for p in result["image_dirs"]
        if Path(p).is_dir()
    ]

    if not result["metadata_exists"]:
        result["error"] = "Metadata CSV not found."
        return result

    try:
        df = load_metadata(csv_path)

        result["required_columns_present"] = True
        result["metadata_rows"] = len(df)
        result["class_counts"] = (
            df["dx"].value_counts().sort_index().to_dict()
        )

        for image_id in df["image_id"]:
            image_path = find_image(image_id, image_dirs)

            if image_path is None:
                result["missing_images"].append(image_id)
                continue

            if check_readable:
                try:
                    with Image.open(image_path) as im:
                        im.verify()
                except (OSError, UnidentifiedImageError):
                    result["unreadable_images"].append(image_id)
                    continue

            result["usable_images"] += 1

    except Exception as exc:
        result["error"] = str(exc)

    return result
