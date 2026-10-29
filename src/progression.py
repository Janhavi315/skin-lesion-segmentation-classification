
"""Educational comparison of lesion segmentation areas."""

import numpy as np


def compare_lesion_areas(previous_mask, current_mask, days_between):
    """Compare two masks and calculate observed area-change rates."""

    if days_between <= 0:
        raise ValueError("The second image date must be later than the first.")

    previous_area = int(np.count_nonzero(previous_mask))
    current_area = int(np.count_nonzero(current_mask))

    if previous_area == 0:
        raise ValueError(
            "The previous segmentation mask is empty. "
            "Area change cannot be calculated."
        )

    change_pixels = current_area - previous_area
    change_percent = (change_pixels / previous_area) * 100

    return {
        "previous_area_pixels": previous_area,
        "current_area_pixels": current_area,
        "change_pixels": change_pixels,
        "change_percent": round(change_percent, 2),
        "change_pixels_per_day": round(change_pixels / days_between, 2),
        "change_percent_per_day": round(change_percent / days_between, 2),
        "days_between": days_between,
        "interpretation": (
            "Segmented area increased"
            if change_pixels > 0
            else "Segmented area decreased"
            if change_pixels < 0
            else "Segmented area unchanged"
        ),
        "warning": (
            "This is an educational comparison of image masks, "
            "not a prediction of disease spread or a medical diagnosis."
        ),
    }
