"""Transparent heuristic checks; these do not establish medical image suitability."""
import cv2
import numpy as np

def check_image_quality(rgb):
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []
    if brightness < 55:
        warnings.append(f"Image may be too dark (mean brightness {brightness:.1f}; threshold < 55).")
    if sharpness < 35:
        warnings.append(f"Image may be blurry (Laplacian variance {sharpness:.1f}; threshold < 35).")
    if min(h, w) < 128:
        warnings.append(f"Image resolution may be insufficient ({w} × {h}; minimum side threshold 128 px).")
    if not warnings:
        warnings.append("Image quality appears acceptable under these simple checks.")
    return {"width": w, "height": h, "brightness": brightness, "sharpness": sharpness, "warnings": warnings}
