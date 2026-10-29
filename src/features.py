"""Small, documented image features for baseline ML models."""
import cv2
import numpy as np

def handcrafted_features(rgb):
    """Compact color histogram + texture/shape descriptors; avoids huge flattened arrays."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    feats = []
    for channel, bins, limits in [(rgb, 8, [0, 256]), (hsv, 8, [0, 256])]:
        for c in range(3):
            hist = cv2.calcHist([channel], [c], None, [bins], limits).flatten()
            hist = hist / max(float(hist.sum()), 1.0)
            feats.extend(hist.tolist())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feats.extend([
        float(np.mean(gray) / 255.0),
        float(np.std(gray) / 255.0),
        float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0),
    ])
    return np.asarray(feats, dtype=np.float32)
