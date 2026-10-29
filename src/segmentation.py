"""Classical approximate skin-lesion segmentation for an educational demo.

This is not a trained segmentation network and is not suitable for diagnosis.
It combines GrabCut, a center prior, colour contrast and cleanup. It can still
fail on off-centre lesions, hair, shadows, skin tones, or low-contrast images.
"""
import cv2
import numpy as np


def segment_lesion(rgb_image):
    """Return (uint8 mask, RGB boundary overlay, explanation). White mask=255."""
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("The uploaded image is empty.")

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    if min(h, w) < 30:
        return np.zeros((h, w), np.uint8), rgb.copy(), (
            "Image is too small for approximate segmentation. Use a clearer, closer image."
        )

    # Work at a moderate resolution for stability and speed, then restore size.
    scale = min(1.0, 600.0 / max(h, w))
    sw, sh = max(30, int(w * scale)), max(30, int(h * scale))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    # Use an inset rectangle for GrabCut, but explicitly seed the outer border
    # as background and a central ellipse as probable foreground.
    mask_gc = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(2, int(min(sw, sh) * 0.035))
    mask_gc[:border, :] = cv2.GC_BGD
    mask_gc[-border:, :] = cv2.GC_BGD
    mask_gc[:, :border] = cv2.GC_BGD
    mask_gc[:, -border:] = cv2.GC_BGD
    center = (sw // 2, sh // 2)
    axes = (max(8, int(sw * 0.22)), max(8, int(sh * 0.22)))
    cv2.ellipse(mask_gc, center, axes, 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(mask_gc, center, (max(3, int(axes[0]*0.55)), max(3, int(axes[1]*0.55))),
                0, 0, 360, cv2.GC_FGD, -1)

    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask_gc, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        binary = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except cv2.error:
        # Fallback: dark/pigmented pixels relative to the image's local median.
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        threshold = min(190, max(35, int(np.percentile(l_channel, 45))))
        binary = np.where(l_channel < threshold, 255, 0).astype(np.uint8)
        binary[:border, :] = 0
        binary[-border:, :] = 0
        binary[:, :border] = 0
        binary[:, -border:] = 0

    # Suppress tiny specks and fill small holes; keep the largest reasonable
    # connected component, which usually corresponds to one central lesion.
    kernel_size = max(3, int(min(sw, sh) * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        if stats[largest, cv2.CC_STAT_AREA] >= max(20, int(sw * sh * 0.002)):
            binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    # If it selected nearly the entire frame, restrict to the central region
    # rather than presenting the full image as a lesion.
    fraction = cv2.countNonZero(binary) / float(sw * sh)
    if fraction > 0.75:
        ellipse = np.zeros_like(binary)
        cv2.ellipse(ellipse, center, (max(5, int(sw * .38)), max(5, int(sh * .38))), 0, 0, 360, 255, -1)
        binary = cv2.bitwise_and(binary, ellipse)
        fraction = cv2.countNonZero(binary) / float(sw * sh)

    if cv2.countNonZero(binary) < max(20, int(sw * sh * 0.002)):
        mask = np.zeros((h, w), np.uint8)
        overlay = rgb.copy()
        message = (
            "No clear candidate region was found. Try a close, sharp image with the lesion near the centre, "
            "even lighting, and little hair or shadow."
        )
        return mask, overlay, message

    if (sw, sh) != (w, h):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = rgb.copy()
    thickness = max(1, round(min(h, w) / 300))
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness)
    area_pct = 100.0 * cv2.countNonZero(binary) / (h * w)
    message = (
        f"Approximate candidate region generated with OpenCV (about {area_pct:.1f}% of the image). "
        "Check the white mask and green outline manually: this classical method can include background "
        "or miss lesion pixels and is not a validated segmentation model."
    )
    return binary, overlay, message


def segment_uploaded_image(rgb_image):
    """Backward-compatible name used by older versions of app.py.

    Returns the same tuple as segment_lesion: (mask, overlay, explanation).
    Pass an RGB NumPy array, not a file path or a PIL Image.
    """
    return segment_lesion(rgb_image)
