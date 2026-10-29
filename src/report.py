"""Generate a dynamic, professionally formatted educational PDF report.

This report is for a college prototype only. It is not a clinical report,
medical diagnosis, or substitute for assessment by a qualified clinician.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
    HRFlowable,
    KeepTogether,
)


def _to_pil(image, mode=None):
    """Convert PIL/NumPy input to a PIL image."""
    if isinstance(image, Image.Image):
        result = image.copy()
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255
            array = np.nan_to_num(array)
            array = np.clip(array, 0, 255).astype(np.uint8)
        result = Image.fromarray(array)
    if mode:
        result = result.convert(mode)
    return result


def _image_flowable(image, max_width=75 * mm, max_height=52 * mm):
    """Create an embedded report image scaled to fit a fixed box."""
    try:
        pil = _to_pil(image)
        buffer = BytesIO()
        pil.save(buffer, format="PNG")
        buffer.seek(0)
        img = ReportImage(buffer)
        scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        img.hAlign = "CENTER"
        return img
    except Exception:
        return Paragraph("Image could not be embedded.", getSampleStyleSheet()["BodyText"])


def _extract_prediction(models):
    """Accept the app's current prediction dictionary or a plain dictionary."""
    if not isinstance(models, dict):
        return "Not available", "Not available", {}
    candidate = models.get("Selected model", models)
    if not isinstance(candidate, dict):
        return "Not available", "Not available", {}
    predicted = candidate.get("class", candidate.get("predicted_class", "Not available"))
    scores = candidate.get("probabilities", candidate.get("scores", {})) or {}
    model_name = models.get("model_name", models.get("selected_model", "Not recorded"))
    return str(predicted), str(model_name), scores if isinstance(scores, dict) else {}


def create_pdf_report(
    original_image,
    mask,
    overlay,
    quality=None,
    area_pixels=None,
    total_pixels=None,
    area_pct=None,
    models=None,
    risk_text=None,
    **extra,
):
    """Return a complete PDF report as bytes, generated from current app results.

    The function signature matches the call used in app.py. Optional extra
    fields can be passed later without breaking existing callers.
    """
    quality = quality if isinstance(quality, dict) else {}
    models = models if isinstance(models, dict) else {}
    predicted_code, model_name, scores = _extract_prediction(models)

    class_labels = {
        "nv": "Melanocytic nevus (dataset category)",
        "mel": "Melanoma (dataset category)",
        "bcc": "Basal cell carcinoma (dataset category)",
        "akiec": "Actinic keratoses / intraepithelial carcinoma (dataset category)",
        "bkl": "Benign keratosis-like lesion (dataset category)",
        "df": "Dermatofibroma (dataset category)",
        "vasc": "Vascular lesion (dataset category)",
        "benign": "Benign (model label)",
    }
    predicted_label = class_labels.get(predicted_code.lower(), predicted_code)

    if risk_text is None:
        risk_text = (
            f"The model predicted {predicted_label}. This does not establish "
            "that the lesion is harmless. Seek professional evaluation for "
            "concerning or changing lesions."
        )

    timestamp = datetime.now().astimezone().strftime("%d %B %Y, %I:%M %p %Z")
    original = _to_pil(original_image, "RGB")
    width, height = original.size

    if area_pixels is None:
        mask_arr = np.asarray(_to_pil(mask).convert("L"))
        area_pixels = int(np.count_nonzero(mask_arr))
    if total_pixels is None:
        total_pixels = int(width * height)
    if area_pct is None:
        area_pct = (100.0 * area_pixels / total_pixels) if total_pixels else 0.0

    warnings = quality.get("warnings", [])
    if not isinstance(warnings, (list, tuple)):
        warnings = [str(warnings)]
    if not warnings:
        quality_summary = "No warnings were returned by the configured basic checks."
    else:
        quality_summary = " ".join(str(item) for item in warnings)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title="Skin-Lesion Image Analysis — Educational Report",
        author="Skin-Lesion Segmentation & Classification College Project",
        subject="Experimental image-analysis summary; not a clinical report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4B5563"),
    ))
    styles.add(ParagraphStyle(
        name="Safety",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        borderColor=colors.HexColor("#B45309"),
        borderWidth=0.8,
        borderPadding=7,
        backColor=colors.HexColor("#FFF7ED"),
        spaceBefore=4,
        spaceAfter=7,
    ))

    story = []
    story.append(Paragraph("SKIN-LESION IMAGE ANALYSIS", styles["ReportTitle"]))
    story.append(Paragraph(
        "Dynamic educational report • Generated from the current application results",
        styles["Subtitle"],
    ))
    story.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#B7C9E2"),
        spaceBefore=2, spaceAfter=8
    ))

    metadata_rows = [
        [Paragraph("<b>Report generated</b>", styles["BodyText"]), timestamp],
        [Paragraph("<b>Project</b>", styles["BodyText"]), "Skin-Lesion Segmentation & Classification"],
        [Paragraph("<b>Report type</b>", styles["BodyText"]), "Experimental educational image-analysis summary"],
        [Paragraph("<b>Image dimensions</b>", styles["BodyText"]), f"{width} × {height} pixels"],
    ]
    meta_table = Table(metadata_rows, colWidths=[42 * mm, 125 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([meta_table, Spacer(1, 7)])

    story.append(Paragraph("1. Important notice", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>This is not a clinical report.</b> The application is a college "
        "prototype and is not a medical device. Its segmentation, image-quality "
        "checks and model classifications are experimental. They must not be "
        "used to diagnose a condition, rule out disease, or decide treatment. "
        "A qualified healthcare professional must assess concerning or changing lesions.",
        styles["Safety"],
    ))

    story.append(Paragraph("2. Image and segmentation review", styles["SectionHeading"]))
    image_table = Table(
        [[
            _image_flowable(original_image),
            _image_flowable(mask),
            _image_flowable(overlay),
        ],
         [
            Paragraph("<b>Original image</b>", styles["SmallText"]),
            Paragraph("<b>Candidate mask</b>", styles["SmallText"]),
            Paragraph("<b>Overlay</b>", styles["SmallText"]),
         ]],
        colWidths=[56 * mm, 56 * mm, 56 * mm],
    )
    image_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "The displayed mask is an algorithm-generated candidate region. "
        "It is not a verified lesion boundary. Visual inspection alone does "
        "not establish segmentation accuracy.",
        styles["BodyText"],
    ))

    story.append(Paragraph("3. Image measurements", styles["SectionHeading"]))
    measurements = [
        ["Measurement", "Result"],
        ["Candidate mask area", f"{int(area_pixels):,} pixels"],
        ["Total image area", f"{int(total_pixels):,} pixels"],
        ["Candidate area / image area", f"{float(area_pct):.2f}%"],
        ["Mean brightness (0–255)", f"{float(quality.get('brightness', np.mean(np.asarray(original_image)))):.2f}"],
        ["Sharpness measure", str(quality.get("sharpness", "Not available"))],
        ["Basic image-quality notes", quality_summary],
    ]
    measurement_table = Table(measurements, colWidths=[58 * mm, 110 * mm], repeatRows=1)
    measurement_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(measurement_table)
    story.append(Paragraph(
        "Area is calculated in image pixels, not millimetres or centimetres. "
        "It depends on framing, image resolution and segmentation behaviour; "
        "it is not a physical lesion-size measurement.",
        styles["SmallText"],
    ))

    story.append(Paragraph("4. Experimental model output", styles["SectionHeading"]))
    model_rows = [
        ["Field", "Result"],
        ["Selected model", model_name],
        ["Predicted dataset category", predicted_label],
        ["Raw model label", predicted_code],
    ]
    if scores:
        for key, value in scores.items():
            try:
                display_value = f"{float(value):.4f}"
            except (TypeError, ValueError):
                display_value = str(value)
            model_rows.append([f"Model score — {class_labels.get(str(key).lower(), str(key))}",
                               display_value])
    model_table = Table(model_rows, colWidths=[58 * mm, 110 * mm], repeatRows=1)
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(model_table)
    story.append(Paragraph(
        "Model scores, if provided, are outputs of this experimental model. "
        "They are not verified clinical probabilities or a measure of disease severity.",
        styles["SmallText"],
    ))

    story.append(Paragraph("5. Experimental educational indicator", styles["SectionHeading"]))
    story.append(Paragraph(str(risk_text), styles["Safety"]))

    story.append(Paragraph("6. Interpretation and next steps", styles["SectionHeading"]))
    story.append(Paragraph(
        "This report records the image-processing outputs produced by the "
        "application at the time shown above. Results may be affected by image "
        "quality, lighting, camera distance, skin appearance, dataset limits, "
        "and model limitations. If a lesion is new, changing, bleeding, painful, "
        "or otherwise concerning, seek evaluation from a qualified healthcare professional. "
        "Do not delay care because of a benign-looking model output.",
        styles["BodyText"],
    ))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(
        width="100%", thickness=0.6, color=colors.HexColor("#B7C9E2"),
        spaceBefore=4, spaceAfter=5
    ))
    story.append(Paragraph(
        "Generated automatically by the Skin-Lesion Segmentation & Classification "
        "college mini-project. This document is an educational output only and "
        "must not be represented as a clinician-issued report.",
        styles["SmallText"],
    ))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(18 * mm, 9 * mm, "Educational prototype — not for clinical use")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()
