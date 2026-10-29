"""Educational Streamlit app for skin-lesion segmentation and classification.

This is a college prototype, not a medical device. It cannot diagnose disease
or predict how quickly a lesion will spread.
"""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import CLASS_NAMES, MODEL_DIR, METADATA_CSV, IMAGE_DIRS
from src.segmentation import segment_lesion
from src.image_quality import check_image_quality
from src.dataset import dataset_report, load_metadata
from src.evaluation import load_evaluation
from src.report import create_pdf_report
from src.progression import compare_lesion_areas


# --------------------------------------------------
# Page setup and shared helpers
# --------------------------------------------------

st.set_page_config(
    page_title="Skin-Lesion Segmentation & Classification",
    page_icon="🔬",
    layout="wide",
)

st.title("Skin-Lesion Segmentation & Classification")



def readable_class(code):
    """Convert a HAM10000 class code to a readable label."""
    return CLASS_NAMES.get(str(code), str(code))


def show_distribution(df):
    """Show the diagnostic category distribution from metadata."""
    if "dx" not in df.columns:
        st.info("The metadata file does not contain a 'dx' column.")
        return

    counts = df["dx"].value_counts().rename(index=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of metadata rows")
    ax.set_xlabel("Dataset category")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def normalize_quality_result(result, image_rgb):
    """Accept either a dict or list of check results from image_quality.py."""
    height, width = image_rgb.shape[:2]

    if isinstance(result, dict):
        quality = dict(result)
        quality.setdefault("warnings", [])
        quality.setdefault("width", width)
        quality.setdefault("height", height)
        quality.setdefault("brightness", float(np.mean(image_rgb)))
        quality.setdefault("sharpness", 0.0)
        return quality

    quality = {
        "warnings": [],
        "width": width,
        "height": height,
        "brightness": float(np.mean(image_rgb)),
        "sharpness": 0.0,
    }

    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message", ""))
            status = str(item.get("status", "")).lower()
            if status in {"warning", "error", "fail", "failed"}:
                quality["warnings"].append(message)
            elif message:
                quality["warnings"].append(message)

    return quality


def get_model_options():
    """List model artifacts that actually exist."""
    options = []
    for model_name, label in [
        ("svm", "SVM"),
        ("random_forest", "Random Forest"),
        ("cnn", "CNN"),
    ]:
        path = MODEL_DIR / ("cnn.keras" if model_name == "cnn"
                            else f"{model_name}.joblib")
        if path.is_file():
            options.append((model_name, label))
    return options


def mask_to_display(mask):
    """Convert a segmentation mask into a visible black/white mask."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8) * 255


def run_segmentation(image_rgb):
    """Run the existing segmentation function and standardize its outputs."""
    result = segment_lesion(image_rgb)
    if not isinstance(result, tuple) or len(result) < 3:
        raise ValueError(
            "segment_lesion() must return (mask, overlay, message). "
            "Please check src/segmentation.py."
        )
    mask, overlay, message = result[:3]
    mask = mask_to_display(mask)
    overlay = np.asarray(overlay)
    return mask, overlay, str(message)


def run_prediction(pil_image, selected_model):
    """Call the existing project prediction functions."""
    if selected_model == "cnn":
        from src.prediction import predict_cnn
        return predict_cnn(pil_image)
    from src.prediction import predict_ml
    return predict_ml(pil_image, selected_model)


def render_prediction_and_indicator():
    """Show the most recent prediction and required educational indicator."""
    last = st.session_state.get("last_prediction")
    if not last:
        risk_text = (
            "No model prediction is available. This indicator is educational "
            "and is not a clinical risk score."
        )
        st.info(risk_text)
        return risk_text

    code = str(last.get("class", "Unknown"))
    label = readable_class(code)
    st.success(f"Predicted dataset category: **{label}** (`{code}`)")

    probabilities = last.get("probabilities") or {}
    if isinstance(probabilities, dict) and probabilities:
        try:
            chart_df = pd.DataFrame({
                "Category": [readable_class(k) for k in probabilities],
                "Model score": [float(v) for v in probabilities.values()],
            }).set_index("Category")
            st.bar_chart(chart_df)
            st.caption(
                "These are model outputs, not verified clinical probabilities."
            )
        except (TypeError, ValueError):
            st.caption("The model returned scores that could not be charted.")

    # The indicator is rendered once in the main analysis section below.
    return (
        f"The model predicted {label}. This does not establish that the "
        "lesion is harmless. Seek professional evaluation for concerning "
        "or changing lesions."
    )


tab_analyze, tab_compare, tab_dataset, tab_progression = st.tabs([
    "Analyze Image",
    "Model Comparison",
    "Dataset & Model Information",
    "Area Change Comparison",
])


# ==================================================
# TAB 1: IMAGE ANALYSIS
# ==================================================

with tab_analyze:
    st.subheader("1. Upload a skin-lesion image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="skin_image_upload",
    )

    if uploaded is not None:
        try:
            pil = Image.open(uploaded).convert("RGB")
            rgb = np.asarray(pil, dtype=np.uint8)

            # Quality checks differ between versions of the project module.
            raw_quality = check_image_quality(rgb)
            quality = normalize_quality_result(raw_quality, rgb)

            mask, overlay, seg_message = run_segmentation(rgb)

            st.subheader("2. Segmentation results")
            c1, c2, c3 = st.columns(3)
            c1.image(rgb, caption="Original image", use_container_width=True)
            c2.image(
                mask,
                caption="Approximate segmentation mask",
                clamp=True,
                use_container_width=True,
            )
            c3.image(
                overlay,
                caption="Segmentation overlay",
                use_container_width=True,
            )
            st.info(seg_message)
            st.caption(
                "The mask is an approximate image-processing candidate, not "
                "a verified lesion boundary. Accuracy must be evaluated "
                "against reliable ground-truth masks."
            )

            st.subheader("3. Image-quality checks")
            warnings = quality.get("warnings", [])
            if warnings:
                for item in warnings:
                    st.write(f"• {item}")
            else:
                st.success(
                    "No warnings were returned by the configured quality "
                    "checker. This does not guarantee that the image is "
                    "suitable for medical interpretation."
                )

            with st.expander("View image measurements"):
                st.write(
                    f"Dimensions: {quality['width']} × "
                    f"{quality['height']} pixels"
                )
                st.write(f"Mean brightness: {quality['brightness']:.1f}")
                st.write(f"Sharpness measurement: {quality['sharpness']:.1f}")

            st.subheader("4. Estimated segmented area")
            area_pixels = int(np.count_nonzero(mask))
            total_pixels = int(mask.shape[0] * mask.shape[1])
            area_pct = (
                100.0 * area_pixels / total_pixels if total_pixels else None
            )

            a1, a2, a3 = st.columns(3)
            a1.metric("Candidate pixels", f"{area_pixels:,}")
            a2.metric("Total image pixels", f"{total_pixels:,}")
            a3.metric(
                "Candidate area percentage",
                f"{area_pct:.2f}%" if area_pct is not None else "Unavailable",
            )
            if area_pixels == 0:
                st.warning(
                    "The segmentation mask is empty. This does not mean the "
                    "lesion has zero area."
                )
            st.caption(
                "This is a pixel-based estimate, not a physical measurement "
                "and not proof of lesion size or growth."
            )

            st.subheader("5. Image classification")
            available = get_model_options()
            last = st.session_state.get("last_prediction")
            risk_text = (
                "No model prediction is available. This indicator is "
                "educational and is not a clinical risk score."
            )

            if not available:
                st.info(
                    "No trained model files were found. Check your dataset "
                    "and train models using `python scripts/train_models.py`."
                )
            else:
                labels = {name: label for name, label in available}
                selected = st.selectbox(
                    "Choose a trained model",
                    options=[name for name, _ in available],
                    format_func=lambda name: labels[name],
                    key="selected_classifier",
                )

                if st.button("Classify uploaded image", key="classify_image"):
                    try:
                        code, probs = run_prediction(pil, selected)
                        st.session_state["last_prediction"] = {
                            "class": str(code),
                            "probabilities": probs,
                        }
                        st.session_state["last_model"] = selected
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

                last = st.session_state.get("last_prediction")
                if last and st.session_state.get("last_model") == selected:
                    risk_text = render_prediction_and_indicator()
                elif not last:
                    st.info("Select a model and click Classify uploaded image.")
                else:
                    st.info(
                        "The displayed prediction belongs to another model. "
                        "Click Classify uploaded image to run the selected model."
                    )

            # Always show the user's requested safety wording with the actual
            # predicted class when a prediction exists.
            
            # 6. Experimental Educational Indicator
            st.subheader("6. Experimental Educational Indicator")

            last = st.session_state.get("last_prediction")

            if last:
                predicted_class = str(last.get("class", "Unknown"))

                class_labels = {
                    "nv": "Benign (Melanocytic nevus)",
                    "mel": "Melanoma",
                    "bcc": "Basal cell carcinoma",
                    "akiec": "Actinic keratoses / intraepithelial carcinoma",
                    "bkl": "Benign keratosis-like lesion",
                    "df": "Dermatofibroma",
                    "vasc": "Vascular lesion",
                    "benign": "Benign",
                }

                predicted_label = class_labels.get(
                    predicted_class.lower(), predicted_class
                )

                st.warning(
                    f"The model predicted {predicted_label}. "
                    "This does not establish that the lesion is harmless. "
                    "Seek professional evaluation for concerning or changing lesions."
                )
            else:
                st.info(
                    "Please upload an image and click 'Classify uploaded image' "
                    "to enable the educational indicator."
                )


            st.subheader("7. Download PDF report")
            if st.button("Prepare PDF report", key="prepare_pdf_report"):
                try:
                    report_bytes = create_pdf_report(
                        rgb,
                        mask,
                        overlay,
                        quality,
                        area_pixels,
                        total_pixels,
                        area_pct,
                        {"Selected model": last} if last else {},
                        risk_text,
                    )
                    st.download_button(
                        "Download PDF report",
                        data=report_bytes,
                        file_name="skin_lesion_analysis_report.pdf",
                        mime="application/pdf",
                        key="download_pdf_report",
                    )
                except Exception as exc:
                    st.error(f"Could not create PDF report: {exc}")

        except (OSError, UnidentifiedImageError, ValueError) as exc:
            st.error(
                "Could not process this image. Please choose a valid JPG, "
                f"JPEG, or PNG file. Details: {exc}"
            )
        except Exception as exc:
            st.error(f"Unexpected image-analysis error: {exc}")
    else:
        st.info(
            "Upload an image to view approximate segmentation, quality "
            "checks, area estimates, classification, and a PDF report."
        )


# ==================================================
# TAB 2: MODEL COMPARISON
# ==================================================

with tab_compare:
    st.subheader("Machine Learning versus Deep Learning")
    try:
        evaluation = load_evaluation()
    except Exception as exc:
        evaluation = None
        st.error(f"Could not load evaluation results: {exc}")

    if not evaluation or not evaluation.get("models"):
        st.info(
            "Evaluation results are not available. Verify the dataset and "
            "train/evaluate models to generate metrics."
        )
    else:
        rows = []
        for model_name, metrics in evaluation["models"].items():
            rows.append({
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "Macro precision": metrics.get("macro_precision"),
                "Macro recall": metrics.get("macro_recall"),
                "Macro F1": metrics.get("macro_f1"),
                "Test samples": metrics.get(
                    "test_samples", evaluation.get("test_samples", "See JSON")
                ),
            })
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        metric_columns = [
            c for c in [
                "Accuracy", "Macro precision", "Macro recall", "Macro F1"
            ] if c in results_df.columns
        ]
        if metric_columns:
            st.subheader("Model performance comparison")
            st.bar_chart(results_df.set_index("Model")[metric_columns])

        model_names = list(evaluation["models"].keys())
        selected_model = st.selectbox(
            "View confusion matrix for",
            options=model_names,
            key="confusion_matrix_model",
        )
        metrics = evaluation["models"][selected_model]
        cm = np.asarray(metrics.get("confusion_matrix", []))
        labels = metrics.get("labels", [])

        if cm.size and len(labels) and cm.ndim == 2:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(cm, interpolation="nearest")
            ax.set_title(f"{selected_model} confusion matrix")
            ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(
            "These metrics describe the saved evaluation results and do not "
            "establish clinical reliability."
        )
        with st.expander("Evaluation details"):
            st.json({k: v for k, v in evaluation.items() if k != "models"})


# ==================================================
# TAB 3: DATASET AND MODEL INFORMATION
# ==================================================

with tab_dataset:
    st.subheader("Dataset verification")
    st.write(f"Metadata CSV: `{METADATA_CSV}`")
    st.write("Configured image folders:")
    for folder in IMAGE_DIRS:
        st.write(f"- `{folder}`")

    if st.button("Check dataset now", key="check_dataset"):
        try:
            st.session_state["dataset_report"] = dataset_report(
                check_readable=True
            )
        except Exception as exc:
            st.error(f"Dataset verification failed: {exc}")

    report = st.session_state.get("dataset_report")
    if report:
        st.json({
            key: value[:20] if isinstance(value, list) else value
            for key, value in report.items()
        })
        if report.get("missing_images"):
            st.warning(
                f"{len(report['missing_images'])} image IDs were not found. "
                "Only the first 20 are displayed."
            )

    if Path(METADATA_CSV).is_file():
        try:
            df = load_metadata()
            st.write(f"Metadata rows with required fields: {len(df):,}")
            show_distribution(df)
        except Exception as exc:
            st.error(f"Could not read metadata: {exc}")
    else:
        st.info(
            "Dataset metadata was not found. Check the path in config.py."
        )

    st.subheader("Saved model files")
    model_files = sorted(MODEL_DIR.glob("*"))
    if model_files:
        for model_file in model_files:
            if model_file.is_file():
                st.write(
                    f"- `{model_file.name}` "
                    f"({model_file.stat().st_size:,} bytes)"
                )
    else:
        st.info("No saved model files were found.")
    st.caption("Models are not trained automatically when the app starts.")


# ==================================================
# TAB 4: EDUCATIONAL AREA-CHANGE COMPARISON
# ==================================================

with tab_progression:
    st.subheader("Experimental lesion-area change comparison")
    st.info(
        "Upload two images believed to show the same lesion on different "
        "dates. This compares segmentation-mask pixel counts only. It does "
        "not calculate disease spread speed, predict future growth, or "
        "establish a diagnosis. Use consistent imaging conditions."
    )

    col1, col2 = st.columns(2)
    with col1:
        earlier_file = st.file_uploader(
            "Earlier image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="progression_earlier_image",
        )
        earlier_date = st.date_input(
            "Date of earlier image",
            value=date.today(),
            key="progression_earlier_date",
        )
    with col2:
        later_file = st.file_uploader(
            "Later image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="progression_later_image",
        )
        later_date = st.date_input(
            "Date of later image",
            value=date.today(),
            key="progression_later_date",
        )

    if st.button("Compare segmented areas", key="compare_lesion_areas"):
        if earlier_file is None or later_file is None:
            st.error("Please upload both images.")
        elif later_date <= earlier_date:
            st.error("The later image date must be after the earlier date.")
        else:
            try:
                earlier_rgb = np.asarray(
                    Image.open(earlier_file).convert("RGB"), dtype=np.uint8
                )
                later_rgb = np.asarray(
                    Image.open(later_file).convert("RGB"), dtype=np.uint8
                )

                earlier_mask, earlier_overlay, earlier_message = run_segmentation(
                    earlier_rgb
                )
                later_mask, later_overlay, later_message = run_segmentation(
                    later_rgb
                )
                days = (later_date - earlier_date).days

                if earlier_mask.shape != later_mask.shape:
                    st.warning(
                        "The images have different dimensions. Pixel counts "
                        "may not be directly comparable. For a fairer "
                        "comparison, use similarly framed images and a "
                        "validated segmentation workflow."
                    )

                result = compare_lesion_areas(
                    earlier_mask, later_mask, days
                )

                st.write(f"Days between images: **{days}**")
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Earlier mask area",
                    f"{result['previous_area_pixels']:,} pixels",
                )
                m2.metric(
                    "Later mask area",
                    f"{result['current_area_pixels']:,} pixels",
                )
                m3.metric(
                    "Observed area change",
                    f"{result['change_percent']:.2f}%",
                )

                st.write(f"**Observed result:** {result['interpretation']}")
                st.write(
                    "**Pixel-count change per day:** "
                    f"{result['change_pixels_per_day']:.2f} pixels/day"
                )
                st.write(
                    "**Percentage change per day:** "
                    f"{result['change_percent_per_day']:.2f}%/day"
                )

                p1, p2 = st.columns(2)
                p1.image(
                    earlier_mask,
                    caption=f"Earlier mask — {earlier_date}",
                    clamp=True,
                    use_container_width=True,
                )
                p2.image(
                    later_mask,
                    caption=f"Later mask — {later_date}",
                    clamp=True,
                    use_container_width=True,
                )
                st.warning(result["warning"])
                st.caption(
                    f"Earlier segmentation note: {earlier_message} "
                    f"Later segmentation note: {later_message}"
                )
            except Exception as exc:
                st.error(f"Could not compare these images: {exc}")
