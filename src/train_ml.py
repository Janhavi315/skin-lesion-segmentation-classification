"""Train SVM and Random Forest from a manageable dataset subset."""
import json, time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from config import MODEL_DIR, OUTPUT_DIR, DEFAULT_TRAIN_LIMIT, TEST_SIZE, RANDOM_SEED
from src.dataset import load_metadata, find_image
from src.features import handcrafted_features
from PIL import Image

def build_feature_dataset(limit=DEFAULT_TRAIN_LIMIT):
    df = load_metadata()
    # Reproducible stratified-ish sample; cap before reading image pixels.
    if limit and len(df) > limit:
        df = df.groupby("dx", group_keys=False).apply(
            lambda group: group.sample(n=max(1, min(len(group), round(limit * len(group) / len(load_metadata())))),
                                       random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        if len(df) > limit:
            df = df.sample(n=limit, random_state=RANDOM_SEED).reset_index(drop=True)
    X, y, kept_ids = [], [], []
    for row in df.itertuples(index=False):
        path = find_image(str(row.image_id))
        if path is None:
            continue
        try:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.uint8)
            X.append(handcrafted_features(rgb))
            y.append(str(row.dx))
            kept_ids.append(str(row.image_id))
        except (OSError, ValueError):
            continue
    if len(X) < 20:
        raise RuntimeError(f"Only {len(X)} readable images found. Check the dataset folders.")
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(kept_ids)

def _metrics(y_true, y_pred, labels):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "labels": list(labels)}

def train_ml(limit=DEFAULT_TRAIN_LIMIT):
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y, ids = build_feature_dataset(limit)
    # IDs are unique image IDs in HAM10000; use them to keep records together.
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    models = {
        "svm": make_pipeline(StandardScaler(), SVC(C=2.0, probability=True, class_weight="balanced",
                                                   random_state=RANDOM_SEED)),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=2,
                                                 class_weight="balanced_subsample", n_jobs=2,
                                                 random_state=RANDOM_SEED),
    }
    report = {"dataset_subset": len(X), "train_samples": len(X_train), "test_samples": len(X_test),
              "split": "80/20 stratified; random_state=42", "feature_description": "RGB/HSV histograms and grayscale statistics",
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        report["models"][name] = _metrics(y_test, pred, sorted(np.unique(y)))
    report["class_names"] = sorted(np.unique(y))
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUTPUT_DIR / "test_predictions.npz", y_test=y_test, ids_test=ids_test)
    return report

if __name__ == "__main__":
    print(json.dumps(train_ml(), indent=2))
