"""Load saved model artifacts and produce real model predictions."""
import json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from config import MODEL_DIR, IMAGE_SIZE
from src.preprocessing import image_to_array
from src.features import handcrafted_features

def predict_ml(image, model_name):
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}. Run scripts/train_models.py first.")
    rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.uint8)
    features = handcrafted_features(rgb).reshape(1, -1)
    model = joblib.load(path)
    pred = str(model.predict(features)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(features)[0]
        probs = {str(c): float(p) for c, p in zip(model.classes_, values)}
    return pred, probs

def predict_cnn(image):
    import tensorflow as tf
    model_path = MODEL_DIR / "cnn.keras"
    classes_path = MODEL_DIR / "cnn_classes.json"
    if not model_path.exists() or not classes_path.exists():
        raise FileNotFoundError("CNN model not found. Run scripts/train_models.py --cnn first.")
    model = tf.keras.models.load_model(model_path)
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    x = image_to_array(image, IMAGE_SIZE)[None, ...]
    probs_array = model.predict(x, verbose=0)[0]
    probs = {str(c): float(p) for c, p in zip(classes, probs_array)}
    pred = str(classes[int(np.argmax(probs_array))])
    return pred, probs
