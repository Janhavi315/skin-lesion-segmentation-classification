
"""Train an improved CNN for skin-lesion classification."""

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from config import (
    MODEL_DIR,
    OUTPUT_DIR,
    DEFAULT_TRAIN_LIMIT,
    IMAGE_SIZE,
    BATCH_SIZE,
    CNN_EPOCHS,
    RANDOM_SEED,
)
from src.dataset import load_metadata, find_image
from src.preprocessing import load_array


def train_cnn(limit=DEFAULT_TRAIN_LIMIT, epochs=CNN_EPOCHS):

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 1. Load dataset metadata
    df = load_metadata()

    if limit and len(df) > limit:
        df = (
            df.groupby("dx", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(
                        1,
                        min(
                            len(g),
                            round(limit * len(g) / len(df)),
                        ),
                    ),
                    random_state=RANDOM_SEED,
                )
            )
            .reset_index(drop=True)
        )

        if len(df) > limit:
            df = df.sample(
                n=limit, random_state=RANDOM_SEED
            ).reset_index(drop=True)

    # 2. Collect valid image paths and labels
    paths, labels = [], []

    for row in df.itertuples(index=False):
        path = find_image(str(row.image_id))

        if path is not None:
            paths.append(str(path))
            labels.append(str(row.dx))

    if len(paths) < 20:
        raise RuntimeError(
            f"Only {len(paths)} images found. Check dataset paths."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # 3. Split into training, validation and test sets
    train_val, test = train_test_split(
        np.arange(len(paths)),
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.1875,
        random_state=RANDOM_SEED,
        stratify=y[train_val],
    )
    # Final split: 65% training, 15% validation, 20% test

    # 4. Stream images in batches to save RAM
    def generator(indices, training=False):
        indices = np.asarray(indices)

        while True:
            order = indices.copy()

            if training:
                np.random.shuffle(order)

            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start:start + BATCH_SIZE]
                xs, ys = [], []

                for i in batch:
                    try:
                        image = load_array(paths[i], IMAGE_SIZE)
                        xs.append(image)
                        ys.append(y[i])
                    except Exception as error:
                        print(
                            f"Skipping unreadable image: {paths[i]} "
                            f"({error})"
                        )

                if xs:
                    yield (
                        np.asarray(xs, dtype=np.float32),
                        np.asarray(ys, dtype=np.int32),
                    )

            if not training:
                # Validation/test batches can be restarted when requested.
                continue

    def steps(indices):
        return max(1, int(np.ceil(len(indices) / BATCH_SIZE)))

    # 5. Calculate weights for imbalanced classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y[train]),
        y=y[train],
    )

    weights = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y[train]), class_weights)
    }

    # 6. Data augmentation and deeper CNN
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        augmentation,

        # CNN Block 1
        tf.keras.layers.Conv2D(
            32, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.15),

        # CNN Block 2
        tf.keras.layers.Conv2D(
            64, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),

        # CNN Block 3
        tf.keras.layers.Conv2D(
            128, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        # CNN Block 4
        tf.keras.layers.Conv2D(
            256, 3, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.GlobalAveragePooling2D(),

        # Classification head
        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        ),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(
            len(encoder.classes_), activation="softmax"
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "cnn.keras"

    # 7. Callbacks to reduce overfitting
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 8. Train the CNN
    history = model.fit(
        generator(train, training=True),
        steps_per_epoch=steps(train),
        epochs=epochs,
        validation_data=generator(val),
        validation_steps=steps(val),
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint before final testing
    model = tf.keras.models.load_model(model_path)

    # 9. Evaluate on the untouched test set
    true, predictions = [], []

    test_gen = generator(test)

    for _ in range(steps(test)):
        xb, yb = next(test_gen)
        probabilities = model.predict(xb, verbose=0)

        predictions.extend(np.argmax(probabilities, axis=1))
        true.extend(yb)

    # A skipped unreadable image can make the final test set smaller.
    true = np.asarray(true)
    predictions = np.asarray(predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        average="macro",
        zero_division=0,
    )

    result = {
        "accuracy": float(accuracy_score(true, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            true,
            predictions,
            labels=np.arange(len(encoder.classes_)),
        ).tolist(),
        "labels": encoder.classes_.tolist(),
        "dataset_subset": len(paths),
        "train_samples": len(train),
        "validation_samples": len(val),
        "test_samples": len(true),
        "split": "65/15/20 stratified",
        "image_size": list(IMAGE_SIZE),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 10. Save class labels and evaluation results
    (MODEL_DIR / "cnn_classes.json").write_text(
        json.dumps(encoder.classes_.tolist()),
        encoding="utf-8",
    )

    report_path = OUTPUT_DIR / "evaluation.json"

    if report_path.exists():
        old = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        old = {"models": {}}

    old.setdefault("models", {})["cnn"] = result
    old["cnn_details"] = result

    report_path.write_text(
        json.dumps(old, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    train_cnn()
