from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(PROJECT_DIR / ".cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument(
        "--variant",
        choices=["baseline", "regularized", "high_capacity"],
        default="regularized",
    )
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(
    validation_size: float,
    seed: int,
    limit_train: int | None,
) -> tuple[np.ndarray, ...]:
    (x_full, y_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    source_indices = np.arange(len(y_full))

    if limit_train is not None:
        if limit_train < 100:
            raise ValueError("--limit-train must be at least 100.")
        indices, _ = train_test_split(
            np.arange(len(y_full)),
            train_size=min(limit_train, len(y_full)),
            stratify=y_full,
            random_state=seed,
        )
        x_full, y_full = x_full[indices], y_full[indices]
        source_indices = source_indices[indices]

    groups = source_indices // 100
    splitter = StratifiedGroupKFold(
        n_splits=max(2, round(1.0 / validation_size)),
        shuffle=True,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        splitter.split(x_full, y_full, groups=groups)
    )
    if np.intersect1d(groups[train_indices], groups[validation_indices]).size:
        raise RuntimeError("Training and validation groups overlap.")

    x_train, y_train = x_full[train_indices], y_full[train_indices]
    x_val, y_val = x_full[validation_indices], y_full[validation_indices]

    x_train = x_train[..., np.newaxis].astype("float32") / 255.0
    x_val = x_val[..., np.newaxis].astype("float32") / 255.0
    x_test = x_test[..., np.newaxis].astype("float32") / 255.0
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model(variant: str = "regularized") -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1), name="image")

    if variant == "baseline":
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
    else:
        augmentation = keras.Sequential(
            [
                layers.RandomTranslation(0.05, 0.05),
                layers.RandomRotation(0.03),
            ],
            name="data_augmentation",
        )
        x = augmentation(inputs)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

        if variant == "high_capacity":
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.4)(x)
        else:
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class")(x)
    return keras.Model(inputs, outputs, name=f"fashion_mnist_{variant}")


def compile_model(model: keras.Model) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def plot_history(history: keras.callbacks.History, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def save_evaluation(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        f"Test loss: {test_loss:.4f}\nTest accuracy: {test_accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        xticks_rotation=45,
        ax=axis,
        colorbar=False,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {metrics['test_macro_f1']:.4f}")
    return metrics


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(
        args.validation_size, args.seed, args.limit_train
    )
    print(
        f"Data ready: train={len(y_train)}, validation={len(y_val)}, "
        f"test={len(y_test)}"
    )

    if args.download_only:
        return

    output_dir = PROJECT_DIR / "results" / "cnn"
    model_dir = PROJECT_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant)
    compile_model(model)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            model_dir / f"cnn_fashion_mnist_{args.variant}.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, output_dir)
    save_evaluation(model, x_test, y_test, output_dir)
    model.save(model_dir / "cnn_fashion_mnist.keras")
    model.save_weights(model_dir / "cnn_fashion_mnist.weights.h5")


if __name__ == "__main__":
    main()
