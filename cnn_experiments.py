from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(SCRIPT_DIR / ".cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from cnn_pipeline import (
    PROJECT_DIR,
    build_model,
    compile_model,
    load_data,
    plot_history,
    save_evaluation,
    set_reproducibility,
)


VARIANTS = ("baseline", "regularized", "high_capacity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Fashion-MNIST CNN variants.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_callbacks(checkpoint: Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True
        ),
    ]


def plot_comparison(rows: list[dict[str, float | int | str]], output: Path) -> None:
    names = [str(row["variant"]) for row in rows]
    train = [float(row["best_train_accuracy"]) for row in rows]
    validation = [float(row["best_validation_accuracy"]) for row in rows]
    positions = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(positions - 0.18, train, width=0.36, label="Entrenamiento")
    axis.bar(positions + 0.18, validation, width=0.36, label="Validación")
    axis.set_xticks(positions, names)
    axis.set_ylim(0.7, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Comparación de experimentos CNN")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(0.2, args.seed, None)

    results_dir = PROJECT_DIR / "results" / "cnn"
    experiments_dir = results_dir / "experiments"
    model_dir = PROJECT_DIR / "models"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for offset, variant in enumerate(VARIANTS):
        keras.backend.clear_session()
        set_reproducibility(args.seed + offset)
        variant_dir = experiments_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = model_dir / f"cnn_{variant}.keras"

        model = build_model(variant)
        compile_model(model)
        print(f"\nTraining variant: {variant}")
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=create_callbacks(checkpoint),
            verbose=2,
        )
        plot_history(history, variant_dir)
        history_data = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        (variant_dir / "history.json").write_text(
            json.dumps(history_data, indent=2), encoding="utf-8"
        )

        best_index = int(np.argmin(history.history["val_loss"]))
        rows.append(
            {
                "variant": variant,
                "parameters": model.count_params(),
                "epochs_run": len(history.history["loss"]),
                "best_epoch": best_index + 1,
                "best_train_loss": history.history["loss"][best_index],
                "best_validation_loss": history.history["val_loss"][best_index],
                "best_train_accuracy": history.history["accuracy"][best_index],
                "best_validation_accuracy": history.history["val_accuracy"][best_index],
            }
        )

    winner = min(rows, key=lambda row: float(row["best_validation_loss"]))
    winner_name = str(winner["variant"])
    winner_model = keras.models.load_model(model_dir / f"cnn_{winner_name}.keras")
    final_model_path = model_dir / "cnn_fashion_mnist.keras"
    final_weights_path = model_dir / "cnn_fashion_mnist.weights.h5"
    winner_model.save(final_model_path)
    winner_model.save_weights(final_weights_path)

    evaluation = save_evaluation(winner_model, x_test, y_test, results_dir)
    summary = {
        "selection_metric": "minimum validation loss",
        "winner": winner_name,
        "data_split": {
            "training_samples": len(y_train),
            "validation_samples": len(y_val),
            "test_samples": len(y_test),
            "source_batch_size": 100,
        },
        "experiments": rows,
        "test_evaluation": evaluation,
    }
    (results_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (results_dir / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_comparison(rows, results_dir / "experiment_comparison.png")
    shutil.copy2(
        experiments_dir / winner_name / "training_curves.png",
        results_dir / "training_curves.png",
    )
    print(f"Selected model: {winner_name}")
    print(f"Saved model: {final_model_path}")
    print(f"Saved weights: {final_weights_path}")


if __name__ == "__main__":
    main()
