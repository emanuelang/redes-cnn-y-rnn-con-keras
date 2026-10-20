from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["MPLCONFIGDIR"] = str(SCRIPT_DIR / ".cache" / "matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from rnn_data import (
    PROJECT_DIR,
    DatasetBundle,
    load_jena_csv,
    download_jena,
    generate_synthetic_series,
    prepare_datasets,
    save_data_summary,
    set_reproducibility,
)


def build_model(
    input_shape: tuple[int, int],
    cell: str = "gru",
    units: int = 32,
    dropout: float = 0.2,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="history")
    recurrent_layer = layers.GRU if cell == "gru" else layers.LSTM
    x = recurrent_layer(units, dropout=dropout, name=cell)(inputs)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, name="forecast")(x)
    return keras.Model(inputs, outputs, name=f"{cell}_forecaster_{units}")


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), keras.metrics.RootMeanSquaredError(name="rmse")],
    )


def create_callbacks(checkpoint: Path, patience: int = 4) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True
        ),
    ]


def inverse_target(values: np.ndarray, bundle: DatasetBundle) -> np.ndarray:
    return values * bundle.std[bundle.target_index] + bundle.mean[bundle.target_index]


def persistent_baseline(bundle: DatasetBundle) -> dict[str, float]:
    actual_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    for features, targets in bundle.test:
        actual_batches.append(targets.numpy().reshape(-1))
        prediction_batches.append(features[:, -1, bundle.target_index].numpy())
    actual = inverse_target(np.concatenate(actual_batches), bundle)
    predicted = inverse_target(np.concatenate(prediction_batches), bundle)
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
    }


def evaluate_model(model: keras.Model, bundle: DatasetBundle) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    normalized_predictions = model.predict(bundle.test, verbose=0).reshape(-1)
    actual = inverse_target(bundle.test_targets, bundle)
    predicted = inverse_target(normalized_predictions, bundle)
    metrics = {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
    }
    return metrics, actual, predicted


def train_model(
    bundle: DatasetBundle,
    cell: str,
    units: int,
    dropout: float,
    epochs: int,
    checkpoint: Path,
) -> tuple[keras.Model, dict[str, list[float]], float]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(
        input_shape=(bundle.train.element_spec[0].shape[1], len(bundle.feature_names)),
        cell=cell,
        units=units,
        dropout=dropout,
    )
    compile_model(model)
    started = time.perf_counter()
    history = model.fit(
        bundle.train,
        validation_data=bundle.validation,
        epochs=epochs,
        callbacks=create_callbacks(checkpoint),
        shuffle=False,
        verbose=2,
    )
    elapsed = time.perf_counter() - started
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    return model, history_data, elapsed


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--sample-every", type=int, default=6)
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)


def load_bundle(args: argparse.Namespace) -> DatasetBundle:
    if args.synthetic:
        values, feature_names, source = generate_synthetic_series(
            args.limit_rows or 3000, args.seed
        )
    else:
        csv_path = args.csv or (download_jena() if args.download else None)
        if csv_path is None:
            raise ValueError("Provide --csv, --download, or --synthetic.")
        values, feature_names, source = load_jena_csv(
            csv_path, sample_every=args.sample_every, limit_rows=args.limit_rows
        )
    return prepare_datasets(
        values,
        feature_names,
        source,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a recurrent forecaster.")
    add_data_arguments(parser)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    bundle = load_bundle(args)
    results_dir = PROJECT_DIR / "results" / "rnn"
    model_dir = PROJECT_DIR / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_data_summary(bundle, results_dir / "data_summary.json", args.lookback, args.horizon)

    checkpoint = model_dir / f"rnn_{args.cell}_{args.units}.keras"
    model, history, elapsed = train_model(
        bundle, args.cell, args.units, args.dropout, args.epochs, checkpoint
    )
    metrics, actual, predicted = evaluate_model(model, bundle)
    baseline = persistent_baseline(bundle)
    final_path = model_dir / "rnn_forecaster.keras"
    model.save(final_path)
    np.savez(results_dir / "predictions.npz", actual=actual, predicted=predicted)
    (results_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "data_source": bundle.source,
        "run_purpose": "development check" if bundle.source == "synthetic" else "dataset experiment",
        "model": {"cell": args.cell, "units": args.units, "dropout": args.dropout, "parameters": model.count_params()},
        "training_seconds": elapsed,
        "epochs_run": len(history["loss"]),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "test": metrics,
        "persistence_baseline": baseline,
    }
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved model to {final_path}")


if __name__ == "__main__":
    main()
