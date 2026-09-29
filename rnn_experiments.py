from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from tensorflow import keras

from rnn_data import PROJECT_DIR, save_data_summary, set_reproducibility
from rnn_pipeline import (
    add_data_arguments,
    evaluate_model,
    load_bundle,
    persistent_baseline,
    train_model,
)


VARIANTS = (
    {"name": "gru_32", "cell": "gru", "units": 32, "dropout": 0.2},
    {"name": "gru_64", "cell": "gru", "units": 64, "dropout": 0.2},
    {"name": "lstm_32", "cell": "lstm", "units": 32, "dropout": 0.2},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recurrent model variants.")
    add_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    bundle = load_bundle(args)
    results_dir = PROJECT_DIR / "results" / "rnn"
    experiments_dir = results_dir / "experiments"
    model_dir = PROJECT_DIR / "models"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_data_summary(bundle, results_dir / "data_summary.json", args.lookback, args.horizon)
    rows: list[dict[str, float | int | str]] = []

    for offset, variant in enumerate(VARIANTS):
        keras.backend.clear_session()
        set_reproducibility(args.seed + offset)
        name = str(variant["name"])
        variant_dir = experiments_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = model_dir / f"rnn_{name}.keras"
        print(f"\nTraining variant: {name}")
        model, history, elapsed = train_model(
            bundle,
            str(variant["cell"]),
            int(variant["units"]),
            float(variant["dropout"]),
            args.epochs,
            checkpoint,
        )
        (variant_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        best_index = int(np.argmin(history["val_loss"]))
        rows.append(
            {
                "variant": name,
                "cell": variant["cell"],
                "units": variant["units"],
                "parameters": model.count_params(),
                "epochs_run": len(history["loss"]),
                "best_epoch": best_index + 1,
                "best_train_mae_normalized": history["mae"][best_index],
                "best_validation_mae_normalized": history["val_mae"][best_index],
                "best_validation_rmse_normalized": history["val_rmse"][best_index],
                "training_seconds": elapsed,
            }
        )

    winner = min(rows, key=lambda row: float(row["best_validation_mae_normalized"]))
    winner_name = str(winner["variant"])
    winner_model = keras.models.load_model(model_dir / f"rnn_{winner_name}.keras")
    final_model_path = model_dir / "rnn_forecaster.keras"
    winner_model.save(final_model_path)
    metrics, actual, predicted = evaluate_model(winner_model, bundle)
    baseline = persistent_baseline(bundle)
    np.savez(results_dir / "predictions.npz", actual=actual, predicted=predicted)
    shutil.copy2(
        experiments_dir / winner_name / "history.json", results_dir / "history.json"
    )

    with (results_dir / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "data_source": bundle.source,
        "run_purpose": "development check" if bundle.source == "synthetic" else "dataset experiment",
        "selection_metric": "minimum validation MAE",
        "winner": winner_name,
        "experiments": rows,
        "test": metrics,
        "persistence_baseline": baseline,
    }
    (results_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
