from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".cache" / "matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figures for the RNN report.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow development-only figures from the synthetic dataset.",
    )
    return parser.parse_args()


def plot_history(history: dict[str, list[float]], output: Path) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["loss"], marker="o", label="Entrenamiento")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validacion")
    axes[0].set(title="Perdida durante el entrenamiento", xlabel="Epoca", ylabel="MSE normalizado")
    axes[0].legend()
    axes[1].plot(epochs, history["mae"], marker="o", label="Entrenamiento")
    axes[1].plot(epochs, history["val_mae"], marker="o", label="Validacion")
    axes[1].set(title="Error absoluto medio", xlabel="Epoca", ylabel="MAE normalizado")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    count = min(300, len(actual))
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(actual[:count], label="Real", linewidth=1.6)
    axis.plot(predicted[:count], label="Prediccion", linewidth=1.2)
    axis.set(title="Pronostico sobre el conjunto de prueba", xlabel="Paso temporal", ylabel="Temperatura (degC)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_experiments(rows: list[dict[str, str]], output: Path) -> None:
    names = [row["variant"] for row in rows]
    validation_mae = [float(row["best_validation_mae_normalized"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, validation_mae, color=["#277da1", "#43aa8b", "#f8961e"])
    axis.set(title="Comparacion de modelos recurrentes", ylabel="Mejor MAE de validacion normalizado")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_DIR / "results" / "rnn"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_summary = json.loads(
        (results_dir / "data_summary.json").read_text(encoding="utf-8")
    )
    if data_summary["source"] == "synthetic" and not args.allow_synthetic:
        raise RuntimeError(
            "Synthetic results are for development only. Use --allow-synthetic "
            "to inspect them, or run the experiments with Jena Climate."
        )
    history = json.loads((results_dir / "history.json").read_text(encoding="utf-8"))
    predictions = np.load(results_dir / "predictions.npz")
    with (results_dir / "experiment_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plot_history(history, figures_dir / "training_curves.png")
    plot_predictions(predictions["actual"], predictions["predicted"], figures_dir / "test_predictions.png")
    plot_experiments(rows, figures_dir / "experiment_comparison.png")
    print(f"Report figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
