from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(PROJECT_DIR / ".cache" / "keras"))
os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".cache" / "matplotlib")

import numpy as np
import tensorflow as tf
from tensorflow import keras


JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
DEFAULT_FEATURES = ("p (mbar)", "T (degC)", "rh (%)", "wv (m/s)", "wd (deg)")
DEFAULT_TARGET = "T (degC)"


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    test_targets: np.ndarray
    feature_names: list[str]
    target_name: str
    target_index: int
    mean: np.ndarray
    std: np.ndarray
    split_rows: dict[str, int]
    source: str


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def download_jena() -> Path:
    archive = Path(
        keras.utils.get_file(
            fname="jena_climate_2009_2016.csv.zip",
            origin=JENA_URL,
            cache_dir=PROJECT_DIR / ".cache",
            cache_subdir="datasets",
        )
    )
    output_dir = archive.parent / "jena_climate"
    csv_path = output_dir / "jena_climate_2009_2016.csv"
    if not csv_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as compressed:
            compressed.extractall(output_dir)
    return csv_path


def load_jena_csv(
    csv_path: Path,
    feature_names: tuple[str, ...] = DEFAULT_FEATURES,
    target_name: str = DEFAULT_TARGET,
    sample_every: int = 6,
    limit_rows: int | None = None,
) -> tuple[np.ndarray, list[str], str]:
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1.")

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        missing = [name for name in (*feature_names, target_name) if name not in header]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")
        feature_indices = [header.index(name) for name in feature_names]
        rows: list[list[float]] = []
        for source_index, row in enumerate(reader):
            if source_index % sample_every:
                continue
            try:
                values = [float(row[index]) for index in feature_indices]
            except (ValueError, IndexError):
                continue
            if all(math.isfinite(value) for value in values):
                rows.append(values)
            if limit_rows is not None and len(rows) >= limit_rows:
                break

    if len(rows) < 100:
        raise ValueError("The dataset must contain at least 100 valid sampled rows.")
    return np.asarray(rows, dtype=np.float32), list(feature_names), str(csv_path)


def generate_synthetic_series(rows: int, seed: int) -> tuple[np.ndarray, list[str], str]:
    if rows < 200:
        raise ValueError("Synthetic data requires at least 200 rows.")
    generator = np.random.default_rng(seed)
    time = np.arange(rows, dtype=np.float32)
    daily = np.sin(2 * np.pi * time / 24)
    weekly = np.sin(2 * np.pi * time / (24 * 7))
    pressure = 1013 + 4 * weekly + generator.normal(0, 0.7, rows)
    temperature = 15 + 8 * daily + 2 * weekly + generator.normal(0, 0.8, rows)
    humidity = 65 - 12 * daily + generator.normal(0, 2.0, rows)
    wind_speed = np.maximum(0, 3 + 1.2 * weekly + generator.normal(0, 0.5, rows))
    wind_direction = (180 + 90 * np.sin(2 * np.pi * time / 72)) % 360
    values = np.column_stack(
        [pressure, temperature, humidity, wind_speed, wind_direction]
    ).astype(np.float32)
    return values, list(DEFAULT_FEATURES), "synthetic"


def make_window_dataset(
    values: np.ndarray,
    target_index: int,
    lookback: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[tf.data.Dataset, np.ndarray]:
    offset = lookback + horizon - 1
    if len(values) <= offset:
        raise ValueError("A split is too short for the requested lookback and horizon.")
    targets = values[offset:, target_index].astype(np.float32)
    values_tensor = tf.convert_to_tensor(values, dtype=tf.float32)
    indices = tf.data.Dataset.from_tensor_slices(
        tf.range(len(values) - offset, dtype=tf.int32)
    )
    if shuffle:
        indices = indices.shuffle(
            buffer_size=min(len(values) - offset, 10000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def build_window(index: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        features = values_tensor[index : index + lookback]
        target = values_tensor[index + offset, target_index]
        return features, target

    dataset = indices.map(build_window, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    )
    return dataset.prefetch(tf.data.AUTOTUNE), targets


def prepare_datasets(
    values: np.ndarray,
    feature_names: list[str],
    source: str,
    target_name: str = DEFAULT_TARGET,
    lookback: int = 72,
    horizon: int = 6,
    batch_size: int = 128,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave a test split.")
    if target_name not in feature_names:
        raise ValueError(f"Target {target_name!r} is not present in feature columns.")

    train_end = int(len(values) * train_fraction)
    validation_end = int(len(values) * (train_fraction + validation_fraction))
    train_raw = values[:train_end]
    validation_raw = values[train_end:validation_end]
    test_raw = values[validation_end:]

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std[std == 0] = 1.0
    train_values = (train_raw - mean) / std
    validation_values = (validation_raw - mean) / std
    test_values = (test_raw - mean) / std
    target_index = feature_names.index(target_name)

    train, _ = make_window_dataset(
        train_values, target_index, lookback, horizon, batch_size, True, seed
    )
    validation, _ = make_window_dataset(
        validation_values, target_index, lookback, horizon, batch_size, False, seed
    )
    test, test_targets = make_window_dataset(
        test_values, target_index, lookback, horizon, batch_size, False, seed
    )
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        test_targets=test_targets,
        feature_names=feature_names,
        target_name=target_name,
        target_index=target_index,
        mean=mean,
        std=std,
        split_rows={
            "total": len(values),
            "training": len(train_raw),
            "validation": len(validation_raw),
            "test": len(test_raw),
        },
        source=source,
    )


def save_data_summary(
    bundle: DatasetBundle,
    output: Path,
    lookback: int,
    horizon: int,
) -> None:
    summary = {
        "source": bundle.source,
        "feature_names": bundle.feature_names,
        "target_name": bundle.target_name,
        "split_strategy": "chronological 70/15/15",
        "split_rows": bundle.split_rows,
        "lookback_steps": lookback,
        "forecast_horizon_steps": horizon,
        "normalization": {
            "fit_on": "training split only",
            "mean": bundle.mean.tolist(),
            "standard_deviation": bundle.std.tolist(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare time-series data for RNNs.")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--sample-every", type=int, default=6)
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> tuple[np.ndarray, list[str], str]:
    if args.synthetic:
        return generate_synthetic_series(args.limit_rows or 3000, args.seed)
    csv_path = args.csv or (download_jena() if args.download else None)
    if csv_path is None:
        raise ValueError("Provide --csv, --download, or --synthetic.")
    return load_jena_csv(csv_path, sample_every=args.sample_every, limit_rows=args.limit_rows)


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    values, feature_names, source = load_source(args)
    bundle = prepare_datasets(
        values,
        feature_names,
        source,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output = PROJECT_DIR / "results" / "rnn" / "data_summary.json"
    save_data_summary(bundle, output, args.lookback, args.horizon)
    print(f"Data ready from {source}: {bundle.split_rows}")
    print(f"Summary saved to {output}")


if __name__ == "__main__":
    main()
