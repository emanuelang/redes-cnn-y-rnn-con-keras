from __future__ import annotations

import unittest

import numpy as np

from rnn_data import make_window_dataset, prepare_datasets
from rnn_pipeline import build_model


class RnnDataTests(unittest.TestCase):
    def test_window_target_uses_requested_future_horizon(self) -> None:
        values = np.arange(30, dtype=np.float32).reshape(-1, 1)
        dataset, targets = make_window_dataset(
            values,
            target_index=0,
            lookback=4,
            horizon=3,
            batch_size=8,
            shuffle=False,
            seed=42,
        )
        features, target = next(iter(dataset))
        np.testing.assert_array_equal(features[0, :, 0].numpy(), [0, 1, 2, 3])
        self.assertEqual(float(target[0]), 6.0)
        self.assertEqual(float(targets[0]), 6.0)

    def test_normalization_is_fit_only_on_training_rows(self) -> None:
        first = np.arange(1000, dtype=np.float32)
        values = np.column_stack([first, first * 2])
        bundle = prepare_datasets(
            values,
            ["T (degC)", "feature"],
            "test",
            lookback=10,
            horizon=2,
            batch_size=16,
        )
        np.testing.assert_allclose(bundle.mean, values[:700].mean(axis=0))
        np.testing.assert_allclose(bundle.std, values[:700].std(axis=0), rtol=1e-5)
        self.assertEqual(bundle.split_rows["validation"], 150)
        self.assertEqual(bundle.split_rows["test"], 150)


class RnnModelTests(unittest.TestCase):
    def test_gru_outputs_one_forecast_per_window(self) -> None:
        model = build_model((12, 5), cell="gru", units=8, dropout=0.0)
        output = model(np.zeros((3, 12, 5), dtype=np.float32))
        self.assertEqual(tuple(output.shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
