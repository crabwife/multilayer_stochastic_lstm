"""Mathematical and simulation checks independent of PyTorch."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retention_data import (beta_shapes, generate, retention_diagnostics,
                            simulate, survival, training_windows)


class RetentionDataTests(unittest.TestCase):
    def test_same_one_step_law_but_distinct_long_survival(self):
        mu, b = .95, .6
        self.assertAlmostEqual(float(survival(1, mu, b, "annealed")), mu)
        self.assertAlmostEqual(float(survival(1, mu, b, "episodic")), mu)
        self.assertAlmostEqual(float(survival(128, mu, .6, "annealed")),
                               float(survival(128, mu, 1.4, "annealed")))
        self.assertGreater(float(survival(128, mu, b, "episodic")),
                           5 * float(survival(128, mu, b, "annealed")))
        diagnostic = retention_diagnostics(mu, b)
        self.assertLess(diagnostic["step_typical_log_retention"],
                        diagnostic["deterministic_log_retention"])
        self.assertTrue(np.isinf(diagnostic["episodic_mean_duration"]))
        self.assertTrue(np.isfinite(retention_diagnostics(mu, 1.4)["episodic_mean_duration"]))

    def test_power_law_exponent_and_mean_threshold(self):
        mu, b = .95, .6
        t = np.array([1_000_000, 2_000_000])
        values = survival(t, mu, b, "episodic")
        self.assertAlmostEqual(float(values[1] / values[0]), 2**(-b), delta=.0001)
        self.assertAlmostEqual(beta_shapes(mu, b)[0] / sum(beta_shapes(mu, b)), mu)

    def test_simulation_tracks_analytic_survival(self):
        mu, b, n, t = .95, .6, 5000, 96
        for policy in ("annealed", "episodic"):
            states = simulate(n, t + 1, mu, b, policy, seed=63)
            empirical = np.mean(np.all(states == states[:, [0]], axis=1))
            exact = float(survival(t, mu, b, policy))
            self.assertAlmostEqual(empirical, exact, delta=.025)

    def test_long_time_averages_separate_infinite_mean_from_control(self):
        ordinary = simulate(512, 2048, .95, .6, "annealed", 63).mean(axis=1)
        slow = simulate(512, 2048, .95, .6, "episodic", 63).mean(axis=1)
        self.assertLess(float(ordinary.var()), .01)
        self.assertGreater(float(slow.var()), .05)

    def test_trajectory_splits_and_path_windows(self):
        settings = {"mean_stay": .95, "beta_second_shape": [.6],
                    "processes": ["annealed", "episodic"],
                    "context": 32, "train_horizon": 16, "test_horizon": 128,
                    "train_trajectories": 5, "val_trajectories": 3,
                    "test_trajectories": 4}
        first, second = generate(settings, 63), generate(settings, 63)
        for key in first:
            for label in ("train", "val", "test"):
                np.testing.assert_array_equal(first[key][label], second[key][label])
            x, y = training_windows(first[key]["train"], 32, 16)
            self.assertEqual(tuple(x.shape), (15, 32, 1))
            self.assertEqual(tuple(y.shape), (15, 16))
            np.testing.assert_array_equal(y[:5], first[key]["train"][:, 32:48])


if __name__ == "__main__":
    unittest.main()
