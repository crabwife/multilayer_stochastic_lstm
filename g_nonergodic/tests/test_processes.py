"""Independent checks on the simulated mechanisms and reference distributions."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import betabinom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from processes import (discrete_crps, ensemble_crps, generate, one_step_windows,
                       oracle_pmf, sticky_count_pmf)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "context": 32, "horizon": 128, "sticky_stay": .95,
            "train_trajectories": 400, "val_trajectories": 20,
            "test_trajectories": 20, "processes": ["polya", "iid", "sticky"]
        }

    def test_splits_reproducible_and_windows_do_not_cross_trajectories(self):
        first = generate(self.settings, 63)
        second = generate(self.settings, 63)
        for process in self.settings["processes"]:
            for split in ("train", "val", "test"):
                np.testing.assert_array_equal(first[process][split], second[process][split])
            x, y = one_step_windows(first[process]["train"], 32)
            np.testing.assert_array_equal(x[:400, :, 0], first[process]["train"][:, :32])
            np.testing.assert_array_equal(y[:400], first[process]["train"][:, 32])
            self.assertEqual(len(x), 3 * 400)

    def test_polya_has_broad_trajectory_means_but_iid_concentrates(self):
        first = generate(self.settings, 63)
        polya = first["polya"]["train"].mean(axis=1)
        iid = first["iid"]["train"].mean(axis=1)
        self.assertGreater(polya.var(), .05)
        self.assertLess(iid.var(), .005)

    def test_polya_posterior_is_beta_binomial(self):
        history = np.r_[np.ones(7), np.zeros(3)]
        pmf = oracle_pmf("polya", history, 20, .95)
        np.testing.assert_allclose(pmf, betabinom.pmf(np.arange(21), 20, 8, 4))
        self.assertAlmostEqual(pmf.sum(), 1.)
        self.assertAlmostEqual(np.dot(pmf, np.arange(21)) / 20, 8 / 12)
        self.assertGreater(pmf.var(), 0)

    def test_sticky_count_distribution_enumeration(self):
        p = .95
        got = sticky_count_pmf(2, 1, p)
        # Start in 1: 00, 01/10, 11 have these total probabilities.
        expected = np.array([(1-p)*p, (1-p)**2 + p*(1-p), p*p])
        np.testing.assert_allclose(got, expected)
        self.assertAlmostEqual(got.sum(), 1.)

    def test_discrete_crps_matches_full_empirical_ensemble(self):
        counts = np.array([0, 0, 1, 2, 2])
        target = 1
        direct = np.abs(counts / 2 - target / 2).mean()
        direct -= np.abs(counts[:, None] - counts[None, :]).mean() / 4
        self.assertAlmostEqual(ensemble_crps(counts, target, 2), direct)
        self.assertAlmostEqual(discrete_crps(np.array([.4, .2, .4]), target), direct)


if __name__ == "__main__":
    unittest.main()
