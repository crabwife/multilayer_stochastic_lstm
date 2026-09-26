"""Check alignment and the trajectory bootstrap unit without a model run."""

import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paired import bootstrap_contrasts, paired_rows


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.on = [{
            "process": "polya", "config": "F", "replicate": rep, "trajectory": i,
            "observed_frequency": i / 10, "oracle_frequency": .5,
            "oracle_crps": .1, "oracle_variance": .01, "oracle_one_nll": .6,
            "crps": .12 + i / 100, "one_nll": .7,
            "frequency_variance": .02, "cover90": 1,
            "predicted_frequency": .5
        } for rep in (0, 1) for i in (0, 1, 2)]
        self.off = [{**row, "crps": row["crps"] + .02}
                    for row in reversed(self.on)]

    def test_shuffle_does_not_break_pairing_or_average_replicates_as_new_data(self):
        paired = paired_rows(self.on, self.off)
        np.testing.assert_allclose(paired.delta_crps, -.02)
        result = bootstrap_contrasts(paired, 40, 63)
        crps = result[result.metric == "crps"].iloc[0]
        self.assertEqual(crps.independent_test_trajectories, 3)
        self.assertEqual(crps.training_replicates, 2)
        self.assertAlmostEqual(crps.delta_on_minus_off, -.02)

    def test_mismatched_oracle_fails(self):
        wrong = [dict(row) for row in self.off]
        wrong[0]["oracle_crps"] = .2
        with self.assertRaises(ValueError):
            paired_rows(self.on, wrong)


if __name__ == "__main__":
    unittest.main()
