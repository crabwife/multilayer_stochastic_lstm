"""Integration check for the three policies when PyTorch is installed."""

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if torch is not None:
    from retention_model import RetentionLSTM


@unittest.skipIf(torch is None, "PyTorch is required for model integration")
class ModelTests(unittest.TestCase):
    def test_joint_likelihood_backward_and_binary_rollout(self):
        x = torch.tensor([[[0.], [1.], [1.]], [[1.], [1.], [0.]]])
        future = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.]])
        sizes = []
        for policy in ("mean", "step", "episode"):
            torch.manual_seed(63)
            model = RetentionLSTM(5, policy)
            sizes.append(sum(p.numel() for p in model.parameters()))
            loss = model.path_nll(x, future, particles=4)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(bool(torch.isfinite(model.decoder.weight.grad).all()))
            if policy != "mean":
                self.assertIsNotNone(model.retention.weight.grad)
                self.assertTrue(bool(torch.isfinite(model.retention.weight.grad).all()))
            paths = model.sample_paths(x, 8, 12, emission_seed=901)
            self.assertEqual(tuple(paths.shape), (2, 8, 12))
            self.assertTrue(bool(((paths == 0) | (paths == 1)).all()))
        self.assertEqual(len(set(sizes)), 1)


if __name__ == "__main__":
    unittest.main()
