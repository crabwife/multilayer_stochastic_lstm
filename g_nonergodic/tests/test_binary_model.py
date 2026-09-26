"""Small integration checks for the actual shared gate layers."""

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "c_fit" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if torch is not None:
    from binary_model import BinaryGateLSTM


@unittest.skipIf(torch is None, "PyTorch is required for model integration tests")
class BinaryModelTests(unittest.TestCase):
    def test_all_modes_train_and_roll_out_binary_paths(self):
        x = torch.tensor([[[0.], [1.], [0.]], [[1.], [1.], [0.]]])
        for config in ("D", "F", "I", "A", "FI", "IF", "AAA"):
            torch.manual_seed(63)
            model = BinaryGateLSTM(config, 4)
            prob = model(x, particles=3)
            self.assertEqual(tuple(prob.shape), (2,))
            self.assertTrue(bool(((prob > 0) & (prob < 1)).all()))
            (-prob.log()).mean().backward()
            self.assertTrue(bool(torch.isfinite(model.layers[0].affine.weight.grad).all()))
            paths = model.sample_paths(x, particles=5, horizon=7)
            self.assertEqual(tuple(paths.shape), (2, 5, 7))
            self.assertTrue(bool(((paths == 0) | (paths == 1)).all()))


if __name__ == "__main__":
    unittest.main()
