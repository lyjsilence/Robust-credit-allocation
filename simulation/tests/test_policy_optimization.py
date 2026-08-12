from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import alternative_optimization_with_holdout as optimization


class PolicyOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            optimization_epochs=50,
            lr=0.1,
            weight_decay=0.0,
            verbose=False,
        )
        self.validation = {
            "X_val": [[0.0]],
            "T_val": [[0]],
            "Y_val": [[0.0]],
            "prop_score_val": [[1.0]],
        }

    def test_nonrobust_uses_ipw_and_restores_best_checkpoint(self):
        policy = torch.nn.Linear(1, 1, bias=False)
        self.args.optimization_epochs = 110
        values = [0.0, 1.0] + [0.5] * 108
        saved_weights = []

        def ipw_value(model, *_):
            saved_weights.append(model.weight.detach().clone())
            return torch.tensor(values[len(saved_weights) - 1])

        pol_dict = {
            "args": self.args,
            "robust_met": "non-robust",
            **self.validation,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pth"
            with patch.object(
                optimization, "eval_IPW", side_effect=ipw_value
            ), patch.object(
                optimization,
                "evaluation",
                side_effect=AssertionError("wrong criterion"),
            ):
                _, best = optimization.optimization(
                    lambda output: output.mean(),
                    torch.ones((1, 1)),
                    pol_dict,
                    policy,
                    checkpoint,
                    torch.device("cpu"),
                    0,
                    torch.nn.Parameter(torch.tensor(1.0)),
                )

        self.assertEqual(best, 1.0)
        self.assertEqual(len(saved_weights), 100)
        self.assertTrue(torch.equal(policy.weight, saved_weights[1]))

    def test_robust_uses_dro_validation(self):
        policy = torch.nn.Linear(1, 1, bias=False)
        self.args.optimization_epochs = 1
        pol_dict = {"args": self.args, "robust_met": "robust", **self.validation}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                optimization,
                "evaluation",
                return_value=(torch.tensor(0.7), None),
            ) as dro, patch.object(
                optimization,
                "eval_IPW",
                side_effect=AssertionError("wrong criterion"),
            ):
                _, best = optimization.optimization(
                    lambda output: output.mean(),
                    torch.ones((1, 1)),
                    pol_dict,
                    policy,
                    Path(directory) / "policy.pth",
                    torch.device("cpu"),
                    0,
                    torch.nn.Parameter(torch.tensor(1.0)),
                )

        self.assertAlmostEqual(best, 0.7, places=6)
        dro.assert_called_once()


if __name__ == "__main__":
    unittest.main()
