import unittest
from pathlib import Path

import numpy as np

from generate_dgp import (
    P0,
    SHIFT_SCALE,
    Y_BAD,
    Y_GOOD,
    Y_STABLE,
    apply_covariate_shift,
    build_parser,
    build_susceptibility,
    calibrated_tilt_weights,
    drifted_outcomes,
    environment_probabilities,
    generate_replication,
    middle_margin_score,
    nonlinear_score,
    output_directory,
    output_stem,
    simulate_catastrophe_dataset,
    simulate_dataset,
    solve_catastrophe_probability,
)


class CatastropheMixtureConceptDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = simulate_catastrophe_dataset(
            10000, 1042, retain_drift_state=True
        )
        cls.susceptibility = build_susceptibility(
            cls.data["X"], n_environments=8
        )

    def test_population_objective_ordering(self):
        risky_mean = P0 * Y_BAD + (1.0 - P0) * Y_GOOD
        self.assertGreater(risky_mean - Y_STABLE, 0.015)
        worst_probability = solve_catastrophe_probability(
            np.ones(1), 0.20
        )[0]
        worst_value = (
            worst_probability * Y_BAD
            + (1.0 - worst_probability) * Y_GOOD
        )
        self.assertGreater(Y_STABLE - worst_value, 0.10)

    def test_no_drift_identity(self):
        probabilities, eta_test, realized_kl = environment_probabilities(
            self.susceptibility[:1], 0.0
        )
        outcomes = drifted_outcomes(
            self.data["catastrophe_uniform"],
            self.data["risky_action"],
            probabilities,
        )
        np.testing.assert_allclose(outcomes[0], self.data["Y_all"], atol=1e-7)
        self.assertEqual(eta_test, 0.0)
        np.testing.assert_allclose(realized_kl, 0.0, atol=1e-12)

    def test_exact_kl_and_nested_catastrophes(self):
        previous = None
        grid = [0.5, 1, 1.5, 2, 2.5, 3]
        expected = [0.005, 0.02, 0.045, 0.08, 0.125, 0.18]
        for lambda_2, target in zip(grid, expected):
            probabilities, eta_test, realized_kl = environment_probabilities(
                self.susceptibility, lambda_2
            )
            self.assertAlmostEqual(eta_test, target)
            np.testing.assert_allclose(realized_kl, target, atol=1e-12)
            if previous is not None:
                self.assertTrue(np.all(probabilities > previous))
            previous = probabilities

    def test_drift_mode_uses_the_saved_data_contract(self):
        data = generate_replication(
            0,
            n_train=400,
            n_test=200,
            g_type="nonlinear",
            mode="drift",
            severity=1.0,
            num_environments=3,
            base_seed_train=42,
            base_seed_test=1042,
            dgp_kwargs={},
        )
        self.assertEqual(
            set(data), {"train", "test"}
        )
        self.assertEqual(
            set(data["test"]),
            {"base", "envs", "lambda2", "eta_test", "kl_list"},
        )
        self.assertEqual(len(data["test"]["envs"]), 3)
        self.assertEqual(data["test"]["envs"][0]["Y_all"].shape, (200, 2))


class CovariateShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = simulate_dataset(2500, 1042, "nonlinear")
        x = cls.base["X"]
        cls.raw_score = nonlinear_score(x[:, 0], x[:, 1])
        cls.shift_score = middle_margin_score(cls.raw_score, gamma=6.0)

    def test_middle_margin_score_has_fixed_peak(self):
        margin = np.array([0.0, 1.0 / 6.0, 1.0 / 3.0])
        score = middle_margin_score(margin, gamma=6.0)
        self.assertEqual(score[0], 0.0)
        self.assertGreater(score[1], score[2])
        np.testing.assert_allclose(
            self.shift_score,
            np.abs(self.raw_score) * np.exp(-6.0 * np.abs(self.raw_score)),
            atol=0.0,
        )

    def test_calibrated_weights_match_requested_kl(self):
        grid = [0.5, 1, 1.5, 2, 2.5, 3]
        expected = [0.005, 0.02, 0.045, 0.08, 0.125, 0.18]
        shifted_means = []
        for severity, target in zip(grid, expected):
            weights, realized_kl = calibrated_tilt_weights(
                self.shift_score, severity
            )
            self.assertAlmostEqual(realized_kl, target, places=10)
            shifted_means.append(weights @ self.shift_score)
        self.assertTrue(np.all(np.diff(shifted_means) > 0.0))

    def test_covariate_resampling_keeps_rows_aligned(self):
        weights, _ = calibrated_tilt_weights(self.shift_score, severity=3.0)
        shifted = apply_covariate_shift(self.base, weights, seed=123)
        lookup = {tuple(row): i for i, row in enumerate(self.base["X"])}
        for row, y_all, mu in zip(
            shifted["X"], shifted["Y_all"], shifted["mu"]
        ):
            index = lookup[tuple(row)]
            np.testing.assert_array_equal(y_all, self.base["Y_all"][index])
            np.testing.assert_array_equal(mu, self.base["mu"][index])


class OutputPathTests(unittest.TestCase):
    def test_mode_command(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--mode", "raw"]).mode, "raw")
        self.assertEqual(parser.parse_args([]).mode, "drift")

    def test_output_directories_match_run_experiment(self):
        base = Path("data")
        self.assertEqual(
            output_directory(base, "nonlinear", "raw", 0),
            base / "nonlinear",
        )
        self.assertEqual(
            output_directory(base, "nonlinear", "shift", 0.5),
            base / "nonlinear_shift=0.5",
        )
        self.assertEqual(
            output_directory(base, "nonlinear", "drift", 1),
            base / "nonlinear_drift=1.0",
        )
        self.assertEqual(output_stem("raw", 0), "Shift_0.0")
        self.assertEqual(output_stem("shift", 0.5), "Shift_0.5")
        self.assertEqual(output_stem("drift", 1), "Drift_1")


if __name__ == "__main__":
    unittest.main()
