import unittest

import numpy as np

from fragile_action_concept_drift import (
    build_fragile_directions,
    drifted_fragile_outcomes,
    empirical_latent_kl,
    simulate_fragile_dataset,
)


class FragileActionConceptDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = simulate_fragile_dataset(
            10000, 1042, retain_drift_state=True
        )
        cls.directions = build_fragile_directions(
            cls.data["X"], n_environments=8
        )

    def test_baseline_mean_and_lower_tail_tradeoff(self):
        fragile = self.data["fragile_action"].reshape(-1)
        stable = 1 - fragile
        rows = np.arange(fragile.size)
        y_fragile = self.data["Y_all"][rows, fragile]
        y_stable = self.data["Y_all"][rows, stable]

        self.assertGreater(y_fragile.mean() - y_stable.mean(), 0.02)
        self.assertGreater(
            np.quantile(y_stable, 0.10)
            - np.quantile(y_fragile, 0.10),
            0.05,
        )

    def test_no_drift_identity_and_stable_action_unchanged(self):
        outcomes, eta_test = drifted_fragile_outcomes(
            self.data["mu"],
            self.data["eps_all"],
            self.data["sigma_all"],
            self.data["fragile_action"],
            self.directions,
            lambda_2=0.0,
        )
        np.testing.assert_allclose(outcomes[0], self.data["Y_all"], atol=1e-7)
        self.assertEqual(eta_test, 0.0)

        drifted, _ = drifted_fragile_outcomes(
            self.data["mu"],
            self.data["eps_all"],
            self.data["sigma_all"],
            self.data["fragile_action"],
            self.directions[:1],
            lambda_2=3.0,
        )
        fragile = self.data["fragile_action"].reshape(-1)
        stable = 1 - fragile
        rows = np.arange(fragile.size)
        np.testing.assert_allclose(
            drifted[0, rows, stable],
            self.data["Y_all"][rows, stable],
            atol=1e-7,
        )
        self.assertTrue(
            np.all(
                drifted[0, rows, fragile]
                < self.data["Y_all"][rows, fragile]
            )
        )

    def test_normalization_kl_and_severity_mapping(self):
        np.testing.assert_allclose(
            np.mean(self.directions**2, axis=1), 1.0, atol=1e-12
        )
        np.testing.assert_allclose(
            empirical_latent_kl(self.directions, 3.0),
            0.18,
            atol=1e-12,
        )
        grid = np.array([0.5, 1, 1.5, 2, 2.5, 3])
        eta = 0.5 * (0.2 * grid) ** 2
        np.testing.assert_allclose(
            eta, [0.005, 0.02, 0.045, 0.08, 0.125, 0.18]
        )
        self.assertTrue(np.all(np.diff(eta) > 0))


if __name__ == "__main__":
    unittest.main()
