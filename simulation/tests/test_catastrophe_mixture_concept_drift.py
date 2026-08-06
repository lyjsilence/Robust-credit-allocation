import unittest

import numpy as np

from catastrophe_mixture_concept_drift import (
    P0,
    Y_BAD,
    Y_GOOD,
    Y_STABLE,
    build_susceptibility,
    covariate_environment_values,
    drifted_outcomes,
    environment_probabilities,
    population_worst_risky_value,
    simulate_catastrophe_dataset,
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
        self.assertGreater(
            Y_STABLE - population_worst_risky_value(0.20), 0.10
        )

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

    def test_covariate_shift_reuses_the_baseline_dgp(self):
        risky = self.data["risky_action"].reshape(-1)
        stable = 1 - risky
        actions = np.stack([risky, stable])
        values, divergences = covariate_environment_values(
            self.data["Y_all"],
            actions,
            severity=0.0,
            n_environments=3,
            replication=0,
        )
        expected = np.array(
            [
                self.data["Y_all"][np.arange(risky.size), risky].mean(),
                self.data["Y_all"][np.arange(risky.size), stable].mean(),
            ]
        )
        np.testing.assert_allclose(
            values, np.repeat(expected[:, None], 3, axis=1)
        )
        np.testing.assert_array_equal(divergences, 0.0)


if __name__ == "__main__":
    unittest.main()
