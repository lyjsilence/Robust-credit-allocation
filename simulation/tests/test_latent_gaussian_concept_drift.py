import unittest

import numpy as np

from generate_dgp import simulate_dataset
from latent_gaussian_concept_drift import (
    build_drift_directions,
    drifted_outcomes,
    empirical_latent_kl,
)


class LatentGaussianConceptDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = simulate_dataset(
            600, 1042, "nonlinear", retain_drift_state=True
        )
        u_base = np.tanh(6.0 * cls.data["score"].reshape(-1))
        cls.directions = build_drift_directions(
            cls.data["X"], u_base, n_environments=8
        )

    def test_no_drift_identity_and_covariates(self):
        original_x = self.data["X"].copy()
        outcomes, eta_test = drifted_outcomes(
            self.data["mu"],
            self.data["eps_all"],
            self.directions,
            lambda_2=0.0,
        )
        np.testing.assert_allclose(outcomes[0], self.data["Y_all"], atol=1e-7)
        np.testing.assert_array_equal(self.data["X"], original_x)
        self.assertEqual(eta_test, 0.0)

    def test_normalization_and_kl_calibration(self):
        squared_norm = np.mean(
            np.sum(self.directions**2, axis=2), axis=1
        )
        np.testing.assert_allclose(squared_norm, 1.0, atol=1e-12)

        kl = empirical_latent_kl(self.directions, lambda_2=3.0)
        np.testing.assert_allclose(kl, 0.18, atol=1e-12)

    def test_eta_test_increases_with_severity(self):
        grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        expected = [0.005, 0.020, 0.045, 0.080, 0.125, 0.180]
        eta = [
            drifted_outcomes(
                self.data["mu"],
                self.data["eps_all"],
                self.directions[:1],
                value,
            )[1]
            for value in grid
        ]
        np.testing.assert_allclose(eta, expected, atol=1e-15)
        self.assertTrue(np.all(np.diff(eta) > 0))


if __name__ == "__main__":
    unittest.main()
