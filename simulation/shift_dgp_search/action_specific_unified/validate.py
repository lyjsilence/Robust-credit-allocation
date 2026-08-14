"""Check completeness, KL calibration, and deterministic regeneration."""

from pathlib import Path

import joblib
import numpy as np

from dgp import SEVERITIES, SHIFT_SCALE, generate_replication


HERE = Path(__file__).resolve().parent


def main():
    result_paths = sorted(HERE.glob("results/rep-*.pkl"))
    result_ids = sorted(int(path.stem.split("-")[1]) for path in result_paths)
    assert result_ids == list(range(100))
    for path in result_paths:
        result = joblib.load(path)
        assert [len(result[key]) for key in ("table1", "table2", "table3")] == [20, 24, 35]

    max_covariate_error = 0.0
    max_concept_error = 0.0
    saved_replication_zero = None
    data_paths = sorted(HERE.glob("data/rep-*.pkl"))
    assert len(data_paths) == 100
    for path in data_paths:
        data = joblib.load(path)
        if path.stem == "rep-0":
            saved_replication_zero = data
        for severity in SEVERITIES:
            target = 0.5 * (SHIFT_SCALE * severity) ** 2
            covariate_error = abs(data["covariate"][f"{severity:g}"]["kl"] - target)
            concept_error = np.max(
                np.abs(data["concept"][f"{severity:g}"]["kl"] - target)
            )
            max_covariate_error = max(max_covariate_error, covariate_error)
            max_concept_error = max(max_concept_error, concept_error)

    regenerated = generate_replication(0)
    arrays = (
        ("linear", "train", "X"),
        ("linear", "train", "Y_all"),
        ("nonlinear", "test", "X"),
        ("tail", "test", "Y_all"),
    )
    assert all(
        np.array_equal(saved_replication_zero[a][b][c], regenerated[a][b][c])
        for a, b, c in arrays
    )
    print("100 complete replications; table rows per replication: 20/24/35")
    print("Replication 0 regeneration: exact")
    print(
        f"Maximum KL error: covariate={max_covariate_error:.3e}, "
        f"concept={max_concept_error:.3e}"
    )


if __name__ == "__main__":
    main()
