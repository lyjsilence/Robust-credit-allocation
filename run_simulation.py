"""Edit the settings below and run this file."""

import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import torch

from simulation.experiment import fit, run


MODE = "fit"  # "fit": one model; "experiment": all widths and evaluation
STUDY = "baseline"  # "baseline" or "shift"
CASES = ["low"]  # "low", "moderate", "high"; shift uses ["moderate"]
REPLICATIONS = [0]  # Use list(range(100)) for all replications
WIDTH = 10  # Used only in fit mode
METHOD = "robust"  # Used only in fit mode: "robust" or "non-robust"
OUTPUT_FOLDER = "results/example"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    if MODE not in ("fit", "experiment") or STUDY not in ("baseline", "shift"):
        raise ValueError("Choose MODE and STUDY from the options above")
    if STUDY == "shift" and CASES != ["moderate"]:
        raise ValueError('The shift experiment requires CASES = ["moderate"]')

    torch.set_num_threads(1)
    output = Path(__file__).resolve().parent / OUTPUT_FOLDER
    if MODE == "fit":
        if len(CASES) != 1 or len(REPLICATIONS) != 1 or WIDTH not in range(2, 257, 2):
            raise ValueError(
                "Fit mode needs one case, one replication and an even width 2–256"
            )
        if METHOD not in ("robust", "non-robust"):
            raise ValueError('METHOD must be "robust" or "non-robust"')
        path = fit(STUDY, CASES[0], REPLICATIONS[0], WIDTH, METHOD, output, DEVICE)
        print(f"Model saved to {path}")
    else:
        run(STUDY, CASES, REPLICATIONS, output, DEVICE)
        print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
