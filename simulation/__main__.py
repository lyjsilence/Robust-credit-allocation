"""Command-line entry point for the simulation experiments."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

from .experiment import main

if __name__ == "__main__":
    main()
