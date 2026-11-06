# Robust Credit Allocation

## Installation

Tested with Python 3.8.20 and PyTorch 2.3.0+cu118.

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Run

Open `run_simulation.py` in your editor and click **Run**, or use:

```bash
python run_simulation.py
```

The default settings train one Low-case model. A GPU is used when available;
otherwise the script uses the CPU. Settings are at the top of the file.

For all three baseline cases, set:

```python
MODE = "experiment"
STUDY = "baseline"
CASES = ["low", "moderate", "high"]
REPLICATIONS = list(range(100))
OUTPUT_FOLDER = "results/baseline"
```

For the covariate-shift and concept-drift experiment, set:

```python
MODE = "experiment"
STUDY = "shift"
CASES = ["moderate"]
REPLICATIONS = list(range(100))
OUTPUT_FOLDER = "results/shift"
```

Use `REPLICATIONS = [0]` for one replication. Experiment mode fits all widths;
it takes substantially longer than the default single-model run. The shift
experiment runs both robust and non-robust methods automatically.

## Output

Results are written under `OUTPUT_FOLDER`, relative to this repository:

- `model.pt` and `training.csv`: fitted models and training losses.
- `selections.csv`: selected widths and coefficients.
- `replication_results.csv` and `summary.csv`: evaluation results.
- `environments.csv`: individual shift environments, for the shift experiment.

Choose a new output folder for each run. Keep `simulation/random_inputs.npz`
with the code: it contains the fixed random seeds and bootstrap draws used by
the experiments. No data-generation or result-aggregation command is needed.
