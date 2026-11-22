# Red-tide forecast models

Forecasts the *Karenia brevis* class on the Sarasota coast (no bloom, at risk, bloom) 1, 2 and 4 weeks ahead.

## Setup

From the repo root:

```bash
git lfs install && git lfs pull
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd models
```

## Retrain

1. Edit the settings in [`model_config.toml`](model_config.toml).
2. Run:

   ```bash
   python run_pipeline.py
   ```

   This reruns cross-validation and retrains the saved models at every horizon.
3. Check the scores in `results/h<H>w/<model>/metrics.csv`.
4. Commit the settings, metrics and saved models together.

| Option | Effect |
|---|---|
| `--models hybrid_lr` | one model |
| `--horizons 1` | one horizon |
| `--skip-cv` | only retrain the saved models |
| `--rebuild-table` | rebuild the weekly table after new data |

## Use a saved model

```python
import joblib
from common.weekly import build_weekly

table, _ = build_weekly()
bundle = joblib.load("results/h1w/hybrid_lr/final__kb_only.joblib")
forecast = bundle["model"].predict(table[bundle["features"]])
```

To use a teammate's models, replace the `final__*.joblib` files in `results/h<H>w/<model>/`.

## Folder layout

| Path | Contents |
|---|---|
| `model_config.toml` | model settings |
| `run_pipeline.py` | retrain command |
| `common/` | weekly table, validation folds, scores, cross-validation |
| `hybrid_escalation/` | Hybrid LR and Hybrid XGB |
| `save_final_models.py` | trains and saves the final models |
| `results/` | scores and saved models per horizon |
| `tests/` | unit tests |

## Add a model

1. Create `<model_name>/train.py` with a `make_model()` function that returns a scikit-learn style classifier.
2. Call `run_model("<model_name>", make_model)` to score it on the same folds as every other model.
3. Add its settings to `model_config.toml` and its tests to `tests/`.

## Run the tests

```bash
python -m pytest --doctest-modules common tests hybrid_escalation save_final_models.py run_pipeline.py
```
