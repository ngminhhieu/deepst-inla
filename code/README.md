# Commune dengue forecasting code

This repository implements probabilistic H1/H2/H3 commune-level dengue forecasts.
The neural path is intentionally explicit:

`inputs -> neural network -> distribution parameters -> predictive distribution -> samples`

The implementation supports a Gaussian latent distribution on `log1p(cases)` (the legacy configuration name is `lognormal`) and a negative-binomial output. For the Gaussian path, predictive draws are transformed back to the case-count scale with `expm1`. Predictive samples drive intervals and sample-based outbreak probabilities.

## Installation

The main pipeline supports Python 3.8 or newer. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e code
python -m pip install -r code/requirements.txt
```

The structured models additionally require R and R-INLA. Install the latter in
an isolated user library with `Rscript code/r/install_inla.R`. Set
`DENGUE_R_LIBRARY` when running the shell pipelines if that library is not the
R default. The TimesFM analysis is optional and requires Python 3.10 or newer:

```bash
python -m pip install -r code/requirements-timesfm3.txt
```

The TimesFM source package is Apache-2.0, but the `google/timesfm-3.0-pytorch`
checkpoint is governed by the TimesFM Non-Commercial License v1.0. Downloading
or using the weights signifies acceptance of that separate licence.

## Main validation runs

The commands in this section are run from the code directory:

```bash
cd code
```

### Deep learning vs INLA vs deep + INLA

```bash
bash scripts/run_inla_deep_comparison.sh config/qc_lognormal.yaml ../results/inla_deep_comparison
```

This creates three directly comparable model families:

- `DEEP-ONLY`: probabilistic GNN alone.
- `INLA-ONLY/INLA-TS`: structured Bayesian baseline.
- `DEEP-INLA/combined`: frozen deep representation plus structured INLA.

R-INLA is optional. If unavailable, the script still runs and scores the deep model.

### Feature / early-warning research suite

```bash
bash scripts/run_research_suite.sh config/qc_lognormal.yaml ../results/research_suite
```

The suite keeps candidate ideas separate instead of mixing all features at once:
future-climate forecasts, mobility/importation pressure, density, explicit epidemic
growth, an early-warning auxiliary objective, and an all-available combination.
See `docs/FEATURE_ROADMAP.md` for external feature groups and leakage rules.

For real future climate forecasts, provide a file keyed by
`fcode, issue_date, target_date`, then set:

```bash
export DENGUE_FUTURE_CLIMATE_PATH=/path/to/future_climate.csv
export DENGUE_FUTURE_CLIMATE_COLUMNS="t2m_forecast rh_forecast tp_forecast"
```

Generate the expected schema with `scripts/make_future_covariate_template.py`.

For observed mobility OD data, convert flows to importation pressure with
`scripts/build_mobility_pressure.py`.

For other vetted aggregate covariates, merge them with
`scripts/merge_external_features.py`, then run one-at-a-time groups with
`scripts/run_external_feature_ablation.py`.

## Outbreak metrics

All scoring code now reports the four distinct classification metrics:

- precision = TP / (TP + FP)
- specificity = TN / (TN + FP)
- recall = sensitivity = TP / (TP + FN)
- F1 = harmonic mean of precision and recall

The old misleading use of the name `specificity` for precision has been removed.
`outbreak_true_negative_rate` remains only as a compatibility alias for true
specificity. `scripts/evaluate_warning_lead_time.py` additionally reports how many
months before an observed outbreak target-month the first H1/H2/H3 alert appeared.

## Tests

```bash
pytest -q
```

For a complete smoke test without restricted surveillance data:

```bash
python scripts/generate_synthetic_data.py --output-dir data/synthetic
pytest -q
```
