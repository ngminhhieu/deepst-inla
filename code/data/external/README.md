# External feature inputs

Do not place undocumented raw datasets directly into the model panel. Keep the original source separately and merge only audited aggregate columns.

## Static/monthly features

Use:

```bash
python scripts/merge_external_features.py \
  --monthly data/processed/qc_v1/monthly_commune.csv \
  --features <feature.csv> \
  --level commune-monthly \
  --columns <col1> <col2> \
  --output data/processed/feature_suite/monthly_with_external.csv
```

Supported levels: `commune-static`, `commune-monthly`, `province-static`, `province-monthly`.

Then run available groups separately:

```bash
python scripts/run_external_feature_ablation.py \
  --config config/qc_lognormal.yaml \
  --monthly data/processed/feature_suite/monthly_with_external.csv \
  --results-root ../results/external_ablation
```

## Future climate forecasts

Create the required schema:

```bash
python scripts/make_future_covariate_template.py \
  --monthly data/processed/qc_v1/monthly_commune.csv \
  --output data/external/future_climate.csv
```

Fill the forecast values with data that were actually available on each `issue_date`.
Never fill this file from realised future weather and then call it a deployable forecast.
