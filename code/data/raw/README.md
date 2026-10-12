# Local data (not committed)

Files below `code/data/raw/` are deliberately excluded from git because they
contain restricted surveillance data, external snapshots, or bulky inputs that
can be restored separately.  Directory names and expected filenames are part
of the data contract.

For the 2026 operational pipeline, place these files in
`code/data/raw/operational_2026/`:

| File | Purpose |
|---|---|
| `mongo_commune_2026.json` | Current commune MongoDB snapshot |
| `rhc_original_mongo_forecasts_2026.json` | Original RHC forecast snapshot |
| `rhc_monthly_2010_to_2026-09.csv` | Historical RHC monthly panel |
| `commune_monthly_thresholds.csv` | Full historical commune threshold input |
| `original_commune_forecast_quantiles.csv` | Original external commune forecast input |

The small, non-restricted operational rule workbook and exported MOH threshold
table are versioned under `code/data/reference/operational_2026/`.

After restoring local data, verify the expected files with:

```bash
bash code/scripts/check_operational_2026_layout.sh
```
