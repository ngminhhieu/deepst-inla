# Data contract

Place private input files here; they are ignored by git.

The 2026 operational input layout is documented in `raw/README.md`; compact,
versioned threshold/rule inputs live under `reference/operational_2026/`.

## `raw/monthly_commune.csv`

One row per commune and calendar month:

```text
fcode,l1_code,year,month,obs_dengue_cases,population,t2m_avg,rh_avg,tp_accum
```

Required properties:

- `fcode` is a stable commune identifier.
- `l1_code` is a stable parent-area identifier used only by the
  province-allocation baseline.
- Cases are non-negative monthly counts.
- Population is positive.
- Climate values at an issue date must be observations available on or before
  that date. Observed future climate must never be used.
- The panel must contain all communes at all monthly timestamps. The validation
  script reports gaps and duplicates rather than silently filling them.

## `static/commune_centroids.csv`

```text
fcode,longitude,latitude
```

## `static/administrative_adjacency.csv`

Directed or undirected pairs are accepted:

```text
from_fcode,to_fcode
```

The code symmetrises these pairs and combines them with geographical nearest
neighbours to define candidate edges. Learned models can change edge weights
but cannot create implausible edges outside this candidate set.

The aliases `from,to` and `source,target` are also accepted.

## Synthetic smoke-test data

Generate a small schema-valid panel and graph with:

```bash
python scripts/generate_synthetic_data.py --output-dir data/synthetic
```

The synthetic series test software execution only and must never be used for
scientific results.
