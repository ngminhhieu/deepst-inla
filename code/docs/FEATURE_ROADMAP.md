# Candidate data roadmap for commune dengue forecasting

The model code now supports one-at-a-time feature ablations instead of putting every possible covariate into one model. This is important because a variable can be associated with dengue but still add no forecasting value, or can leak future information.

## Priority 1 — most useful for earlier warning

### Future climate forecasts
Use forecasts that were available at the forecast issue month (not realised future weather). Suggested variables: future temperature, rainfall and humidity for H1/H2/H3. Copernicus seasonal forecast products provide operational forecasts and reforecasts. The code expects a table keyed by `fcode, issue_date, target_date` and can add these values to the horizon-specific context.

### Human mobility / importation pressure
Best input: observed origin-destination mobility or transport flows. If unavailable, the repository can construct a gravity-connectivity proxy from commune population and pairwise distance. This follows the same general fallback used in published Vietnam dengue work when detailed mobility records were unavailable. A separate script converts observed OD flows into `mobility_import_pressure`.

### Direct outbreak objective
`DEEP-EARLY-WARNING` keeps the probabilistic count distribution but adds a separate outbreak-classification head. This tests whether optimising explicitly for warning events reduces late alerts without replacing the count forecast.

## Priority 2 — strong plausible public/administrative covariates

### Water and sanitation infrastructure
Candidate aggregate features: improved/piped water access and hygienic toilet access. Vietnam research has found these variables to explain spatial heterogeneity and interactions with drought. Census-based data are the preferred source.

### Urbanisation / built environment
Candidate features: built-up land share, urban expansion, road density and night-time lights. Public satellite products can provide built-up/land-cover and night-light proxies.

### Hydrology and land surface
Candidate features: soil moisture, surface water, vegetation/NDVI, runoff and drought indicators. ERA5-Land and satellite products provide long time series suitable for lagged covariates.

### Age structure
WorldPop provides gridded age/sex population estimates for Viet Nam. Aggregate commune age shares can be joined as static/slow covariates. Their value should be tested rather than assumed.

### Dengue serotype
Best source: routine laboratory surveillance with date and location. GenBank/Nextstrain can be used for exploratory serotype circulation indicators but sequencing is sparse and non-random, so sequence counts must not be treated as population case counts. `scripts/build_serotype_features.py` converts sample-level typed records into trailing province-month serotype shares plus sample-count/availability indicators. Merge those province-month features into the commune panel before running the `serotype` ablation.

### Vector surveillance
If available from public-health partners, monthly Breteau/house/container indices or adult mosquito measures are highly relevant. These are not assumed to be public or complete, so the repository only provides a clean ingestion path.

## Priority 3 — useful but often harder to obtain or interpret

Healthcare access/reporting intensity, housing quality, wealth, travel time to hospitals, and intervention/vector-control activity can help explain observed case reporting and transmission. They should be treated as potential confounders and evaluated with blocked time validation.

## Variables intentionally not enabled by default

### Ethnicity
Aggregate census ethnicity may be obtainable, but it is sensitive, highly confounded by geography/infrastructure/socioeconomic conditions, and can produce ecological-fallacy interpretations. Do not add it simply because it is available. Use only with a pre-specified scientific rationale, aggregate data and appropriate governance.

### ABO blood group and host genetics
These are individual-level biological attributes. Published associations are mainly about individual susceptibility/severity, not reliable commune-level monthly outbreak forecasting. There is no credible, routinely updated commune-level distribution for this use case, and aggregating them would create severe ecological-fallacy and governance issues. They are therefore not default model features.

## Public source starting points

- Copernicus Climate Data Store: seasonal forecasts, ERA5/ERA5-Land.
- WorldPop: gridded population and age/sex structure for Viet Nam.
- Viet Nam National Statistics Office / 2019 Population and Housing Census: demographics, housing, water and sanitation tables.
- Google Earth Engine catalog: MODIS vegetation/land products, VIIRS night lights, JRC surface water, SRTM/elevation products.
- Nextstrain dengue (backed by GenBank): serotype/genomic metadata for exploratory surveillance.
- Meta Data for Good mobility products can be considered if access/licensing and spatial/temporal coverage are appropriate; otherwise use transport data or the gravity proxy.

## Leakage rule

Every feature must satisfy: **could this exact value have been known on `issue_date`?** If not, it must not enter a deployable forecast. Realised H1/H2/H3 climate is not a forecast feature. It may only be used in a clearly labelled oracle/upper-bound analysis, which is not part of the default scripts.
