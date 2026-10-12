#!/usr/bin/env Rscript

# Bayesian calibration of frozen deep representations.
#
# This is intentionally one model, not an ensemble. It contains no temporal
# random effect, spatial random effect, learned adjacency, or HHH4 component:
# those structures have already contributed to the supplied representation.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(paste(
    "Usage: m6_inla.R representations.csv predictions.csv",
    paste(
      "[draws] [seed] [horizon] [model_label] [pca_components] [clip_sd]",
      "[fixed_precision] [exposure_mode] [likelihood_mode]"
    )
  ))
}
input_path <- args[[1]]
output_path <- args[[2]]
n_draws <- if (length(args) >= 3L) as.integer(args[[3]]) else 500L
seed <- if (length(args) >= 4L) as.integer(args[[4]]) else 1234L
forecast_horizon <- if (length(args) >= 5L) as.integer(args[[5]]) else 3L
model_label <- if (length(args) >= 6L) args[[6]] else "M6-INLA"
pca_components <- if (length(args) >= 7L) as.integer(args[[7]]) else 0L
clip_sd <- if (length(args) >= 8L) as.numeric(args[[8]]) else Inf
fixed_precision <- if (length(args) >= 9L) as.numeric(args[[9]]) else 1
exposure_mode <- if (length(args) >= 10L) args[[10]] else "population"
likelihood_mode <- if (length(args) >= 11L) args[[11]] else "nbinomial"
if (!forecast_horizon %in% 1:3) stop("horizon must be 1, 2, or 3")
if (is.na(pca_components) || pca_components < 0L) stop("pca_components must be >= 0")
if (is.na(clip_sd) || clip_sd <= 0) stop("clip_sd must be positive")
if (is.na(fixed_precision) || fixed_precision <= 0) {
  stop("fixed_precision must be positive")
}
if (!exposure_mode %in% c("population", "unit")) {
  stop("exposure_mode must be 'population' or 'unit'")
}
if (!likelihood_mode %in% c("nbinomial", "gaussian_log1p")) {
  stop("likelihood_mode must be 'nbinomial' or 'gaussian_log1p'")
}

if (!requireNamespace("INLA", quietly = TRUE)) {
  stop("R-INLA is required; run Rscript r/install_inla.R")
}

set.seed(seed)
rows <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c(
  "split", "fcode", "issue_date", "target_date", "horizon",
  "target_cases", "population", "outbreak_threshold"
)
missing <- setdiff(required, names(rows))
if (length(missing)) {
  stop(paste("Missing representation columns:", paste(missing, collapse = ", ")))
}
z_columns <- grep("^z_[0-9]+$", names(rows), value = TRUE)
if (!length(z_columns)) {
  stop("No representation columns named z_01, z_02, ... were found")
}

# Fit a separate calibration model for each requested forecast horizon.
rows <- rows[rows$horizon == forecast_horizon, , drop = FALSE]
fit_rows <- rows$split %in% c("train", "validation")
test_rows <- rows$split == "test"
if (!any(fit_rows) || !any(test_rows)) {
  stop("Representations must contain fitting and test rows")
}

# Scaling and any dimension reduction are learned on fitting rows only and frozen.
raw_representation <- as.matrix(rows[, z_columns, drop = FALSE])
centres <- colMeans(raw_representation[fit_rows, , drop = FALSE])
spreads <- apply(raw_representation[fit_rows, , drop = FALSE], 2, stats::sd)
active <- is.finite(spreads) & spreads >= 1e-8
if (!any(active)) stop("All representation dimensions are constant on fitting rows")
standardised <- sweep(raw_representation[, active, drop = FALSE], 2, centres[active], "-")
standardised <- sweep(standardised, 2, spreads[active], "/")
if (is.finite(clip_sd)) {
  standardised[standardised < -clip_sd] <- -clip_sd
  standardised[standardised > clip_sd] <- clip_sd
}

if (pca_components > 0L) {
  component_count <- min(pca_components, ncol(standardised))
  pca <- stats::prcomp(
    standardised[fit_rows, , drop = FALSE],
    center = FALSE,
    scale. = FALSE,
    rank. = component_count
  )
  transformed <- standardised %*% pca$rotation[, seq_len(component_count), drop = FALSE]
  score_centres <- colMeans(transformed[fit_rows, , drop = FALSE])
  score_spreads <- apply(transformed[fit_rows, , drop = FALSE], 2, stats::sd)
  score_spreads[!is.finite(score_spreads) | score_spreads < 1e-8] <- 1
  transformed <- sweep(transformed, 2, score_centres, "-")
  transformed <- sweep(transformed, 2, score_spreads, "/")
  if (is.finite(clip_sd)) {
    transformed[transformed < -clip_sd] <- -clip_sd
    transformed[transformed > clip_sd] <- clip_sd
  }
  predictor_columns <- sprintf("pc_%02d", seq_len(component_count))
} else {
  transformed <- standardised
  predictor_columns <- sprintf("z_active_%02d", seq_len(ncol(transformed)))
}
rows[predictor_columns] <- as.data.frame(transformed)
rows$exposure <- if (exposure_mode == "population") rows$population / 100000 else 1
rows$response <- ifelse(
  fit_rows,
  if (likelihood_mode == "gaussian_log1p") log1p(rows$target_cases) else rows$target_cases,
  NA_real_
)
formula <- stats::as.formula(
  paste("response ~", paste(predictor_columns, collapse = " + "))
)

fit_arguments <- list(
  formula = formula,
  data = rows,
  family = if (likelihood_mode == "gaussian_log1p") "gaussian" else "nbinomial",
  control.predictor = list(compute = TRUE, link = 1),
  control.compute = list(config = TRUE, dic = TRUE, waic = TRUE, cpo = TRUE),
  control.fixed = list(
    mean.intercept = 0, prec.intercept = 0.04,
    mean = 0, prec = fixed_precision
  )
)
if (likelihood_mode == "nbinomial") fit_arguments$E <- rows$exposure
fit <- do.call(INLA::inla, fit_arguments)

posterior <- INLA::inla.posterior.sample(n_draws, fit, seed = seed)
test_indices <- which(test_rows)
sample_matrix <- matrix(
  NA_real_, nrow = length(test_indices), ncol = n_draws
)

for (draw_index in seq_along(posterior)) {
  latent <- posterior[[draw_index]]$latent
  latent_names <- rownames(latent)
  predictor_positions <- grep("^Predictor:", latent_names)
  if (length(predictor_positions) != nrow(rows)) {
    stop("Could not map INLA posterior predictors back to representation rows")
  }
  linear_predictor <- as.numeric(latent[predictor_positions, 1])
  hyperparameter <- as.numeric(posterior[[draw_index]]$hyperpar[1])
  if (likelihood_mode == "gaussian_log1p") {
    observation_precision <- exp(hyperparameter)
    log_cases <- stats::rnorm(
      length(test_indices),
      mean = linear_predictor[test_indices],
      sd = sqrt(1 / observation_precision)
    )
    sample_matrix[, draw_index] <- pmax(expm1(log_cases), 0)
  } else {
    size <- exp(hyperparameter)
    expected_cases <- exp(linear_predictor[test_indices]) * rows$exposure[test_indices]
    sample_matrix[, draw_index] <- stats::rnbinom(
      length(test_indices), mu = expected_cases, size = size
    )
  }
}

test <- rows[test_rows, required, drop = FALSE]
test$model <- model_label
test$predicted_mean <- rowMeans(sample_matrix)
test$predicted_median <- apply(sample_matrix, 1, stats::median)
test$lower_50 <- apply(sample_matrix, 1, stats::quantile, probs = 0.25)
test$upper_50 <- apply(sample_matrix, 1, stats::quantile, probs = 0.75)
test$lower_80 <- apply(sample_matrix, 1, stats::quantile, probs = 0.10)
test$upper_80 <- apply(sample_matrix, 1, stats::quantile, probs = 0.90)
test$lower_95 <- apply(sample_matrix, 1, stats::quantile, probs = 0.025)
test$upper_95 <- apply(sample_matrix, 1, stats::quantile, probs = 0.975)
test$outbreak_probability <- rowMeans(
  sweep(sample_matrix, 1, test$outbreak_threshold, FUN = ">")
)
test <- test[
  c(
    "model", "fcode", "issue_date", "target_date", "horizon",
    "target_cases", "population", "predicted_mean", "predicted_median",
    "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95",
    "outbreak_threshold", "outbreak_probability"
  )
]

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
writeLines(
  capture.output(sessionInfo()),
  file.path(dirname(output_path), "r_session_info.txt")
)
write.csv(test, output_path, row.names = FALSE)

observed_incidence <- test$target_cases / test$population * 100000
sample_incidence <- sample_matrix / test$population * 100000
predicted_incidence <- rowMeans(sample_incidence)
error <- predicted_incidence - observed_incidence
coverage <- function(level) {
  tail <- (1 - level) / 2
  lower <- apply(sample_incidence, 1, stats::quantile, probs = tail)
  upper <- apply(sample_incidence, 1, stats::quantile, probs = 1 - tail)
  mean(observed_incidence >= lower & observed_incidence <= upper)
}
crps_row <- function(observed, draws) {
  ordered <- sort(draws)
  count <- length(ordered)
  weights <- 2 * seq_len(count) - count - 1
  mean(abs(draws - observed)) - sum(ordered * weights) / (count * count)
}
crps <- vapply(
  seq_len(nrow(test)),
  function(index) crps_row(observed_incidence[index], sample_incidence[index, ]),
  numeric(1)
)
outbreak_observed <- as.numeric(test$target_cases > test$outbreak_threshold)
outbreak_decision_threshold <- 0.50
outbreak_predicted <- as.numeric(
  test$outbreak_probability >= outbreak_decision_threshold
)
tp <- sum(outbreak_observed == 1 & outbreak_predicted == 1)
fp <- sum(outbreak_observed == 0 & outbreak_predicted == 1)
fn <- sum(outbreak_observed == 1 & outbreak_predicted == 0)
tn <- sum(outbreak_observed == 0 & outbreak_predicted == 0)
safe_ratio <- function(numerator, denominator) {
  if (denominator == 0) 0 else numerator / denominator
}
precision <- safe_ratio(tp, tp + fp)
recall <- safe_ratio(tp, tp + fn)
sensitivity <- recall
specificity <- safe_ratio(tn, tn + fp)
metrics <- data.frame(
  model = model_label,
  split = "test",
  horizon = forecast_horizon,
  n = nrow(test),
  mae = mean(abs(error)),
  rmse = sqrt(mean(error ^ 2)),
  bias = mean(error),
  crps = mean(crps),
  coverage_50 = coverage(0.50),
  coverage_80 = coverage(0.80),
  coverage_95 = coverage(0.95),
  outbreak_brier = mean((test$outbreak_probability - outbreak_observed) ^ 2),
  outbreak_decision_threshold = outbreak_decision_threshold,
  outbreak_observed_positive = sum(outbreak_observed),
  outbreak_predicted_positive = sum(outbreak_predicted),
  outbreak_tp = tp,
  outbreak_fp = fp,
  outbreak_fn = fn,
  outbreak_tn = tn,
  outbreak_accuracy = safe_ratio(tp + tn, length(outbreak_observed)),
  outbreak_precision = precision,
  outbreak_specificity = specificity,
  outbreak_recall = recall,
  outbreak_sensitivity = sensitivity,
  outbreak_true_negative_rate = specificity,
  outbreak_balanced_accuracy = (sensitivity + specificity) / 2,
  outbreak_f1 = safe_ratio(2 * precision * recall, precision + recall)
)
write.csv(metrics, file.path(dirname(output_path), "metrics.csv"), row.names = FALSE)

write.csv(
  data.frame(
    model = model_label,
    horizon = forecast_horizon,
    formula = deparse(formula)[1],
    dic = fit$dic$dic,
    waic = fit$waic$waic,
    n_fit = sum(fit_rows),
    n_test = sum(test_rows),
    input_representation_dimensions = length(z_columns),
    active_representation_dimensions = sum(active),
    representation_dimensions = length(predictor_columns),
    pca_components = pca_components,
    clip_sd = clip_sd,
    fixed_precision = fixed_precision,
    exposure_mode = exposure_mode,
    likelihood_mode = likelihood_mode
  ),
  file.path(dirname(output_path), "diagnostics.csv"),
  row.names = FALSE
)
