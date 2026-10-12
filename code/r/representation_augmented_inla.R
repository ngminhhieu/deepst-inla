#!/usr/bin/env Rscript

# INLA-TS augmented with a frozen deep representation.  This script covers the
# fusion families that are not represented by m6_inla.R: retain the original
# temporal fixed effects and BYM2 field, then add latent PCs, residualised PCs,
# a supervised deep score, a nonlinear score effect, or a deep offset.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5L) {
  stop(paste(
    "Usage: representation_augmented_inla.R rows_cache.rds adjacency.csv",
    "representation.csv output_dir horizon [draws] [seed] [model_label]",
    "[fusion_mode] [pca_components] [clip_sd] [deep_precision] [eval_split]"
  ))
}

rows_cache_path <- args[[1L]]
adjacency_path <- args[[2L]]
representation_path <- args[[3L]]
output_dir <- args[[4L]]
forecast_horizon <- as.integer(args[[5L]])
n_draws <- if (length(args) >= 6L) as.integer(args[[6L]]) else 200L
seed <- if (length(args) >= 7L) as.integer(args[[7L]]) else 1234L
model_label <- if (length(args) >= 8L) args[[8L]] else "INLA-TS-DEEP"
fusion_mode <- if (length(args) >= 9L) args[[9L]] else "score_linear"
pca_components <- if (length(args) >= 10L) as.integer(args[[10L]]) else 4L
clip_sd <- if (length(args) >= 11L) as.numeric(args[[11L]]) else 5
deep_precision <- if (length(args) >= 12L) as.numeric(args[[12L]]) else 4
evaluation_split <- if (length(args) >= 13L) tolower(args[[13L]]) else "validation"

valid_modes <- c(
  "latent_pca", "latent_residual_pca", "score_linear", "score_hinge",
  "score_rw2", "offset", "offset_shrink"
)
if (!forecast_horizon %in% 1:3) stop("horizon must be 1, 2, or 3")
if (!fusion_mode %in% valid_modes) stop("Unknown fusion mode: ", fusion_mode)
if (!evaluation_split %in% c("validation", "test")) {
  stop("eval_split must be validation or test")
}
if (!is.finite(clip_sd) || clip_sd <= 0) stop("clip_sd must be positive")
if (!is.finite(deep_precision) || deep_precision <= 0) {
  stop("deep_precision must be positive")
}
if (!requireNamespace("INLA", quietly = TRUE)) stop("R-INLA is required")
INLA::inla.setOption(
  num.threads = Sys.getenv("DENGUE_INLA_THREADS", "32:1")
)

set.seed(seed)
cached <- readRDS(rows_cache_path)
rows <- cached$rows
units <- as.character(cached$units)
rows$fcode <- as.character(rows$fcode)
rows$issue_date <- as.Date(rows$issue_date)
rows$target_date <- as.Date(rows$target_date)
rows <- rows[rows$horizon == forecast_horizon, , drop = FALSE]

excluded_units_path <- Sys.getenv("DENGUE_EXCLUDED_UNITS_FILE", "")
excluded_units <- if (nzchar(excluded_units_path)) {
  if (!file.exists(excluded_units_path)) stop("Missing excluded-units file: ", excluded_units_path)
  unique(trimws(readLines(excluded_units_path, warn = FALSE)))
} else character()
excluded_units <- excluded_units[nzchar(excluded_units)]

deep <- read.csv(representation_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c(
  "split", "fcode", "issue_date", "target_date", "horizon", "target_cases",
  "population", "outbreak_threshold"
)
missing <- setdiff(required, names(deep))
if (length(missing)) stop("Missing representation columns: ", paste(missing, collapse = ", "))
deep$fcode <- as.character(deep$fcode)
deep$issue_date <- as.Date(deep$issue_date)
deep$target_date <- as.Date(deep$target_date)
deep <- deep[deep$horizon == forecast_horizon, , drop = FALSE]

make_key <- function(frame) paste(
  frame$fcode, frame$issue_date, frame$target_date, frame$horizon, sep = "|"
)
positions <- match(make_key(rows), make_key(deep))
if (anyNA(positions)) stop("Deep representation does not cover all cached INLA rows")
deep <- deep[positions, , drop = FALSE]
if (any(abs(rows$target_cases - deep$target_cases) > 1e-8)) {
  stop("Target mismatch after aligning INLA and deep rows")
}

fit_rows <- if (evaluation_split == "validation") {
  rows$split == "train"
} else {
  rows$split %in% c("train", "validation")
}
fit_rows <- fit_rows & !rows$fcode %in% excluded_units
test_rows <- rows$split == evaluation_split
if (!any(fit_rows) || !any(test_rows)) stop("Empty fitting or evaluation split")

temporal_columns <- c(
  "log_recent_0", "log_recent_1", "log_recent_2", "log_recent_5",
  "log_recent_11", paste0("seasonal_log_", 1:5),
  paste0("seasonal_available_", 1:5), "seasonal_mean", "seasonal_sd",
  "seasonal_trend", "seasonal_zero_share", "target_sine", "target_cosine"
)
for (column in temporal_columns) {
  centre <- mean(rows[[column]][fit_rows])
  spread <- stats::sd(rows[[column]][fit_rows])
  if (!is.finite(spread) || spread < 1e-8) spread <- 1
  rows[[column]] <- (rows[[column]] - centre) / spread
}

standardise <- function(values) {
  values <- as.matrix(values)
  centres <- colMeans(values[fit_rows, , drop = FALSE])
  spreads <- apply(values[fit_rows, , drop = FALSE], 2L, stats::sd)
  active <- is.finite(spreads) & spreads >= 1e-8
  if (!any(active)) stop("All deep representation dimensions are constant")
  result <- sweep(values[, active, drop = FALSE], 2L, centres[active], "-")
  result <- sweep(result, 2L, spreads[active], "/")
  pmax(pmin(result, clip_sd), -clip_sd)
}

deep_terms <- character()
random_deep_term <- character()
offset_term <- character()
deep_dimension <- 0L

if (fusion_mode %in% c("latent_pca", "latent_residual_pca")) {
  z_columns <- grep("^z_[0-9]+$", names(deep), value = TRUE)
  if (!length(z_columns)) stop("No z_XX latent columns were found")
  transformed <- standardise(deep[, z_columns, drop = FALSE])
  if (fusion_mode == "latent_residual_pca") {
    design <- cbind(1, as.matrix(rows[, temporal_columns, drop = FALSE]))
    ridge <- diag(ncol(design)) * 0.1
    ridge[1L, 1L] <- 0
    coefficients <- solve(
      crossprod(design[fit_rows, , drop = FALSE]) + ridge,
      crossprod(design[fit_rows, , drop = FALSE], transformed[fit_rows, , drop = FALSE])
    )
    transformed <- transformed - design %*% coefficients
    transformed <- standardise(transformed)
  }
  component_count <- min(pca_components, ncol(transformed))
  pca <- stats::prcomp(
    transformed[fit_rows, , drop = FALSE], center = FALSE, scale. = FALSE,
    rank. = component_count
  )
  transformed <- transformed %*% pca$rotation[, seq_len(component_count), drop = FALSE]
  transformed <- standardise(transformed) / sqrt(deep_precision)
  deep_terms <- sprintf("deep_pc_%02d", seq_len(ncol(transformed)))
  rows[deep_terms] <- as.data.frame(transformed)
  deep_dimension <- length(deep_terms)
} else {
  if (!"deep_expected_cases" %in% names(deep)) {
    stop(fusion_mode, " requires a deep_expected_cases column")
  }
  exposure <- rows$population / 100000
  raw_offset <- log(pmax(as.numeric(deep$deep_expected_cases), 0.05) / exposure)
  fit_limits <- stats::quantile(
    raw_offset[fit_rows], probs = c(0.001, 0.999), na.rm = TRUE, names = FALSE
  )
  raw_offset <- pmax(pmin(raw_offset, fit_limits[2L]), fit_limits[1L])
  score <- as.numeric(standardise(raw_offset))
  if (fusion_mode == "score_linear") {
    rows$deep_score <- score / sqrt(deep_precision)
    deep_terms <- "deep_score"
    deep_dimension <- 1L
  } else if (fusion_mode == "score_hinge") {
    hinge_point <- as.numeric(stats::quantile(score[fit_rows], 0.90))
    rows$deep_score <- score / sqrt(deep_precision)
    rows$deep_peak_hinge <- pmax(score - hinge_point, 0) / sqrt(deep_precision)
    deep_terms <- c("deep_score", "deep_peak_hinge")
    deep_dimension <- 2L
  } else if (fusion_mode == "score_rw2") {
    breaks <- unique(as.numeric(stats::quantile(
      score[fit_rows], probs = seq(0, 1, length.out = 31L), names = FALSE
    )))
    if (length(breaks) < 5L) stop("Too few unique deep-score bins")
    rows$deep_score_bin <- findInterval(score, breaks, all.inside = TRUE)
    random_deep_term <- "f(deep_score_bin, model='rw2', constr=TRUE, scale.model=TRUE)"
    deep_dimension <- length(breaks) - 1L
  } else if (fusion_mode == "offset") {
    rows$deep_offset <- raw_offset
    offset_term <- "offset(deep_offset)"
    deep_dimension <- 1L
  } else if (fusion_mode == "offset_shrink") {
    # The offset supplies coefficient one.  The extra centred term estimates a
    # regularised deviation from one, so INLA can attenuate or amplify it.
    offset_centre <- mean(raw_offset[fit_rows])
    rows$deep_offset <- raw_offset
    rows$deep_offset_adjustment <- (raw_offset - offset_centre) / sqrt(deep_precision)
    offset_term <- "offset(deep_offset)"
    deep_terms <- "deep_offset_adjustment"
    deep_dimension <- 1L
  }
}

rows$commune_index <- match(rows$fcode, units)
rows$exposure <- rows$population / 100000
rows$response <- ifelse(fit_rows, rows$target_cases, NA_real_)

edges <- read.csv(adjacency_path, stringsAsFactors = FALSE, check.names = FALSE)
edge_columns <- if (all(c("from", "to") %in% names(edges))) {
  c("from", "to")
} else if (all(c("from_fcode", "to_fcode") %in% names(edges))) {
  c("from_fcode", "to_fcode")
} else stop("Adjacency must contain from/to or from_fcode/to_fcode")
edges[[edge_columns[1L]]] <- as.character(edges[[edge_columns[1L]]])
edges[[edge_columns[2L]]] <- as.character(edges[[edge_columns[2L]]])
neighbours <- vector("list", length(units))
for (index in seq_len(nrow(edges))) {
  left <- match(edges[[edge_columns[1L]]][index], units)
  right <- match(edges[[edge_columns[2L]]][index], units)
  if (!is.na(left) && !is.na(right) && left != right) {
    neighbours[[left]] <- unique(c(neighbours[[left]], right))
    neighbours[[right]] <- unique(c(neighbours[[right]], left))
  }
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(
  capture.output(sessionInfo()),
  file.path(output_dir, "r_session_info.txt")
)
graph_path <- file.path(output_dir, "administrative_graph.adj")
graph_lines <- c(as.character(length(units)))
for (index in seq_along(units)) {
  adjacent <- sort(neighbours[[index]])
  graph_lines <- c(graph_lines, paste(c(index, length(adjacent), adjacent), collapse = " "))
}
writeLines(graph_lines, graph_path)
graph <- INLA::inla.read.graph(graph_path)

terms <- c(
  temporal_columns,
  deep_terms,
  random_deep_term,
  offset_term,
  "f(commune_index, model='bym2', graph=graph, scale.model=TRUE, constr=TRUE)"
)
terms <- terms[nzchar(terms)]
formula <- stats::as.formula(paste("response ~", paste(terms, collapse = " + ")))
message("Fitting ", model_label, " H", forecast_horizon, " (", evaluation_split, ")")
fit <- INLA::inla(
  formula, data = rows, family = "nbinomial", E = exposure,
  num.threads = Sys.getenv("DENGUE_INLA_THREADS", "32:1"),
  control.predictor = list(compute = TRUE, link = 1),
  control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
  control.fixed = list(mean.intercept = 0, prec.intercept = 0.04, mean = 0, prec = 1)
)

test <- rows[test_rows, c(
  "fcode", "issue_date", "target_date", "horizon", "target_cases", "population",
  "outbreak_threshold", "exposure"
)]
predictor <- fit$summary.linear.predictor[test_rows, ]
location <- predictor$mean
uncertainty <- pmax(predictor$sd, 1e-6)
size <- as.numeric(fit$summary.hyperpar$mean[1L])
if (!is.finite(size) || size <= 0) size <- 1
latent <- matrix(
  stats::rnorm(nrow(test) * n_draws, rep(location, n_draws), rep(uncertainty, n_draws)),
  nrow = nrow(test), ncol = n_draws
)
expected <- exp(pmin(latent, 20)) * test$exposure
samples <- matrix(
  stats::rnbinom(length(expected), mu = as.vector(expected), size = size),
  nrow = nrow(test), ncol = n_draws
)
writeBin(
  as.double(samples), file.path(output_dir, sprintf("samples_H%d.bin", forecast_horizon)),
  size = 8L
)
write.csv(
  data.frame(rows = nrow(samples), draws = ncol(samples)),
  file.path(output_dir, sprintf("samples_H%d_shape.csv", forecast_horizon)), row.names = FALSE
)

test$model <- model_label
test$predicted_mean <- rowMeans(samples)
test$predicted_median <- apply(samples, 1L, stats::median)
for (item in list(c(50, .25, .75), c(80, .10, .90), c(95, .025, .975))) {
  level <- item[1L]
  test[[paste0("lower_", level)]] <- apply(samples, 1L, stats::quantile, probs = item[2L])
  test[[paste0("upper_", level)]] <- apply(samples, 1L, stats::quantile, probs = item[3L])
}
test$outbreak_probability <- rowMeans(sweep(samples, 1L, test$outbreak_threshold, FUN = ">"))
prediction_columns <- c(
  "model", "fcode", "issue_date", "target_date", "horizon", "target_cases",
  "population", "predicted_mean", "predicted_median", "lower_50", "upper_50",
  "lower_80", "upper_80", "lower_95", "upper_95", "outbreak_threshold",
  "outbreak_probability"
)
write.csv(test[prediction_columns], file.path(output_dir, "predictions.csv"), row.names = FALSE)

crps_rows <- function(observed, values) {
  ordered <- t(apply(values, 1L, sort))
  count <- ncol(values)
  weights <- 2 * seq_len(count) - count - 1
  rowMeans(abs(values - observed)) - as.vector(ordered %*% weights) / (count * count)
}
safe_ratio <- function(a, b) if (b == 0) 0 else a / b
observed_incidence <- test$target_cases / test$population * 100000
sample_incidence <- samples / test$population * 100000
error <- rowMeans(sample_incidence) - observed_incidence
coverage <- function(level) {
  tail <- (1 - level) / 2
  lower <- apply(sample_incidence, 1L, stats::quantile, probs = tail)
  upper <- apply(sample_incidence, 1L, stats::quantile, probs = 1 - tail)
  mean(observed_incidence >= lower & observed_incidence <= upper)
}
observed_outbreak <- test$target_cases > test$outbreak_threshold
predicted_outbreak <- test$outbreak_probability >= 0.5
tp <- sum(observed_outbreak & predicted_outbreak)
fp <- sum(!observed_outbreak & predicted_outbreak)
fn <- sum(observed_outbreak & !predicted_outbreak)
tn <- sum(!observed_outbreak & !predicted_outbreak)
precision <- safe_ratio(tp, tp + fp)
recall <- safe_ratio(tp, tp + fn)
sensitivity <- recall
specificity <- safe_ratio(tn, tn + fp)
metrics <- data.frame(
  model = model_label, split = evaluation_split, horizon = forecast_horizon,
  n = nrow(test), mae = mean(abs(error)), rmse = sqrt(mean(error^2)),
  bias = mean(error), crps = mean(crps_rows(observed_incidence, sample_incidence)),
  coverage_50 = coverage(.50), coverage_80 = coverage(.80), coverage_95 = coverage(.95),
  outbreak_brier = mean((test$outbreak_probability - observed_outbreak)^2),
  outbreak_decision_threshold = .50, outbreak_observed_positive = sum(observed_outbreak),
  outbreak_predicted_positive = sum(predicted_outbreak), outbreak_tp = tp,
  outbreak_fp = fp, outbreak_fn = fn, outbreak_tn = tn,
  outbreak_accuracy = safe_ratio(tp + tn, length(observed_outbreak)),
  outbreak_precision = precision, outbreak_specificity = specificity,
  outbreak_recall = recall, outbreak_sensitivity = sensitivity,
  outbreak_true_negative_rate = specificity,
  outbreak_balanced_accuracy = (sensitivity + specificity) / 2,
  outbreak_f1 = safe_ratio(2 * precision * recall, precision + recall)
)
write.csv(metrics, file.path(output_dir, "metrics.csv"), row.names = FALSE)
write.csv(data.frame(
  model = model_label, split = evaluation_split, horizon = forecast_horizon,
  fusion_mode = fusion_mode, pca_components = pca_components, clip_sd = clip_sd,
  deep_precision = deep_precision, deep_dimension = deep_dimension,
  formula = paste(deparse(formula), collapse = " "), dic = fit$dic$dic,
  waic = fit$waic$waic, n_fit = sum(fit_rows), n_test = sum(test_rows),
  excluded_unit_count = length(excluded_units)
), file.path(output_dir, "diagnostics.csv"), row.names = FALSE)
message("Completed ", model_label, " H", forecast_horizon)
