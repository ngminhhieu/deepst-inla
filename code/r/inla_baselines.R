#!/usr/bin/env Rscript

# Reproducible INLA-only H1--H3 baselines using the same rolling-origin information
# and outbreak definition as the neural models.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop(
    "Usage: inla_baselines.R monthly.csv adjacency.csv results_dir [draws] [seed]"
  )
}
monthly_path <- args[[1]]
adjacency_path <- args[[2]]
results_dir <- args[[3]]
n_draws <- if (length(args) >= 4L) as.integer(args[[4]]) else 500L
seed <- if (length(args) >= 5L) as.integer(args[[5]]) else 1234L
evaluation_split <- tolower(Sys.getenv("DENGUE_EVAL_SPLIT", "test"))
if (!evaluation_split %in% c("validation", "test")) {
  stop("DENGUE_EVAL_SPLIT must be validation or test")
}
train_start_date <- as.Date(Sys.getenv("DENGUE_TRAIN_START", "2010-01-01"))
train_end_date <- as.Date(Sys.getenv("DENGUE_TRAIN_END", "2020-12-01"))
validation_start_date <- as.Date(Sys.getenv("DENGUE_VALIDATION_START", "2021-01-01"))
validation_end_date <- as.Date(Sys.getenv("DENGUE_VALIDATION_END", "2022-12-01"))
test_start_date <- as.Date(Sys.getenv("DENGUE_TEST_START", "2023-01-01"))
test_end_date <- as.Date(Sys.getenv("DENGUE_TEST_END", "2025-12-01"))
split_on_target_date <- tolower(Sys.getenv("DENGUE_SPLIT_ON_TARGET_DATE", "false")) %in%
  c("1", "true", "yes")

if (!requireNamespace("INLA", quietly = TRUE)) {
  stop("R-INLA is required; set R_LIBS_USER to the repository .r-library")
}
set.seed(seed)

rows_cache_path <- Sys.getenv("DENGUE_ROWS_CACHE", "")
if (nzchar(rows_cache_path) && file.exists(rows_cache_path)) {
  cached_rows <- readRDS(rows_cache_path)
  rows <- cached_rows$rows
  units <- cached_rows$units
} else {
monthly <- read.csv(monthly_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("fcode", "year", "month", "obs_dengue_cases", "population")
missing <- setdiff(required, names(monthly))
if (length(missing)) stop(paste("Missing monthly columns:", paste(missing, collapse = ", ")))
monthly$fcode <- as.character(monthly$fcode)
monthly$date <- as.Date(sprintf("%04d-%02d-01", monthly$year, monthly$month))
monthly <- monthly[order(monthly$fcode, monthly$date), ]
units <- sort(unique(monthly$fcode))

make_rows <- function(unit) {
  frame <- monthly[monthly$fcode == unit, ]
  frame <- frame[order(frame$date), ]
  cases <- as.numeric(frame$obs_dengue_cases)
  output <- list()
  output_index <- 0L
  for (horizon in 1:3) for (target in seq_len(nrow(frame))) {
    issue <- target - horizon
    if (issue < 12L) next
    # All horizons share forecast origins. Split membership is anchored by
    # the H3 target at that origin, matching the joint neural pipeline.
    if (issue + 3L > nrow(frame)) next
    target_date <- frame$date[target]
    split_anchor_date <- frame$date[issue + 3L]
    if (split_anchor_date < as.Date("2011-03-01") ||
        target_date > test_end_date) next
    recent_positions <- issue - c(0L, 1L, 2L, 5L, 11L)
    seasonal_positions <- target - 12L * seq_len(5L)
    available <- seasonal_positions >= 1L
    seasonal_raw <- rep(0, 5L)
    seasonal_raw[available] <- cases[seasonal_positions[available]]
    seasonal_log <- log1p(seasonal_raw)
    available_log <- seasonal_log[available]
    seasonal_mean <- if (length(available_log)) mean(available_log) else 0
    seasonal_sd <- if (length(available_log) > 1L) stats::sd(available_log) else 0
    seasonal_trend <- if (length(available_log) > 1L) {
      available_log[1L] - available_log[length(available_log)]
    } else 0
    seasonal_zero_share <- if (any(available)) mean(seasonal_raw[available] == 0) else 0
    threshold <- if (sum(available) >= 2L) {
      mean(seasonal_raw[available]) + 2 * stats::sd(seasonal_raw[available])
    } else Inf
    split_date <- if (split_on_target_date) target_date else split_anchor_date
    split <- if (split_date >= train_start_date && split_date <= train_end_date) {
      "train"
    } else if (split_date >= validation_start_date && split_date <= validation_end_date) {
      "validation"
    } else if (split_date >= test_start_date && split_date <= test_end_date) {
      "test"
    } else {
      "excluded"
    }
    output_index <- output_index + 1L
    row <- data.frame(
      split = split,
      fcode = unit,
      issue_date = frame$date[issue],
      target_date = target_date,
      horizon = horizon,
      target_cases = cases[target],
      population = as.numeric(frame$population[target]),
      outbreak_threshold = threshold,
      log_recent_0 = log1p(cases[recent_positions[1L]]),
      log_recent_1 = log1p(cases[recent_positions[2L]]),
      log_recent_2 = log1p(cases[recent_positions[3L]]),
      log_recent_5 = log1p(cases[recent_positions[4L]]),
      log_recent_11 = log1p(cases[recent_positions[5L]]),
      seasonal_mean = seasonal_mean,
      seasonal_sd = seasonal_sd,
      seasonal_trend = seasonal_trend,
      seasonal_zero_share = seasonal_zero_share,
      target_sine = sin(2 * pi * as.integer(format(target_date, "%m")) / 12),
      target_cosine = cos(2 * pi * as.integer(format(target_date, "%m")) / 12),
      stringsAsFactors = FALSE
    )
    for (lag in seq_len(5L)) {
      row[[paste0("seasonal_log_", lag)]] <- seasonal_log[lag]
      row[[paste0("seasonal_available_", lag)]] <- as.numeric(available[lag])
    }
    output[[output_index]] <- row
  }
  do.call(rbind, output[seq_len(output_index)])
}

rows <- do.call(rbind, lapply(units, make_rows))
rows <- rows[order(rows$target_date, rows$fcode), ]
if (nzchar(rows_cache_path)) {
  dir.create(dirname(rows_cache_path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(list(rows = rows, units = units), rows_cache_path)
}
}
if (tolower(Sys.getenv("DENGUE_CACHE_ONLY", "false")) %in% c("1", "true", "yes")) {
  quit(save = "no", status = 0L)
}
rows$commune_index <- match(rows$fcode, units)
rows$commune_dynamic_index <- rows$commune_index
rows$time_index <- match(rows$target_date, sort(unique(rows$target_date)))
rows$exposure <- rows$population / 100000
fit_rows <- if (evaluation_split == "validation") {
  rows$split == "train"
} else {
  rows$split %in% c("train", "validation")
}
test_rows <- rows$split == evaluation_split

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

edges <- read.csv(adjacency_path, stringsAsFactors = FALSE, check.names = FALSE)
edge_columns <- if (all(c("from", "to") %in% names(edges))) {
  c("from", "to")
} else if (all(c("from_fcode", "to_fcode") %in% names(edges))) {
  c("from_fcode", "to_fcode")
} else {
  stop("Adjacency must contain from/to or from_fcode/to_fcode")
}
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
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(
  capture.output(sessionInfo()),
  file.path(results_dir, "r_session_info.txt")
)
graph_path <- file.path(results_dir, "administrative_graph.adj")
graph_lines <- c(as.character(length(units)))
for (index in seq_along(units)) {
  adjacent <- sort(neighbours[[index]])
  graph_lines <- c(
    graph_lines,
    paste(c(index, length(adjacent), adjacent), collapse = " ")
  )
}
writeLines(graph_lines, graph_path)
graph <- INLA::inla.read.graph(graph_path)

crps_rows <- function(observed, samples) {
  ordered <- t(apply(samples, 1L, sort))
  count <- ncol(samples)
  weights <- 2 * seq_len(count) - count - 1
  rowMeans(abs(samples - observed)) - as.vector(ordered %*% weights) / (count * count)
}

safe_ratio <- function(numerator, denominator) {
  if (denominator == 0) 0 else numerator / denominator
}

fit_one <- function(model_name, temporal, spatial, forecast_horizon,
                    temporal_random = FALSE, dynamic_spatial = FALSE) {
  message("Fitting ", model_name, " H", forecast_horizon)
  model_rows <- rows[rows$horizon == forecast_horizon, , drop = FALSE]
  model_fit_rows <- if (evaluation_split == "validation") {
    model_rows$split == "train"
  } else {
    model_rows$split %in% c("train", "validation")
  }
  model_test_rows <- model_rows$split == evaluation_split
  model_rows$response <- ifelse(
    model_fit_rows, model_rows$target_cases, NA_real_
  )
  terms <- if (temporal) temporal_columns else character()
  if (spatial) {
    terms <- c(
      terms,
      "f(commune_index, model='bym2', graph=graph, scale.model=TRUE, constr=TRUE)"
    )
  }
  if (temporal_random) {
    terms <- c(terms, "f(time_index, model='ar1', constr=TRUE)")
  }
  if (dynamic_spatial) {
    terms <- c(
      terms,
      paste0(
        "f(commune_dynamic_index, model='besag', graph=graph, ",
        "group=time_index, control.group=list(model='ar1'), ",
        "scale.model=TRUE, constr=TRUE)"
      )
    )
  }
  formula <- stats::as.formula(
    paste("response ~", if (length(terms)) paste(terms, collapse = " + ") else "1")
  )
  fit <- INLA::inla(
    formula,
    data = model_rows,
    family = "nbinomial",
    E = exposure,
    control.predictor = list(compute = TRUE, link = 1),
    control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
    control.fixed = list(mean.intercept = 0, prec.intercept = 0.04, mean = 0, prec = 1)
  )
  test <- model_rows[model_test_rows, c(
    "fcode", "issue_date", "target_date", "horizon", "target_cases",
    "population", "outbreak_threshold", "exposure"
  )]
  predictor <- fit$summary.linear.predictor[model_test_rows, ]
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
  if (identical(tolower(Sys.getenv("DENGUE_SAVE_SAMPLES", "false")), "true")) {
    sample_dir <- file.path(results_dir, model_name)
    dir.create(sample_dir, recursive = TRUE, showWarnings = FALSE)
    writeBin(
      as.double(samples),
      file.path(sample_dir, sprintf("samples_H%d.bin", forecast_horizon)),
      size = 8L
    )
    write.csv(
      data.frame(rows = nrow(samples), draws = ncol(samples)),
      file.path(sample_dir, sprintf("samples_H%d_shape.csv", forecast_horizon)),
      row.names = FALSE
    )
  }
  test$model <- model_name
  test$predicted_mean <- rowMeans(samples)
  test$predicted_median <- apply(samples, 1L, stats::median)
  test$lower_50 <- apply(samples, 1L, stats::quantile, probs = 0.25)
  test$upper_50 <- apply(samples, 1L, stats::quantile, probs = 0.75)
  test$lower_80 <- apply(samples, 1L, stats::quantile, probs = 0.10)
  test$upper_80 <- apply(samples, 1L, stats::quantile, probs = 0.90)
  test$lower_95 <- apply(samples, 1L, stats::quantile, probs = 0.025)
  test$upper_95 <- apply(samples, 1L, stats::quantile, probs = 0.975)
  test$outbreak_probability <- rowMeans(
    sweep(samples, 1L, test$outbreak_threshold, FUN = ">")
  )
  prediction_columns <- c(
    "model", "fcode", "issue_date", "target_date", "horizon", "target_cases",
    "population", "predicted_mean", "predicted_median", "lower_50", "upper_50",
    "lower_80", "upper_80", "lower_95", "upper_95", "outbreak_threshold",
    "outbreak_probability"
  )
  model_dir <- file.path(results_dir, model_name)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  prediction_path <- file.path(model_dir, "predictions.csv")
  if (forecast_horizon == 1L) {
    write.csv(test[prediction_columns], prediction_path, row.names = FALSE)
  } else {
    write.table(
      test[prediction_columns], prediction_path, append = TRUE, row.names = FALSE,
      col.names = FALSE, sep = ","
    )
  }

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
  predicted_outbreak <- test$outbreak_probability >= 0.50
  tp <- sum(observed_outbreak & predicted_outbreak)
  fp <- sum(!observed_outbreak & predicted_outbreak)
  fn <- sum(observed_outbreak & !predicted_outbreak)
  tn <- sum(!observed_outbreak & !predicted_outbreak)
  precision <- safe_ratio(tp, tp + fp)
  recall <- safe_ratio(tp, tp + fn)
  sensitivity <- recall
  specificity <- safe_ratio(tn, tn + fp)
  metrics <- data.frame(
    model = model_name, split = evaluation_split, horizon = forecast_horizon, n = nrow(test),
    mae = mean(abs(error)), rmse = sqrt(mean(error^2)), bias = mean(error),
    crps = mean(crps_rows(observed_incidence, sample_incidence)),
    coverage_50 = coverage(0.50), coverage_80 = coverage(0.80),
    coverage_95 = coverage(0.95),
    outbreak_brier = mean((test$outbreak_probability - observed_outbreak)^2),
    outbreak_decision_threshold = 0.50,
    outbreak_observed_positive = sum(observed_outbreak),
    outbreak_predicted_positive = sum(predicted_outbreak),
    outbreak_tp = tp, outbreak_fp = fp, outbreak_fn = fn, outbreak_tn = tn,
    outbreak_accuracy = safe_ratio(tp + tn, length(observed_outbreak)),
    outbreak_precision = precision, outbreak_specificity = specificity,
    outbreak_recall = recall, outbreak_sensitivity = sensitivity,
    outbreak_true_negative_rate = specificity,
    outbreak_balanced_accuracy = (sensitivity + specificity) / 2,
    outbreak_f1 = safe_ratio(2 * precision * recall, precision + recall)
  )
  metrics_path <- file.path(model_dir, "metrics.csv")
  if (forecast_horizon == 1L) {
    write.csv(metrics, metrics_path, row.names = FALSE)
  } else {
    write.table(
      metrics, metrics_path, append = TRUE, row.names = FALSE,
      col.names = FALSE, sep = ","
    )
  }
  diagnostics <- data.frame(
    model = model_name, horizon = forecast_horizon,
    formula = paste(deparse(formula), collapse = " "), dic = fit$dic$dic,
    waic = fit$waic$waic, n_fit = sum(model_fit_rows), n_test = sum(model_test_rows),
    n_fixed = nrow(fit$summary.fixed),
    n_spatial = if (spatial) length(units) else 0,
    n_temporal = if (temporal_random) length(unique(model_rows$time_index)) else 0,
    n_dynamic_spatial = if (dynamic_spatial) {
      length(units) * length(unique(model_rows$time_index))
    } else 0
  )
  diagnostics_path <- file.path(model_dir, "diagnostics.csv")
  if (forecast_horizon == 1L) {
    write.csv(diagnostics, diagnostics_path, row.names = FALSE)
  } else {
    write.table(
      diagnostics, diagnostics_path, append = TRUE, row.names = FALSE,
      col.names = FALSE, sep = ","
    )
  }
  message("Completed ", model_name, " H", forecast_horizon)
}

if (!identical(tolower(Sys.getenv("DENGUE_SKIP_STATIC_INLA", "false")), "true")) {
  requested_static <- trimws(strsplit(
    Sys.getenv("DENGUE_STATIC_MODELS", "INLA-T,INLA-S,INLA-TS"),
    ",", fixed = TRUE
  )[[1L]])
  if ("INLA-T" %in% requested_static) {
  for (model_horizon in 1:3) {
    fit_one("INLA-T", temporal = TRUE, spatial = FALSE,
            forecast_horizon = model_horizon)
  }
  }
  if ("INLA-S" %in% requested_static) {
  for (model_horizon in 1:3) {
    fit_one("INLA-S", temporal = FALSE, spatial = TRUE,
            forecast_horizon = model_horizon)
  }
  }
  if ("INLA-TS" %in% requested_static) {
  for (model_horizon in 1:3) {
    fit_one("INLA-TS", temporal = TRUE, spatial = TRUE,
            forecast_horizon = model_horizon)
  }
  }
}

# Stronger latent spatiotemporal sensitivity models are deliberately opt-in:
# the grouped Besag-by-AR1 field is much larger than the static BYM2 baseline.
if (identical(tolower(Sys.getenv("DENGUE_RUN_STRONG_INLA", "false")), "true")) {
  requested <- strsplit(
    Sys.getenv(
      "DENGUE_STRONG_MODELS",
      "INLA-T-AR1,INLA-ST-AR1,INLA-ST-DYN"
    ), ",", fixed = TRUE
  )[[1L]]
  requested <- trimws(requested)
  # Complete every horizon of the cheaper sensitivity models before starting
  # the much larger dynamic Besag-by-AR1 field. This leaves a usable partial
  # result if the final model is still running at manuscript freeze.
  if ("INLA-T-AR1" %in% requested) {
    for (model_horizon in 1:3) {
      fit_one(
        "INLA-T-AR1", temporal = TRUE, spatial = FALSE,
        forecast_horizon = model_horizon, temporal_random = TRUE
      )
    }
  }
  if ("INLA-ST-AR1" %in% requested) {
    for (model_horizon in 1:3) {
      fit_one(
        "INLA-ST-AR1", temporal = TRUE, spatial = TRUE,
        forecast_horizon = model_horizon, temporal_random = TRUE
      )
    }
  }
  if ("INLA-ST-DYN" %in% requested) {
    for (model_horizon in 1:3) {
      fit_one(
        "INLA-ST-DYN", temporal = TRUE, spatial = TRUE,
        forecast_horizon = model_horizon, temporal_random = TRUE,
        dynamic_spatial = TRUE
      )
    }
  }
}
