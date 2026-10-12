#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(3L, 4L)) {
  stop("Usage: filter_inla_rows_cache.R source_cache.rds qc_monthly.csv output_cache.rds [thresholds.csv]")
}
source_path <- args[[1L]]
monthly_path <- args[[2L]]
output_path <- args[[3L]]
threshold_path <- if (length(args) >= 4L) args[[4L]] else ""
if (!file.exists(source_path)) stop("Missing source cache: ", source_path)

cached <- readRDS(source_path)
if (is.null(cached$rows) || is.null(cached$units)) stop("Invalid INLA rows cache")
monthly <- read.csv(monthly_path, stringsAsFactors = FALSE, check.names = FALSE)
monthly$fcode <- as.character(monthly$fcode)
monthly$date <- as.Date(sprintf("%04d-%02d-01", monthly$year, monthly$month))
retained <- sort(unique(monthly$fcode))

rows <- cached$rows
rows$fcode <- as.character(rows$fcode)
rows$target_date <- as.Date(rows$target_date)
rows <- rows[rows$fcode %in% retained, , drop = FALSE]
units <- sort(intersect(as.character(cached$units), retained))
if (!identical(units, retained)) stop("QC monthly units do not match filtered cache units")

# Validate all cached outcome and exposure values against the materialised QC panel.
monthly_key <- paste(monthly$fcode, monthly$date, sep = "|")
row_key <- paste(rows$fcode, rows$target_date, sep = "|")
positions <- match(row_key, monthly_key)
if (anyNA(positions)) stop("Filtered cache contains target rows absent from QC monthly data")
if (any(abs(rows$target_cases - monthly$obs_dengue_cases[positions]) > 1e-8)) {
  stop("Filtered cache target cases do not match QC monthly data")
}
if (any(abs(rows$population - monthly$population[positions]) > 1e-8)) {
  stop("Filtered cache population does not match QC monthly data")
}

# The cache is only a deterministic feature-design cache, but validate every
# cached case-derived covariate so it cannot silently survive a source update.
monthly$month_index <- monthly$year * 12L + monthly$month
case_lookup <- setNames(
  as.numeric(monthly$obs_dengue_cases),
  paste(monthly$fcode, monthly$month_index, sep = "|")
)
date_month_index <- function(values) {
  values <- as.Date(values)
  as.integer(format(values, "%Y")) * 12L + as.integer(format(values, "%m"))
}
lookup_cases <- function(fcode, month_index) {
  unname(case_lookup[paste(fcode, month_index, sep = "|")])
}
issue_index <- date_month_index(rows$issue_date)
target_index <- date_month_index(rows$target_date)
for (offset in c(0L, 1L, 2L, 5L, 11L)) {
  expected <- log1p(lookup_cases(rows$fcode, issue_index - offset))
  column <- paste0("log_recent_", offset)
  if (anyNA(expected) || any(abs(rows[[column]] - expected) > 1e-8)) {
    stop("Filtered cache mismatch in ", column)
  }
}
for (lag in seq_len(5L)) {
  raw <- lookup_cases(rows$fcode, target_index - 12L * lag)
  available <- !is.na(raw)
  expected_log <- ifelse(available, log1p(raw), 0)
  log_column <- paste0("seasonal_log_", lag)
  available_column <- paste0("seasonal_available_", lag)
  if (any(abs(rows[[log_column]] - expected_log) > 1e-8)) {
    stop("Filtered cache mismatch in ", log_column)
  }
  if (any(rows[[available_column]] != as.numeric(available))) {
    stop("Filtered cache mismatch in ", available_column)
  }
}
if (!identical(sort(unique(rows$fcode)), retained)) {
  stop("Filtered cache is missing retained communes")
}

if (nzchar(threshold_path)) {
  thresholds <- read.csv(threshold_path, stringsAsFactors = FALSE, check.names = FALSE)
  threshold_column <- Sys.getenv("DENGUE_OUTBREAK_THRESHOLD_COLUMN", "epi_threshold")
  required_threshold <- c("fcode", "year", "month", threshold_column)
  missing_threshold <- setdiff(required_threshold, names(thresholds))
  if (length(missing_threshold)) {
    stop("Missing threshold columns: ", paste(missing_threshold, collapse = ", "))
  }
  thresholds$fcode <- as.character(thresholds$fcode)
  thresholds$date <- as.Date(sprintf("%04d-%02d-01", thresholds$year, thresholds$month))
  threshold_key <- paste(thresholds$fcode, thresholds$date, sep = "|")
  if (anyDuplicated(threshold_key)) stop("Duplicate commune-month threshold rows")
  values <- as.numeric(thresholds[[threshold_column]])
  if (anyNA(values) || any(values < 0)) stop("Thresholds must be non-negative numbers")
  threshold_lookup <- setNames(values, threshold_key)
  matched <- unname(threshold_lookup[row_key])
  rows$outbreak_threshold <- ifelse(is.na(matched), Inf, matched)
  evaluation_rows <- rows$split %in% c("validation", "test")
  if (any(!is.finite(rows$outbreak_threshold[evaluation_rows]))) {
    stop("Threshold file does not cover all validation/test target rows")
  }
}

if (tolower(Sys.getenv("DENGUE_RELABEL_SPLITS", "false")) %in% c("1", "true", "yes")) {
  train_start <- as.Date(Sys.getenv("DENGUE_TRAIN_START", "2010-01-01"))
  train_end <- as.Date(Sys.getenv("DENGUE_TRAIN_END", "2020-12-01"))
  validation_start <- as.Date(Sys.getenv("DENGUE_VALIDATION_START", "2021-01-01"))
  validation_end <- as.Date(Sys.getenv("DENGUE_VALIDATION_END", "2022-12-01"))
  test_start <- as.Date(Sys.getenv("DENGUE_TEST_START", "2023-01-01"))
  test_end <- as.Date(Sys.getenv("DENGUE_TEST_END", "2025-12-01"))
  rows$split <- ifelse(
    rows$target_date >= train_start & rows$target_date <= train_end,
    "train",
    ifelse(
      rows$target_date >= validation_start & rows$target_date <= validation_end,
      "validation",
      ifelse(
        rows$target_date >= test_start & rows$target_date <= test_end,
        "test",
        "excluded"
      )
    )
  )
}

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(rows = rows, units = units), output_path)
cat(sprintf("filtered INLA cache: %d units, %d rows\n", length(units), nrow(rows)))
