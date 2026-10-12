#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: make_rolling_refit_rows_cache.R source_cache.rds issue_date output.rds")
}
source <- readRDS(args[[1L]])
issue <- as.Date(args[[2L]])
rows <- source$rows
rows$issue_date <- as.Date(rows$issue_date)
rows$target_date <- as.Date(rows$target_date)
# Match the multi-horizon deep training mask: an origin enters training only
# after its H3 target has been observed. The current issue is evaluation only.
last_training_issue <- seq(issue, by = "-1 month", length.out = 4L)[4L]
keep <- (rows$issue_date <= last_training_issue) | (rows$issue_date == issue)
rows <- rows[keep, , drop = FALSE]
rows$split <- ifelse(rows$issue_date == issue, "test", "train")
if (!any(rows$split == "train") || sum(rows$split == "test") != length(source$units) * 3L) {
  stop("Invalid rolling train/test row selection for ", issue)
}
dir.create(dirname(args[[3L]]), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(rows = rows, units = source$units), args[[3L]])
message("Rolling cache ", issue, ": train=", sum(rows$split == "train"),
        ", test=", sum(rows$split == "test"))
