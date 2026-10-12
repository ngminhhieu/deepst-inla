#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: filter_operational_h3_rows.R input.rds output.rds")
cached <- readRDS(args[[1L]])
rows <- cached$rows
keep <- rows$horizon == 3L & rows$split %in% c("train", "validation", "test")
rows <- rows[keep, , drop = FALSE]
if (!any(rows$split == "test")) stop("No operational H3 test rows")
saveRDS(list(rows = rows, units = cached$units), args[[2L]])
cat(sprintf("rows=%d train=%d validation=%d test=%d\n",
  nrow(rows), sum(rows$split == "train"), sum(rows$split == "validation"),
  sum(rows$split == "test")))
