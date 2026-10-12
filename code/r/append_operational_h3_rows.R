#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: append_operational_h3_rows.R cache.rds test.csv output.rds")
cached <- readRDS(args[[1L]])
history <- cached$rows[cached$rows$horizon == 3L & cached$rows$split %in% c("train", "validation"), ]
test <- read.csv(args[[2L]], stringsAsFactors = FALSE, check.names = FALSE)
missing <- setdiff(names(history), names(test))
if (length(missing)) stop("Missing test columns: ", paste(missing, collapse = ", "))
test <- test[, names(history), drop = FALSE]
rows <- rbind(history, test)
rows$fcode <- as.character(rows$fcode)
rows$issue_date <- as.Date(rows$issue_date)
rows$target_date <- as.Date(rows$target_date)
saveRDS(list(rows = rows, units = cached$units), args[[3L]])
cat(sprintf("rows=%d train=%d validation=%d test=%d\n",
  nrow(rows), sum(rows$split == "train"), sum(rows$split == "validation"),
  sum(rows$split == "test")))
