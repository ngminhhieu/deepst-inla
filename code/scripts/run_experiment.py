#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dengue_h3.config import load_config
from dengue_h3.pipeline import run_experiment
from dengue_h3.specs import MODEL_SPECS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study_lognormal.yaml")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--output-name")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Output root; defaults to the repository results directory.",
    )
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--dilations", type=int, nargs="+")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--peak-loss-weight", type=float)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument(
        "--refit-on-train-validation",
        choices=("true", "false"),
        help="Override whether the final test model is refitted on train plus validation.",
    )
    parser.add_argument("--monthly-path")
    parser.add_argument("--climate-columns", nargs="+")
    parser.add_argument("--history-covariate-columns", nargs="+")
    parser.add_argument("--future-covariates-path")
    parser.add_argument("--future-covariate-columns", nargs="+")
    parser.add_argument("--outbreak-thresholds-path")
    parser.add_argument("--outbreak-threshold-column")
    parser.add_argument("--likelihood", choices=["lognormal", "negative_binomial"])
    parser.add_argument("--outbreak-loss-weight", type=float)
    parser.add_argument("--outbreak-positive-weight")
    parser.add_argument("--outbreak-focal-gamma", type=float)
    parser.add_argument("--alert-probability-source", choices=["samples", "auxiliary"])
    parser.add_argument("--draws", type=int)
    parser.add_argument(
        "--export-representations",
        action="store_true",
        help="Export train, validation, and test latent features for INLA.",
    )
    parser.add_argument(
        "--export-validation-predictions",
        action="store_true",
        help="Save validation forecasts and predictive samples for stacking.",
    )
    args = parser.parse_args()
    root, config = load_config(args.config)
    if args.hidden_size is not None:
        config["model"]["hidden_size"] = args.hidden_size
    if args.sequence_length is not None:
        config["forecast"]["sequence_length"] = args.sequence_length
    if args.dilations is not None:
        config["model"]["dilations"] = args.dilations
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.peak_loss_weight is not None:
        config["model"]["peak_loss_weight"] = args.peak_loss_weight
    if args.training_seed is not None:
        config["training"]["seed"] = args.training_seed
    if args.refit_on_train_validation is not None:
        config["training"]["refit_on_train_validation"] = (
            args.refit_on_train_validation == "true"
        )
    if args.monthly_path is not None:
        config["data"]["monthly_path"] = args.monthly_path
    if args.climate_columns is not None:
        config["data"]["climate_columns"] = args.climate_columns
    if args.history_covariate_columns is not None:
        config["data"]["history_covariate_columns"] = args.history_covariate_columns
    if args.future_covariates_path is not None:
        config["data"]["future_covariates_path"] = args.future_covariates_path
    if args.future_covariate_columns is not None:
        config["data"]["future_covariate_columns"] = args.future_covariate_columns
    if args.outbreak_thresholds_path is not None:
        config["data"]["outbreak_thresholds_path"] = args.outbreak_thresholds_path
    if args.outbreak_threshold_column is not None:
        config["data"]["outbreak_threshold_column"] = args.outbreak_threshold_column
    if args.likelihood is not None:
        config["model"]["likelihood"] = args.likelihood
    if args.outbreak_loss_weight is not None:
        config["model"]["outbreak_loss_weight"] = args.outbreak_loss_weight
    if args.outbreak_positive_weight is not None:
        config["model"]["outbreak_positive_weight"] = (
            "auto"
            if args.outbreak_positive_weight.lower() == "auto"
            else float(args.outbreak_positive_weight)
        )
    if args.outbreak_focal_gamma is not None:
        config["model"]["outbreak_focal_gamma"] = args.outbreak_focal_gamma
    if args.alert_probability_source is not None:
        config["model"]["alert_probability_source"] = args.alert_probability_source
    if args.draws is not None:
        config["sampling"]["draws"] = args.draws
    output = run_experiment(
        root,
        config,
        args.model,
        args.output_name,
        results_root=args.results_root,
        export_representations=True if args.export_representations else None,
        export_validation_predictions=args.export_validation_predictions,
    )
    print(output)


if __name__ == "__main__":
    main()
