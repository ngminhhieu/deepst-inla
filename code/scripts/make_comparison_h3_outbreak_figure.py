#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).resolve().parent / "make_final_h3_commune_grid.py"
DEFAULT_COLORS = ("#009E73", "#6B7280", "#0072B2", "#D55E00", "#CC79A7", "#F0E442")


def _load_helper():
    spec = importlib.util.spec_from_file_location("h3_grid", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_model_specs(values: list[str], results_root: Path) -> dict[str, Path]:
    paths = {}
    for value in values:
        if "=" in value:
            name, relative = value.split("=", 1)
            path = Path(relative)
            paths[name] = path if path.is_absolute() else results_root / path
        else:
            paths[value] = results_root / value / "predictions.csv"
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/inla_deep_comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples",
    )
    parser.add_argument(
        "--names-path",
        type=Path,
        default=ROOT.parent
        / "dengue_forecast_v1/dengue_training/data/static/commune_names.csv",
    )
    parser.add_argument("--commune-count", type=int, default=3)
    parser.add_argument("--communes")
    parser.add_argument(
        "--start-date",
        help="Optional inclusive lower bound for target_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        help="Optional inclusive upper bound for target_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "DEEP-ONLY=DEEP-ONLY/predictions.csv",
            "INLA-TS=INLA-ONLY/INLA-TS/predictions.csv",
            "DEEP-INLA=DEEP-INLA/combined/predictions.csv",
        ],
    )
    parser.add_argument("--labels", nargs="+")
    parser.add_argument(
        "--title",
        default="H3 predicted outbreak examples: deep-only vs INLA-only vs deep+INLA",
    )
    args = parser.parse_args()
    helper = _load_helper()
    model_paths = parse_model_specs(args.models, args.results_root)
    display = {
        name: (
            args.labels[index]
            if args.labels is not None and index < len(args.labels)
            else name.replace("-", " ")
        )
        for index, name in enumerate(model_paths)
    }
    colors = {
        name: DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        for index, name in enumerate(model_paths)
    }
    helper.MODELS = tuple(model_paths)
    helper.DISPLAY_NAMES = display
    helper.COLORS = colors
    helper.PREDICTION_PATHS = {key: Path(path.name) for key, path in model_paths.items()}

    predictions = {}
    for model, path in model_paths.items():
        if not path.exists():
            raise SystemExit(f"Required prediction file is missing: {path}")
        frame = pd.read_csv(path, dtype={"fcode": str})
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        frame = frame[frame["horizon"].eq(3)].copy()
        if args.start_date:
            frame = frame[frame["target_date"].ge(pd.Timestamp(args.start_date))]
        if args.end_date:
            frame = frame[frame["target_date"].le(pd.Timestamp(args.end_date))]
        if frame.empty:
            raise SystemExit(
                f"No H3 predictions remain for {model} after applying the date window"
            )
        predictions[model] = frame

    if args.communes:
        communes = [value.strip() for value in args.communes.split(",") if value.strip()]
        selection = helper.describe_communes(predictions, communes)
    else:
        selection = helper.select_communes(predictions, args.commune_count)
    names = helper.load_names(args.names_path)
    selection["region_name"] = selection["fcode"].map(
        lambda value: names.get(str(value), (str(value), ""))[0]
    )
    selection["province_name"] = selection["fcode"].map(
        lambda value: names.get(str(value), (str(value), ""))[1]
    )
    selection_path = args.output.with_name(args.output.name + "_selection.csv")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)
    helper.plot_grid(
        predictions,
        selection,
        names,
        helper.overall_outbreak_metrics(predictions),
        args.output,
        args.title,
    )
    print(f"wrote {args.output.with_suffix('.png')}")
    print(selection[["selection_rank", "fcode", "region_name", "province_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
