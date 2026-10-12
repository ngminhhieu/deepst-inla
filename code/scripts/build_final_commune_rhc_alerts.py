#!/usr/bin/env python3
"""Create the final commune x RHC alert comparison from H3 forecasts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "forecast_month", "commune_id", "commune_name", "province", "rhc_id", "rhc_name",
    "commune_prediction", "commune_alert", "groundtruth", "rhc_alert", "commune_rhc_match",
    "comparison_category",
]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1


def excel_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime(1899, 12, 30) + dt.timedelta(days=float(value)))


def read_alert_workbook(path: Path) -> pd.DataFrame:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{main_ns}}}si"):
                strings.append("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t")))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relations}
        sheet = workbook.find(f".//{{{main_ns}}}sheet")
        target = targets[sheet.attrib[f"{{{rel_ns}}}id"]]
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        sheet_xml = ET.fromstring(archive.read(target))
        raw_rows = []
        for row in sheet_xml.findall(f".//{{{main_ns}}}row"):
            values = {}
            for cell in row.findall(f"{{{main_ns}}}c"):
                node = cell.find(f"{{{main_ns}}}v")
                value = "" if node is None else node.text
                if cell.attrib.get("t") == "s" and value:
                    value = strings[int(value)]
                values[column_index(cell.attrib["r"])] = value
            raw_rows.append([values.get(index, "") for index in range(13)])
    header = raw_rows[0]
    rows = []
    for values in raw_rows[1:]:
        row = dict(zip(header, values))
        if not row.get("selected_fcode"):
            continue
        rows.append({
            "target_date": excel_date(row["selected_date"]).to_period("M").to_timestamp(),
            "rhc_id": str(row["selected_fcode"]).strip(),
            "group": row.get("group", ""),
        })
    output = pd.DataFrame(rows).drop_duplicates(["target_date", "rhc_id"])
    if output.duplicated(["target_date", "rhc_id"]).any():
        raise ValueError("Duplicate RHC-month alerts in workbook")
    return output


def rhc_name(value: str) -> str:
    parts = re.sub(r"^TTYTKV_", "", value).split("_")
    if len(parts) > 1 and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return "TTYT khu vực " + " ".join(parts).title()


def category(commune: int, rhc: object) -> str:
    if pd.isna(rhc):
        return "rhc_alert_unavailable"
    if commune == 1 and int(rhc) == 1:
        return "both_alerted"
    if commune == 1:
        return "commune_only"
    if int(rhc) == 1:
        return "rhc_only"
    return "neither_alerted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--commune-observations", type=Path, required=True)
    parser.add_argument("--rhc-workbook", type=Path, required=True)
    parser.add_argument("--rhc-input", type=Path, required=True)
    parser.add_argument("--mongo-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-cutoff", type=float, default=0.50)
    args = parser.parse_args()
    if "ytkv" in args.thresholds.name.lower():
        raise ValueError("The commune alert pipeline cannot use a YTKV threshold file")

    forecast = pd.read_csv(args.predictions, dtype={"fcode": str})
    samples = np.load(args.samples)["samples"]
    if len(forecast) != len(samples):
        raise ValueError("Prediction/sample mismatch")
    forecast["target_date"] = pd.to_datetime(forecast["target_date"]).dt.to_period("M").dt.to_timestamp()
    if not forecast["horizon"].eq(3).all():
        raise ValueError("Final operational output must contain H3 only")
    forecast["commune_prediction"] = samples.mean(axis=1)

    thresholds = pd.read_csv(args.thresholds, dtype={"fcode": str})
    thresholds["target_date"] = pd.to_datetime(dict(
        year=thresholds.year, month=thresholds.month, day=1
    ))
    thresholds = thresholds[[
        "fcode", "target_date", "alert_threshold_moh", "epi_threshold_moh"
    ]]
    forecast = forecast.merge(thresholds, on=["fcode", "target_date"], how="left", validate="one_to_one")
    if forecast["alert_threshold_moh"].isna().any():
        raise ValueError("Missing commune MOH thresholds")
    probability = np.mean(samples >= forecast["alert_threshold_moh"].to_numpy(float)[:, None], axis=1)
    forecast["commune_alert"] = pd.array(probability >= args.probability_cutoff, dtype="Int64")
    observed = pd.read_csv(
        args.commune_observations,
        usecols=["fcode", "year", "month", "obs_dengue_cases"],
        dtype={"fcode": str},
    )
    observed["target_date"] = pd.to_datetime(dict(
        year=observed.year, month=observed.month, day=1
    ))
    observed = observed[["fcode", "target_date", "obs_dengue_cases"]]
    forecast = forecast.merge(
        observed, on=["fcode", "target_date"], how="left", validate="one_to_one"
    )
    forecast["groundtruth"] = pd.array([
        pd.NA if pd.isna(cases) else int(float(cases) > float(threshold))
        for cases, threshold in zip(
            forecast["obs_dengue_cases"], forecast["epi_threshold_moh"]
        )
    ], dtype="Int64")

    mongo = json.loads(args.mongo_export.read_text(encoding="utf-8"))
    mapping_rows = []
    for row in mongo["boundary"]:
        props = row.get("properties") or {}
        mapping_rows.append({
            "fcode": str(row.get("fcode")), "commune_name": props.get("l2_name_loc"),
            "province": props.get("l1_name_loc"), "rhc_id": props.get("ttytkv"),
        })
    mapping = pd.DataFrame(mapping_rows).drop_duplicates("fcode")
    mapping["rhc_name"] = mapping["rhc_id"].fillna("").map(rhc_name)
    forecast = forecast.merge(mapping, on="fcode", how="left", validate="many_to_one")

    rhc_universe = set(pd.read_csv(args.rhc_input, usecols=["fcode"], dtype=str)["fcode"].dropna())
    workbook = read_alert_workbook(args.rhc_workbook)
    alerted = set(zip(workbook["target_date"], workbook["rhc_id"]))
    rhc_values = []
    for row in forecast.itertuples(index=False):
        if not row.rhc_id or row.rhc_id not in rhc_universe:
            rhc_values.append(pd.NA)
        else:
            rhc_values.append(int((row.target_date, row.rhc_id) in alerted))
    forecast["rhc_alert"] = pd.array(rhc_values, dtype="Int64")
    forecast["commune_rhc_match"] = pd.array([
        pd.NA if pd.isna(rhc) else int(int(commune) == 1 and int(rhc) == 1)
        for commune, rhc in zip(forecast["commune_alert"], forecast["rhc_alert"])
    ], dtype="Int64")
    forecast["comparison_category"] = [
        category(int(commune), rhc)
        for commune, rhc in zip(forecast["commune_alert"], forecast["rhc_alert"])
    ]
    forecast["forecast_month"] = forecast["target_date"].dt.strftime("%Y-%m")
    forecast = forecast.rename(columns={"fcode": "commune_id"})
    output = forecast[OUTPUT_COLUMNS].sort_values(["forecast_month", "commune_id"])
    if output.duplicated(["forecast_month", "commune_id"]).any():
        raise ValueError("Output is not unique by forecast_month x commune")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, na_rep="NA")

    manifest = {
        "rows": len(output), "communes": int(output.commune_id.nunique()),
        "forecast_months": sorted(output.forecast_month.unique().tolist()),
        "forecast_horizon": 3,
        "issue_to_target_rule": "target month = issue month + 3 months",
        "commune_threshold_file": portable_path(args.thresholds),
        "rhc_threshold_file_used": False,
        "rhc_alert_source": portable_path(args.rhc_workbook),
        "rhc_workbook_alert_rows": len(workbook),
        "rhc_alert_semantics": "1 if RHC-month appears in combined alert workbook; 0 if absent but RHC is covered by the 128-unit input; NA otherwise",
        "commune_alert_rule": f"P(cases >= alert_threshold_moh) >= {args.probability_cutoff:.2f}",
        "groundtruth_rule": "observed cases > epi_threshold_moh (MOH mean + 2 SD); NA when observations are unavailable",
        "groundtruth_available_rows": int(output.groundtruth.notna().sum()),
        "groundtruth_unavailable_rows": int(output.groundtruth.isna().sum()),
        "rhc_alert_available_rows": int(output.rhc_alert.notna().sum()),
        "rhc_alert_unavailable_rows": int(output.rhc_alert.isna().sum()),
        "comparison_category_counts": {
            str(key): int(value) for key, value in output.comparison_category.value_counts(dropna=False).items()
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
