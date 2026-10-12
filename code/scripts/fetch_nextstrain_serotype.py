#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import urllib.request
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from build_serotype_features import SEROTYPES, normalise_serotype


METADATA_URL = "https://data.nextstrain.org/files/workflows/dengue/metadata_all.tsv.zst"
PROVINCE_ALIASES = {
    "80": ("tay ninh", "tây ninh"),
    "82": ("dong thap", "đồng tháp", "dongthap"),
    "86": ("vinh long", "vĩnh long"),
    "91": ("an giang",),
    "92": ("can tho", "cần thơ", "cantho"),
    "96": ("ca mau", "cà mau"),
}


def decompress_zst(payload: bytes) -> bytes:
    try:
        import zstandard
    except ImportError as exc:
        raise SystemExit(
            "Python package zstandard is required to read Nextstrain metadata"
        ) from exc
    decompressor = zstandard.ZstdDecompressor()
    try:
        return decompressor.decompress(payload)
    except zstandard.ZstdError:
        with decompressor.stream_reader(io.BytesIO(payload)) as reader:
            return reader.read()


def normalise_place(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def map_province(row: pd.Series, codes: tuple[str, ...]) -> str | None:
    text = " ".join(
        normalise_place(row.get(column, ""))
        for column in ("division", "location", "region", "admin1", "province")
        if column in row.index
    )
    for code in codes:
        aliases = PROVINCE_ALIASES.get(code, ())
        if any(alias in text for alias in aliases):
            return code
    return None


def rolling_shares(samples: pd.DataFrame, window_months: int, minimum_typed: int) -> pd.DataFrame:
    monthly = (
        samples.groupby(["l1_code", "date", "serotype"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SEROTYPES), fill_value=0)
    )
    outputs = []
    for province, part in monthly.groupby(level=0):
        part = part.droplevel(0)
        dates = pd.date_range(part.index.min(), part.index.max(), freq="MS")
        part = part.reindex(dates, fill_value=0)
        counts = part.rolling(window_months, min_periods=1).sum()
        total = counts.sum(axis=1)
        shares = counts.div(total.replace(0, np.nan), axis=0)
        shares.columns = [f"denv{number}_share" for number in SEROTYPES]
        result = shares.reset_index(names="date")
        result.insert(0, "l1_code", str(province))
        result["typed_samples_window"] = total.to_numpy()
        result["serotype_feature_available"] = (
            result["typed_samples_window"] >= minimum_typed
        ).astype(int)
        result["denv2_dominant"] = (
            (result["denv2_share"].fillna(0) >= result[[f"denv{n}_share" for n in SEROTYPES]].max(axis=1))
            & result["denv2_share"].fillna(0).gt(0)
        ).astype(int)
        for column in [f"denv{number}_share" for number in SEROTYPES]:
            result.loc[result["serotype_feature_available"].eq(0), column] = 0.0
            result[column] = result[column].fillna(0.0)
        result.loc[result["serotype_feature_available"].eq(0), "denv2_dominant"] = 0
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path)
    parser.add_argument("--url", default=METADATA_URL)
    parser.add_argument("--window-months", type=int, default=6)
    parser.add_argument("--minimum-typed-samples", type=int, default=3)
    args = parser.parse_args()

    monthly = pd.read_csv(args.monthly, dtype={"fcode": str, "l1_code": str})
    province_codes = tuple(sorted(monthly["l1_code"].astype(str).unique()))
    date_min = pd.Timestamp(year=int(monthly["year"].min()), month=1, day=1)
    date_max = pd.Timestamp(year=int(monthly["year"].max()), month=12, day=1)

    with urllib.request.urlopen(args.url, timeout=120) as response:
        payload = response.read()
    table = pd.read_csv(io.BytesIO(decompress_zst(payload)), sep="\t", dtype=str, low_memory=False)
    country = table.get("country", pd.Series("", index=table.index)).fillna("").str.lower()
    vietnam = table[country.str.contains("viet")].copy()
    if vietnam.empty:
        raise SystemExit("Nextstrain metadata contained no Vietnam rows")

    date_column = "date" if "date" in vietnam.columns else "collection_date"
    serotype_column = "serotype_genbank" if "serotype_genbank" in vietnam.columns else "serotype"
    vietnam["date"] = pd.to_datetime(vietnam[date_column], errors="coerce").dt.to_period("M").dt.to_timestamp()
    vietnam["serotype"] = vietnam[serotype_column].map(normalise_serotype)
    vietnam = vietnam.dropna(subset=["date", "serotype"]).copy()
    vietnam = vietnam[(vietnam["date"] >= date_min) & (vietnam["date"] <= date_max)]
    if vietnam.empty:
        raise SystemExit("No dated Vietnam serotype samples remain after filtering")

    mapped = vietnam.apply(lambda row: map_province(row, province_codes), axis=1)
    vietnam["l1_code"] = mapped
    local_count = int(vietnam["l1_code"].notna().sum())
    use_local = local_count >= 30
    if use_local:
        samples = vietnam.dropna(subset=["l1_code"])[["date", "l1_code", "serotype"]].copy()
    else:
        samples = vietnam[["date", "serotype"]].copy()
        samples["l1_code"] = "VN"

    if args.samples_output is not None:
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(args.samples_output, index=False)

    panel = rolling_shares(samples, args.window_months, args.minimum_typed_samples)
    if not use_local:
        copies = []
        for code in province_codes:
            part = panel.copy()
            part["l1_code"] = code
            copies.append(part)
        panel = pd.concat(copies, ignore_index=True)
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel = panel.drop(columns="date")

    full_index = monthly[["l1_code", "year", "month"]].drop_duplicates()
    merged = full_index.merge(panel, on=["l1_code", "year", "month"], how="left")
    share_columns = [f"denv{number}_share" for number in SEROTYPES]
    merged[share_columns] = merged[share_columns].fillna(0.0)
    merged["typed_samples_window"] = merged["typed_samples_window"].fillna(0.0)
    merged["serotype_feature_available"] = merged["serotype_feature_available"].fillna(0).astype(int)
    merged["denv2_dominant"] = merged["denv2_dominant"].fillna(0).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(
        f"wrote {args.output}; vietnam_samples={len(vietnam)}; "
        f"mapped_local={local_count}; mode={'province' if use_local else 'national'}"
    )


if __name__ == "__main__":
    main()
