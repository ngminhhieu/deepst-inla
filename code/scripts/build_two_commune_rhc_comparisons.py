#!/usr/bin/env python3
"""Build commune-vs-parent-RHC alert comparisons under two RHC definitions.

The commune side is fixed: proposed hybrid H3 forecasts classified with the
commune MOH definition.  The RHC side is either:

1. original-model H3 forecasts re-scored with the RHC MOH definition; or
2. the operational season-1 rule (all three methods agree and forecast mean
   incidence is greater than 20 cases per 100,000), represented by the final
   positive-alert list in
   ``code/data/reference/operational_2026/rhc_three_method_alerts_2026.xlsx``.

Only Python's standard library is used so this audit step is portable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


MONTH_NUMBER = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
MAIN_COLUMNS = [
    "forecast_month", "commune_id", "commune_name", "province",
    "rhc_id", "rhc_name", "commune_prediction", "commune_alert",
    "groundtruth", "rhc_prediction", "rhc_alert", "comparison_category",
]

# RHC identifiers are ASCII slugs, but display names must retain Vietnamese
# diacritics.  Keeping this lookup here also prevents future output rebuilds
# from reintroducing unaccented names.
RHC_AREA_NAMES = {
    "AN_BIEN": "An Biên", "AN_MINH": "An Minh", "AN_PHU": "An Phú",
    "BAC_LIEU": "Bạc Liêu", "BA_TRI": "Ba Tri", "BEN_LUC": "Bến Lức",
    "BEN_TRE": "Bến Tre", "BINH_DAI": "Bình Đại", "BINH_MINH": "Bình Minh",
    "BINH_PHU": "Bình Phú", "BINH_TAN": "Bình Tân", "BINH_THUY": "Bình Thủy",
    "CAI_BE": "Cái Bè", "CAI_LAY": "Cai Lậy", "CAI_NUOC": "Cái Nước",
    "CAI_RANG": "Cái Răng", "CANG_LONG": "Càng Long", "CAN_DUOC": "Cần Đước",
    "CAN_GIUOC": "Cần Giuộc", "CAO_LANH_1": "Cao Lãnh 1",
    "CAO_LANH_2": "Cao Lãnh 2", "CAU_KE": "Cầu Kè", "CAU_NGANG": "Cầu Ngang",
    "CA_MAU": "Cà Mau", "CHAU_DOC": "Châu Đốc", "CHAU_PHU": "Châu Phú",
    "CHAU_THANH": "Châu Thành", "CHAU_THANH_A": "Châu Thành A",
    "CHO_GAO": "Chợ Gạo", "CHO_LACH": "Chợ Lách", "CHO_MOI": "Chợ Mới",
    "CU_LAO_DUNG": "Cù Lao Dung", "DAM_DOI": "Đầm Dơi", "DONG_HAI": "Đông Hải",
    "DUC_HOA": "Đức Hòa", "DUC_HUE": "Đức Huệ", "DUYEN_HAI": "Duyên Hải",
    "GIANG_THANH": "Giang Thành", "GIA_RAI": "Giá Rai",
    "GIONG_RIENG": "Giồng Riềng", "GIONG_TROM": "Giồng Trôm",
    "GO_CONG_DONG": "Gò Công Đông", "GO_CONG_TAY": "Gò Công Tây",
    "GO_CONG": "Gò Công", "GO_QUAO": "Gò Quao", "HA_TIEN": "Hà Tiên",
    "HOA_BINH": "Hòa Bình", "HONG_DAN": "Hồng Dân", "HONG_NGU_1": "Hồng Ngự 1",
    "HONG_NGU_2": "Hồng Ngự 2", "HON_DAT": "Hòn Đất", "KE_SACH": "Kế Sách",
    "KIEN_HAI": "Kiên Hải", "KIEN_LUONG": "Kiên Lương", "KIEN_TUONG": "Kiến Tường",
    "LAI_VUNG": "Lai Vung", "LAP_VO": "Lấp Vò", "LONG_HO": "Long Hồ",
    "LONG_MY": "Long Mỹ", "LONG_PHU": "Long Phú", "LONG_THANH": "Long Thành",
    "LONG_XUYEN": "Long Xuyên", "MANG_THIT": "Mang Thít", "MOC_HOA": "Mộc Hóa",
    "MO_CAY_BAC": "Mỏ Cày Bắc", "MO_CAY_NAM": "Mỏ Cày Nam", "MY_THO": "Mỹ Tho",
    "MY_TU": "Mỹ Tú", "MY_XUYEN": "Mỹ Xuyên", "NAM_CAN": "Năm Căn",
    "NGA_NAM": "Ngã Năm", "NGOC_HIEN": "Ngọc Hiển",
    "NGUYEN_VAN_THU": "Nguyễn Văn Thủ", "O_MON": "Ô Môn",
    "PHUNG_HIEP": "Phụng Hiệp", "PHUOC_LONG": "Phước Long", "PHU_QUOC": "Phú Quốc",
    "PHU_TAN": "Phú Tân", "RACH_GIA": "Rạch Giá", "SA_DEC_1": "Sa Đéc 1",
    "SA_DEC_2": "Sa Đéc 2", "TAM_BINH": "Tam Bình", "TAM_NONG": "Tam Nông",
    "TAM_VU": "Tầm Vu", "TAN_AN": "Tân An", "TAN_CHAU": "Tân Châu",
    "TAN_HIEP": "Tân Hiệp", "TAN_HONG": "Tân Hồng", "TAN_HUNG": "Tân Hưng",
    "TAN_PHUOC": "Tân Phước", "TAN_PHU_DONG": "Tân Phú Đông",
    "TAN_THANH": "Tân Thạnh", "TAN_TRU": "Tân Trụ", "THANH_BINH": "Thanh Bình",
    "THANH_HOA": "Thạnh Hóa", "THANH_PHO_SOC_TRANG": "Thành phố Sóc Trăng",
    "THANH_PHU": "Thạnh Phú", "THANH_TRI": "Thạnh Trị", "THAP_MUOI": "Tháp Mười",
    "THOAI_SON": "Thoại Sơn", "THOI_BINH": "Thới Bình", "THOI_LAI": "Thới Lai",
    "THOT_NOT": "Thốt Nốt", "THU_THUA": "Thủ Thừa", "TIEU_CAN": "Tiểu Cần",
    "TINH_BIEN": "Tịnh Biên", "TRAN_DE": "Trần Đề", "TRAN_VAN_THOI": "Trần Văn Thời",
    "TRA_CU": "Trà Cú", "TRA_ON": "Trà Ôn", "TRA_VINH": "Trà Vinh",
    "TRI_TON": "Tri Tôn", "U_MINH": "U Minh", "U_MINH_THUONG": "U Minh Thượng",
    "VINH_CHAU": "Vĩnh Châu", "VINH_HUNG": "Vĩnh Hưng", "VINH_LOI": "Vĩnh Lợi",
    "VINH_LONG": "Vĩnh Long", "VINH_THANH": "Vĩnh Thạnh", "VINH_THUAN": "Vĩnh Thuận",
    "VI_THUY": "Vị Thủy",
}


def rhc_display_name(rhc_id: str) -> str:
    parts = re.sub(r"^TTYTKV_", "", rhc_id).split("_")
    if len(parts) > 1 and len(parts[-1]) <= 3:
        parts = parts[:-1]
    area_key = "_".join(parts)
    if area_key not in RHC_AREA_NAMES:
        raise ValueError(f"Missing Vietnamese display name for RHC: {rhc_id}")
    return f"TTYT khu vực {RHC_AREA_NAMES[area_key]}"
SUMMARY_COLUMNS = [
    "rhc_alert_definition", "forecast_month", "total_commune_month_rows",
    "proposed_commune_alerts", "alerted_parent_rhc_months",
    "communes_whose_parent_rhc_alerted",
    "both_commune_and_parent_rhc_alerted",
    "commune_alerted_parent_rhc_not_alerted",
    "commune_not_alerted_parent_rhc_alerted", "neither_alerted",
    "parent_rhc_alert_unavailable", "match_rate_among_proposed_commune_alerts",
]
SUMMARY_OUTPUT_COLUMNS = ["record_type", *SUMMARY_COLUMNS, "note_column", "note_meaning"]
SUMMARY_COLUMN_NOTES = {
    "record_type": "Loại dòng: data là kết quả thống kê; column_note là dòng giải thích cột.",
    "rhc_alert_definition": "Cách xác định alert của original RHC model: MOH hoặc đồng thuận 3 phương pháp kèm ngưỡng >20 ca/100.000.",
    "forecast_month": "Tháng được dự báo; ALL là tổng gộp từ tháng 04 đến tháng 11/2026.",
    "total_commune_month_rows": "Tổng số đơn vị commune-tháng trong dòng thống kê.",
    "proposed_commune_alerts": "Số commune-tháng được proposed hybrid commune model cảnh báo theo MOH threshold.",
    "alerted_parent_rhc_months": "Số cặp tháng-RHC cha được original RHC model cảnh báo.",
    "communes_whose_parent_rhc_alerted": "Số commune-tháng nằm trong một RHC cha được cảnh báo cùng tháng; một RHC có thể chứa nhiều commune.",
    "both_commune_and_parent_rhc_alerted": "Số commune-tháng mà cả proposed commune model và RHC cha đều cảnh báo; đây là số match.",
    "commune_alerted_parent_rhc_not_alerted": "Số commune-tháng được proposed commune model cảnh báo nhưng RHC cha không cảnh báo.",
    "commune_not_alerted_parent_rhc_alerted": "Số commune-tháng không được proposed commune model cảnh báo nhưng RHC cha được cảnh báo.",
    "neither_alerted": "Số commune-tháng mà cả commune model và RHC cha đều không cảnh báo.",
    "parent_rhc_alert_unavailable": "Số commune-tháng không có thông tin alert của RHC cha; bằng 0 trong kết quả hiện tại.",
    "match_rate_among_proposed_commune_alerts": "both_commune_and_parent_rhc_alerted chia cho proposed_commune_alerts.",
    "note_column": "Tên cột đang được giải thích trên các dòng column_note.",
    "note_meaning": "Ý nghĩa của cột tương ứng trong note_column.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet of a small xlsx using its XML representation."""
    spreadsheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns = {"m": spreadsheet}
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(f"{{{spreadsheet}}}t"))
            for item in shared_root.findall("m:si", ns)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    matrix: list[dict[str, str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", ns):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group(0)
            value_node = cell.find("m:v", ns)
            value = "" if value_node is None else (value_node.text or "")
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[column] = value
        matrix.append(values)
    if not matrix:
        return []
    headers = {column: value.strip() for column, value in matrix[0].items()}
    return [
        {headers[column]: values.get(column, "").strip() for column in headers}
        for values in matrix[1:]
        if any(values.get(column, "").strip() for column in headers)
    ]


def operational_alert_keys(path: Path) -> tuple[set[tuple[str, str]], Counter]:
    rows = xlsx_rows(path)
    keys: set[tuple[str, str]] = set()
    counts: Counter = Counter()
    for row in rows:
        rhc_id = row.get("selected_fcode", "").strip()
        month_text = row.get("month", "").strip().lower()
        month_number = row.get("monthN", "").strip()
        if not rhc_id:
            continue
        if month_number:
            month = int(float(month_number))
        elif month_text in MONTH_NUMBER:
            month = MONTH_NUMBER[month_text]
        else:
            raise ValueError(f"Cannot identify forecast month in workbook row: {row}")
        key = (f"2026-{month:02d}", rhc_id)
        if key in keys:
            raise ValueError(f"Duplicate operational alert: {key}")
        keys.add(key)
        counts[key[0]] += 1
    expected_months = {f"2026-{month:02d}" for month in range(4, 12)}
    if set(counts) != expected_months:
        raise ValueError(f"Operational workbook months are {sorted(counts)}")
    return keys, counts


def category(commune_alert: int, rhc_alert: int) -> str:
    if commune_alert and rhc_alert:
        return "both_alerted"
    if commune_alert:
        return "commune_only"
    if rhc_alert:
        return "rhc_only"
    return "neither_alerted"


def write_comparison(
    path: Path,
    commune_rows: list[dict[str, str]],
    rhc_rows: dict[tuple[str, str], dict[str, str]],
    alert_field: str,
) -> Counter:
    counts: Counter = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM lets Excel detect Vietnamese text correctly on direct open.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAIN_COLUMNS)
        writer.writeheader()
        for source in commune_rows:
            key = (source["forecast_month"], source["rhc_id"])
            rhc = rhc_rows.get(key)
            commune_alert = int(source["commune_alert"])
            if rhc is None or rhc.get(alert_field, "") in ("", "NA"):
                rhc_alert = "NA"
                comparison = "rhc_alert_unavailable"
                prediction = "NA"
            else:
                rhc_alert_int = int(float(rhc[alert_field]))
                rhc_alert = str(rhc_alert_int)
                comparison = category(commune_alert, rhc_alert_int)
                prediction = rhc.get("original_prediction", "NA") or "NA"
            row = {
                "forecast_month": source["forecast_month"],
                "commune_id": source["commune_id"],
                "commune_name": source["commune_name"],
                "province": source["province"],
                "rhc_id": source["rhc_id"],
                "rhc_name": rhc_display_name(source["rhc_id"]),
                "commune_prediction": source["commune_prediction"],
                "commune_alert": source["commune_alert"],
                "groundtruth": source["groundtruth"],
                "rhc_prediction": prediction,
                "rhc_alert": rhc_alert,
                "comparison_category": comparison,
            }
            writer.writerow(row)
            counts[(source["forecast_month"], comparison)] += 1
    return counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer repository-relative paths in committed manifests."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def summarise_comparison(
    definition: str, rows: list[dict[str, str]], period: str
) -> dict[str, object]:
    commune_alerts = sum(int(row["commune_alert"]) for row in rows)
    available = [row for row in rows if row["rhc_alert"] != "NA"]
    alerted = [row for row in available if int(row["rhc_alert"]) == 1]
    matched = sum(
        int(row["commune_alert"] == "1" and row["rhc_alert"] == "1")
        for row in available
    )
    categories = Counter(row["comparison_category"] for row in rows)
    return {
        "rhc_alert_definition": definition,
        "forecast_month": period,
        "total_commune_month_rows": len(rows),
        "proposed_commune_alerts": commune_alerts,
        "alerted_parent_rhc_months": len({
            (row["forecast_month"], row["rhc_id"]) for row in alerted
        }),
        "communes_whose_parent_rhc_alerted": len(alerted),
        "both_commune_and_parent_rhc_alerted": matched,
        "commune_alerted_parent_rhc_not_alerted": categories["commune_only"],
        "commune_not_alerted_parent_rhc_alerted": categories["rhc_only"],
        "neither_alerted": categories["neither_alerted"],
        "parent_rhc_alert_unavailable": categories["rhc_alert_unavailable"],
        "match_rate_among_proposed_commune_alerts": (
            f"{matched / commune_alerts:.6f}" if commune_alerts else "NA"
        ),
    }


def write_summary(path: Path, sources: list[tuple[str, Path]]) -> None:
    summary_rows: list[dict[str, object]] = []
    for definition, source in sources:
        rows = read_csv(source)
        months = sorted({row["forecast_month"] for row in rows})
        for month in months:
            summary_rows.append(summarise_comparison(
                definition, [row for row in rows if row["forecast_month"] == month], month
            ))
        summary_rows.append(summarise_comparison(definition, rows, "ALL"))
    # UTF-8 BOM lets Excel detect Vietnamese note rows correctly on direct open.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows({"record_type": "data", **row} for row in summary_rows)
        writer.writerows(
            {
                "record_type": "column_note",
                "note_column": column,
                "note_meaning": SUMMARY_COLUMN_NOTES[column],
            }
            for column in SUMMARY_OUTPUT_COLUMNS
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commune", type=Path, required=True)
    parser.add_argument("--rhc-moh", type=Path, required=True)
    parser.add_argument("--operational-alerts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    commune_rows = read_csv(args.commune)
    moh_rows_list = read_csv(args.rhc_moh)
    moh_rows = {(row["forecast_month"], row["rhc_id"]): row for row in moh_rows_list}
    if len(moh_rows) != len(moh_rows_list):
        raise ValueError("Duplicate forecast_month x RHC rows in MOH source")

    operational_keys, operational_positive_counts = operational_alert_keys(
        args.operational_alerts
    )
    # The workbook is the final positive-alert list.  Add its indicator to the
    # complete RHC-month frame so absence means a negative alert, not missing.
    operational_rows: dict[tuple[str, str], dict[str, str]] = {}
    for key, row in moh_rows.items():
        enriched = dict(row)
        enriched["operational_three_of_three_alert"] = str(int(key in operational_keys))
        operational_rows[key] = enriched

    expected = len({row["commune_id"] for row in commune_rows}) * 8
    if len(commune_rows) != expected:
        raise ValueError(f"Expected 8 rows per commune ({expected}), found {len(commune_rows)}")
    missing_rhc = sorted({
        (row["forecast_month"], row["rhc_id"])
        for row in commune_rows
        if (row["forecast_month"], row["rhc_id"]) not in moh_rows
    })
    if missing_rhc:
        raise ValueError(f"Missing RHC source rows; examples: {missing_rhc[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    moh_path = args.output_dir / "commune_vs_rhc_moh_2026-04_to_11.csv"
    operational_path = args.output_dir / "commune_vs_rhc_three_methods_2026-04_to_11.csv"
    moh_counts = write_comparison(
        moh_path, commune_rows, moh_rows, "original_moh_outbreak_alert",
    )
    operational_counts = write_comparison(
        operational_path, commune_rows, operational_rows,
        "operational_three_of_three_alert",
    )
    summary_path = args.output_dir / "commune_vs_rhc_comparison_summary_2026-04_to_11.csv"
    write_summary(summary_path, [
        ("moh", moh_path),
        ("three_methods_consensus_plus_20_per_100k", operational_path),
    ])

    def serialise_counts(counts: Counter) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for (month, label), value in sorted(counts.items()):
            result.setdefault(month, {})[label] = value
        return result

    manifest = {
        "unit": "one row per forecast_month x commune",
        "months": [f"2026-{month:02d}" for month in range(4, 12)],
        "communes": len({row["commune_id"] for row in commune_rows}),
        "rows_per_file": len(commune_rows),
        "commune_definition_both_files": "Proposed hybrid H3; commune MOH outbreak alert",
        "match_rule": "comparison_category=both_alerted only when commune_alert=1 and its parent RHC alert=1 in the same month",
        "removed_redundant_columns": {
            "commune_rhc_match": "Fully determined by commune_alert, rhc_alert, and comparison_category",
            "rhc_alert_definition": "Constant within each file; the RHC definition is encoded in the filename",
        },
        "moh_rhc_source": portable_path(args.rhc_moh),
        "three_method_rhc_source": portable_path(args.operational_alerts),
        "three_method_rule": "epidemic_flag_risk=1 AND epidemic_flag_z_score=1 AND epidemic_flag_pred_mean=1 AND predicted mean incidence >20/100,000",
        "operational_positive_rhcs_by_month": dict(sorted(operational_positive_counts.items())),
        "comparison_counts": {
            "moh": serialise_counts(moh_counts),
            "three_methods": serialise_counts(operational_counts),
        },
        "sha256": {
            moh_path.name: sha256(moh_path),
            operational_path.name: sha256(operational_path),
            summary_path.name: sha256(summary_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
