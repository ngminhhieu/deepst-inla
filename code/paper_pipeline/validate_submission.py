#!/usr/bin/env python3
"""Fail loudly when the Nature Communications package is not submission-ready."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def plain_words(value: str) -> list[str]:
    value = re.sub(r"\\[A-Za-z*]+(?:\[[^]]*\])?", " ", value)
    value = value.replace("--", "-")
    value = re.sub(r"[{}$~]", " ", value)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", value)


def prose_only(value: str) -> str:
    """Remove display material before approximate journal word counts."""
    value = re.sub(r"(?m)(?<!\\)%.*$", " ", value)
    for environment in ("figure", "table", "equation", "align", "longtable"):
        value = re.sub(
            rf"\\begin\{{{environment}\*?\}}.*?\\end\{{{environment}\*?\}}",
            " ", value, flags=re.S,
        )
    value = re.sub(r"\\(?:section|subsection)\*?\{.*?\}", " ", value, flags=re.S)
    value = re.sub(r"\\(?:input|includegraphics|label|ref|url)\{.*?\}", " ", value, flags=re.S)
    value = re.sub(r"\\cite\{.*?\}", " ", value, flags=re.S)
    return value


def braced_arguments(value: str, command: str) -> list[str]:
    """Extract balanced braced arguments such as figure captions."""
    out: list[str] = []
    needle = rf"\{command}"
    cursor = 0
    while True:
        start = value.find(needle, cursor)
        if start < 0:
            return out
        brace = value.find("{", start + len(needle))
        if brace < 0:
            return out
        depth = 0
        for index in range(brace, len(value)):
            if value[index] == "{" and (index == 0 or value[index - 1] != "\\"):
                depth += 1
            elif value[index] == "}" and (index == 0 or value[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    out.append(value[brace + 1:index])
                    cursor = index + 1
                    break
        else:
            return out


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    tex_files = sorted(PAPER.rglob("*.tex")) + sorted((ROOT / "metadata").rglob("*.tex"))
    placeholders = (
        "AUTHOR INFORMATION REQUIRED",
        "AFFILIATIONS REQUIRED",
        "CORRESPONDING AUTHOR AND EMAIL REQUIRED",
        "Before submission, the authors must insert",
        "final submission must identify",
        "Before publication, the code should be deposited",
        "To be completed before submission",
        "Generative-AI disclosure requiring author confirmation",
    )
    for path in tex_files:
        content = path.read_text(encoding="utf-8", errors="ignore")
        for token in placeholders:
            if token in content:
                errors.append(f"{path.relative_to(ROOT)} contains unresolved metadata: {token}")
        if "\\pending{" in content or "PENDING --" in content:
            errors.append(f"{path.relative_to(ROOT)} contains an unresolved pending marker")

    for path in sorted((ROOT / "code").rglob("*")):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix not in {".py", ".sh", ".R", ".yaml", ".toml", ".txt", ".md"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "/home/" in content or ".local/share/Trash" in content:
            errors.append(f"{path.relative_to(ROOT)} contains a workstation-specific path")

    required_results = {
        "simple seasonal baseline": ROOT / "results/seasonal_naive/metrics.csv",
        "simple parent-area baseline": ROOT / "results/province_allocation/metrics.csv",
        "paired block bootstrap": ROOT / "results/pending/bootstrap_pairwise.csv",
        "threshold sensitivity": ROOT / "results/pending/threshold_sensitivity.csv",
        "decision utility": ROOT / "results/pending/decision_utility_summary.csv",
        "five-seed replication": ROOT / "results/pending/multiseed_summary.csv",
        "spatial hold-out": ROOT / "results/pending/spatial_holdout_summary.csv",
        "TimesFM-3 comparison": ROOT / "results/pending/timesfm3_summary.csv",
    }
    for label, path in required_results.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing completed analysis ({label}): {path.relative_to(ROOT)}")

    metrics = ROOT / "results/current/final_metrics_by_horizon.csv"
    if not metrics.is_file():
        errors.append("missing primary locked metrics")
    else:
        frame = pd.read_csv(metrics)
        if set(frame["horizon"].astype(int)) != {1, 2, 3}:
            errors.append("primary locked metrics do not contain H1, H2 and H3")
        if len(frame) != 15 or frame["model"].nunique() != 5:
            errors.append("primary locked metrics should contain five reported models x three horizons")
        if "n" not in frame or set(frame["n"].astype(int)) != {33240}:
            errors.append("five reported models do not contain 33,240 rows per horizon")

    shape_expectations = {
        ROOT / "results/pending/bootstrap_pairwise.csv": (12, None),
        ROOT / "results/pending/threshold_sensitivity.csv": (12, None),
        ROOT / "results/pending/decision_utility_summary.csv": (12, None),
        ROOT / "results/pending/spatial_holdout_summary.csv": (3, None),
    }
    for path, (expected_rows, _) in shape_expectations.items():
        if path.is_file():
            observed_rows = len(pd.read_csv(path))
            if observed_rows != expected_rows:
                errors.append(
                    f"{path.relative_to(ROOT)} has {observed_rows} rows; expected {expected_rows}"
                )
    multiseed = ROOT / "results/pending/multiseed_summary.csv"
    if multiseed.is_file():
        seeds = pd.read_csv(multiseed)
        macro = seeds[seeds["scope"].eq("mean_H1_H3")]
        if set(macro["family"]) != {"base", "alert"} or set(macro["seeds"].astype(int)) != {5}:
            errors.append("multi-seed summary does not contain five seeds for both neural families")

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    title_match = re.search(r"\\title\{([^}]*)\}", main_tex, re.S)
    title_count = len(plain_words(title_match.group(1))) if title_match else 0
    if not title_match:
        errors.append("main manuscript title was not found")
    elif title_count > 15:
        errors.append(f"title has {title_count} words; Nature Communications limit is 15")

    abstract_tex = (PAPER / "sections/00_abstract.tex").read_text(encoding="utf-8")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", abstract_tex, re.S)
    abstract_count = len(plain_words(abstract_match.group(1))) if abstract_match else 0
    if not abstract_match:
        errors.append("abstract environment was not found")
    elif abstract_count > 200:
        errors.append(f"abstract has {abstract_count} words; limit is 200")
    if "\\cite{" in abstract_tex:
        errors.append("abstract contains a citation; it must be unreferenced")

    main_section_paths = [
        PAPER / "sections/01_introduction.tex",
        PAPER / "sections/02_results.tex",
        PAPER / "sections/03_discussion.tex",
    ]
    main_word_count = sum(
        len(plain_words(prose_only(path.read_text(encoding="utf-8"))))
        for path in main_section_paths
    )
    if main_word_count > 5000:
        errors.append(f"Introduction + Results + Discussion have {main_word_count} words; expected at most 5,000")
    methods_word_count = len(plain_words(prose_only(
        (PAPER / "sections/04_methods.tex").read_text(encoding="utf-8")
    )))
    if methods_word_count > 3000:
        warnings.append(f"Methods have {methods_word_count} words; Nature Communications typically recommends under 3,000")

    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]
    )
    display_count = len(re.findall(r"\\begin\{(?:figure|table)\}", manuscript_text))
    if display_count > 10:
        errors.append(f"main manuscript has {display_count} display items; expected at most 10")

    for number, caption in enumerate(braced_arguments(manuscript_text, "caption"), start=1):
        caption_words = len(plain_words(caption))
        if caption_words > 350:
            errors.append(f"main figure/table legend {number} has {caption_words} words; limit is 350")

    cited: set[str] = set()
    for value in re.findall(r"\\cite\{([^}]+)\}", manuscript_text):
        cited.update(key.strip() for key in value.split(","))
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript_text))
    reference_count = len(defined)
    if reference_count > 70:
        errors.append(f"reference list has {reference_count} entries; expected at most 70")
    missing_citations = sorted(cited - defined)
    if missing_citations:
        errors.append("undefined bibliography keys: " + ", ".join(missing_citations))

    required_outputs = (
        PAPER / "main.pdf",
        PAPER / "supplement.pdf",
        ROOT / "submission/manuscript_submission.tex",
        ROOT / "submission/manuscript_submission.pdf",
        ROOT / "submission/Supplementary_Information.pdf",
        ROOT / "submission/source_data.zip",
    )
    for path in required_outputs:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing submission output: {path.relative_to(ROOT)}")
    if not any(ROOT.glob("LICENSE*")):
        errors.append("repository has no code licence file")
    code_statement = (PAPER / "sections/05_data_code.tex").read_text(encoding="utf-8")
    if "https://github.com/ngminhhieu/dengue-paper" not in code_statement:
        errors.append("Code availability does not contain the requested GitHub repository URL")
    if "persistent archive DOI will be added" in code_statement:
        warnings.append("Code availability has no persistent release DOI yet")

    expected_source = {
        "Table1_model_summary.csv",
        "TableS3_simple_baselines.csv",
        "TableS4_bootstrap.csv",
        "TableS5_threshold_sensitivity.csv",
        "TableS6_decision_utility.csv",
        "TableS7_multiseed.csv",
        "TableS8_spatial_holdout.csv",
        "TableS9_timesfm3.csv",
    }
    present_source = {path.name for path in (ROOT / "source_data").glob("*.csv")}
    for name in sorted(expected_source - present_source):
        errors.append(f"source-data archive input is missing {name}")
    if not (ROOT / "source_data/README.md").is_file():
        errors.append("source-data archive input is missing README.md")

    archive = ROOT / "submission/source_data.zip"
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as handle:
                archived = {Path(name).name for name in handle.namelist() if not name.endswith("/")}
            for name in sorted(expected_source - archived):
                errors.append(f"source_data.zip is missing {name}")
            if "README.md" not in archived:
                errors.append("source_data.zip is missing README.md")
        except zipfile.BadZipFile:
            errors.append("submission/source_data.zip is not a valid ZIP archive")

    for log_name in ("main.log", "supplement.log"):
        log_path = PAPER / log_name
        if not log_path.is_file():
            continue
        log = log_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"undefined references|There were undefined", log, re.I):
            errors.append(f"{log_path.relative_to(ROOT)} reports undefined references")
        if "Overfull \\hbox" in log:
            warnings.append(f"{log_path.relative_to(ROOT)} contains overfull boxes")

    print(
        f"Journal limits: title={title_count}/15 words; abstract={abstract_count}/200 words; "
        f"main text={main_word_count}/5,000 words; methods={methods_word_count}/3,000 words; "
        f"main displays={display_count}/10; references={reference_count}/70."
    )
    if warnings:
        print("WARNINGS:")
        print("\n".join(f" - {item}" for item in warnings))
    if errors:
        print("NOT FINAL-SUBMISSION READY:")
        print("\n".join(f" - {item}" for item in errors))
        return 1
    print("Submission readiness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
