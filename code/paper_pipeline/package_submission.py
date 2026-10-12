#!/usr/bin/env python3
"""Copy compiled PDFs and build a deterministic source-data ZIP."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SUBMISSION = ROOT / "submission"
SOURCE = ROOT / "source_data"


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PAPER / "supplement.pdf", SUBMISSION / "Supplementary_Information.pdf")
    archive = SUBMISSION / "source_data.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in SOURCE.iterdir() if item.is_file()):
            handle.write(path, arcname=path.name)
    for suffix in (".aux", ".fdb_latexmk", ".fls", ".log", ".out"):
        for path in SUBMISSION.glob(f"*{suffix}"):
            path.unlink()
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
