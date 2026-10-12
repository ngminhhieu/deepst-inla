#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dengue_h3.config import load_config, resolve
from dengue_h3.data import load_monthly, validate_complete_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study_lognormal.yaml")
    args = parser.parse_args()
    root, config = load_config(args.config)
    frame = load_monthly(
        resolve(root, config["data"]["monthly_path"]), config["data"]
    )
    validate_complete_panel(frame)
    zero_rate = (frame["cases"] == 0).mean()
    print(
        f"valid rows={len(frame)} communes={frame.fcode.nunique()} "
        f"months={frame.date.nunique()} zero_rate={zero_rate:.3f}"
    )


if __name__ == "__main__":
    main()
