#!/usr/bin/env python

"""
Prepare core data for the stat-arb project.

This script:
1. Loads Bloomberg universe files for a set of years.
2. Collects the union of tickers across those years.
3. Downloads daily EXCESS returns via yfinance (stock - rf).
4. Saves the wide returns panel to data/processed as a parquet file.

Configuration is read from configs/paths.yaml. A minimal example:

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  bloomberg:
    years: [2020, 2021, 2022, 2023, 2024, 2025]
    pattern: "{year}0101_US_Port.xlsx"
  panel_name: "daily_excess_returns.parquet"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stat_arb.config import get_paths_config, DATA_DIR
from stat_arb import data_pipeline


def main() -> None:
    # --- Load config ---
    paths_cfg = get_paths_config()

    data_cfg = paths_cfg.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", DATA_DIR / "raw"))
    processed_dir = Path(data_cfg.get("processed_dir", DATA_DIR / "processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    bloomberg_cfg = data_cfg.get("bloomberg", {})
    years = bloomberg_cfg.get("years", [2020, 2021, 2022, 2023, 2024, 2025])
    pattern = bloomberg_cfg.get("pattern", "{year}0101_US_Port.xlsx")

    panel_name = data_cfg.get("panel_name", "daily_excess_returns.parquet")
    panel_path = processed_dir / panel_name

    print("=== Step 1: Loading Bloomberg universes ===")
    year2blg = {}
    all_tickers = set()

    for yr in years:
        fn = raw_dir / pattern.format(year=yr)
        print(f"  - Year {yr}: {fn}")
        if not fn.exists():
            print(f"    WARNING: file not found, skipping {fn}")
            continue

        blg = data_pipeline.load_bloomberg_universe_excel(fn)
        year2blg[yr] = blg
        all_tickers.update(blg.index.tolist())

    if not all_tickers:
        raise RuntimeError("No tickers found from Bloomberg universe files.")

    tickers = sorted(all_tickers)
    print(f"Total unique tickers across all years: {len(tickers)}")

    # --- Build daily excess returns panel ---
    print("=== Step 2: Building daily EXCESS returns panel ===")
    print("This will download yfinance data, so it may take a while the first time.")
    returns_panel = data_pipeline.build_returns_panel(
        tickers,
        use_excess=True,  # uses get_daily_returns (stock - rf)
    )

    print("Panel shape:", returns_panel.shape)

    # --- Save to disk ---
    print("=== Step 3: Saving panel ===")
    print(f"Saving to: {panel_path}")
    returns_panel.to_parquet(panel_path)

    # Optional: also save a CSV with just the index/columns (metadata)
    meta_path = processed_dir / "daily_excess_returns_meta.csv"
    meta_df = pd.DataFrame(
        {
            "num_dates": [returns_panel.shape[0]],
            "num_tickers": [returns_panel.shape[1]],
        }
    )
    meta_df.to_csv(meta_path, index=False)
    print("Saved metadata to:", meta_path)

    print("Done. You can now load this panel in notebooks or backtest scripts.")


if __name__ == "__main__":
    main()