#!/usr/bin/env python

"""
Run an unhedged reversal strategy on the prepared daily excess returns panel.

Pipeline:
1. Load daily panel from data/processed (excess returns).
2. Build event-driven reversal signal using a threshold.
3. Use next-day returns as forward returns.
4. Run a long/short backtest on the signal.
5. Compute rank-IC and performance summary, then print them.

Configuration is read from:
- configs/paths.yaml           (for data paths)
- configs/unhedged_reversal.yaml (for strategy params)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stat_arb.config import get_paths_config, get_strategy_config, DATA_DIR
from stat_arb import strategies, backtest, performance


def main() -> None:
    # --- Load configs ---
    paths_cfg = get_paths_config()
    strat_cfg = get_strategy_config("unhedged_reversal")

    data_cfg = paths_cfg.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", DATA_DIR / "processed"))
    panel_name = strat_cfg.get(
        "panel_name",
        data_cfg.get("panel_name", "daily_excess_returns.parquet"),
    )
    panel_path = processed_dir / panel_name

    print("=== Unhedged Reversal Strategy ===")
    print("Loading returns panel from:", panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")

    # Daily excess returns (stock - rf)
    returns = pd.read_parquet(panel_path)
    returns = returns.sort_index()

    # --- Build reversal signal ---
    threshold = float(strat_cfg.get("threshold", 0.02))
    print(f"Using reversal threshold: {threshold:.4f} (absolute daily return)")

    # Event-driven reversal: signal_t = -ret_t if |ret_t| > threshold else NaN
    reversal_signal = strategies.compute_reversal_signal(returns, threshold=threshold)

    # Forward returns: ret_{t+1} used as payoff for signal at t
    fwd_returns = returns.shift(-1)
    reversal_signal = reversal_signal.iloc[:-1]
    fwd_returns = fwd_returns.iloc[:-1]

    # --- Run long/short backtest ---
    cfg = backtest.BacktestConfig(
        top_frac=float(strat_cfg.get("top_frac", 0.1)),
        bottom_frac=float(strat_cfg.get("bottom_frac", 0.1)),
        cost_bps=float(strat_cfg.get("cost_bps", 0.0)),
        min_names=int(strat_cfg.get("min_names", 10)),
    )

    print("Backtest config:", cfg)

    portfolio_returns, weights = backtest.long_short_backtest(
        fwd_returns=fwd_returns,
        signal=reversal_signal,
        cfg=cfg,
    )

    # --- Compute rank-IC and performance summary ---
    ic_series = performance.rank_ic_by_date(
        signal=reversal_signal,
        fwd_returns=fwd_returns,
    )
    summary = performance.summarize_portfolio_performance(
        portfolio_returns=portfolio_returns,
        ic_series=ic_series,
        periods_per_year=252,
    )

    # --- Print results ---
    print("\n=== Performance Summary (Unhedged Reversal) ===")
    print(f"Mean IC:           {summary.mean_ic: .4f}")
    print(f"IC Hit Rate:       {summary.ic_hit_rate: .3f}")
    print(f"Cumulative Return: {summary.cumulative_return: .4f}")
    print(f"Sharpe:            {summary.sharpe: .3f}")
    print(f"Max Drawdown:      {summary.max_drawdown: .4f}")

    # Optional: save outputs for plotting/analysis
    out_dir = processed_dir / "unhedged_reversal"
    out_dir.mkdir(parents=True, exist_ok=True)

    portfolio_returns.to_parquet(out_dir / "portfolio_returns.parquet")
    ic_series.to_parquet(out_dir / "rank_ic.parquet")
    weights.to_parquet(out_dir / "weights.parquet")

    print("\nSaved portfolio_returns, rank_ic, and weights under:", out_dir)


if __name__ == "__main__":
    main()