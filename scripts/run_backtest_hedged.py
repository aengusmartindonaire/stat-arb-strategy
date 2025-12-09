#!/usr/bin/env python

"""
Run a SPY-hedged reversal strategy.

Pipeline:
1. Load daily stock EXCESS returns panel (data/processed).
2. Download daily EXCESS returns for hedge ETF (e.g. SPY).
3. Load precomputed per-stock betas vs SPY from data/processed.
4. Construct hedged returns: r_hedged = r_stock - beta * r_SPY.
5. Build reversal signal on hedged returns (event-driven threshold).
6. Use next-day hedged returns as forward returns.
7. Run long/short backtest and summarize performance.

Configuration is read from:
- configs/paths.yaml           (for data paths)
- configs/factor_hedged.yaml  (for strategy params)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stat_arb.config import get_paths_config, get_strategy_config, DATA_DIR
from stat_arb import data_pipeline, strategies, backtest, performance


def main() -> None:
    # --- Load configs ---
    paths_cfg = get_paths_config()
    strat_cfg = get_strategy_config("factor_hedged")  # or "hedged_reversal" if you rename the yaml

    data_cfg = paths_cfg.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", DATA_DIR / "processed"))
    panel_name = strat_cfg.get(
        "panel_name",
        data_cfg.get("panel_name", "daily_excess_returns.parquet"),
    )
    panel_path = processed_dir / panel_name

    hedge_ticker = strat_cfg.get("hedge_ticker", "SPY")
    betas_file = strat_cfg.get("betas_file", "betas_spy.parquet")
    betas_path = processed_dir / betas_file

    print("=== SPY-Hedged Reversal Strategy ===")
    print("Loading returns panel from:", panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")

    returns = pd.read_parquet(panel_path)
    returns = returns.sort_index()

    print(f"Loading betas from: {betas_path}")
    if not betas_path.exists():
        raise FileNotFoundError(
            f"Betas file not found: {betas_path}\n"
            "You should generate this from your beta-estimation notebook "
            "and save as a parquet with index=ticker and a 'beta' column."
        )

    betas_df = pd.read_parquet(betas_path)
    # Assume index is ticker and the beta column is named 'beta'
    if "beta" not in betas_df.columns:
        raise KeyError("Expected a 'beta' column in betas file.")
    betas = betas_df["beta"]

    print(f"Downloading hedge ETF returns for: {hedge_ticker}")
    hedge_returns = data_pipeline.get_daily_returns(hedge_ticker)

    # --- Compute hedged returns ---
    print("Computing hedged stock returns (stock - beta * hedge).")
    hedged_returns = strategies.compute_hedged_returns(
        stock_returns=returns,
        hedge_returns=hedge_returns,
        betas=betas,
    )

    # Align with original stock set (columns intersection)
    hedged_returns = hedged_returns[returns.columns]

    # --- Build reversal signal on hedged returns ---
    threshold = float(strat_cfg.get("threshold", 0.02))
    print(f"Using reversal threshold on hedged returns: {threshold:.4f}")

    hedged_signal = strategies.compute_reversal_signal(
        returns=hedged_returns,
        threshold=threshold,
    )

    # Forward hedged returns: r_{t+1} on hedged series
    hedged_fwd_returns = hedged_returns.shift(-1)
    hedged_signal = hedged_signal.iloc[:-1]
    hedged_fwd_returns = hedged_fwd_returns.iloc[:-1]

    # --- Backtest ---
    cfg = backtest.BacktestConfig(
        top_frac=float(strat_cfg.get("top_frac", 0.1)),
        bottom_frac=float(strat_cfg.get("bottom_frac", 0.1)),
        cost_bps=float(strat_cfg.get("cost_bps", 0.0)),
        min_names=int(strat_cfg.get("min_names", 10)),
    )
    print("Backtest config:", cfg)

    portfolio_returns, weights = backtest.long_short_backtest(
        fwd_returns=hedged_fwd_returns,
        signal=hedged_signal,
        cfg=cfg,
    )

    # --- Performance summary ---
    ic_series = performance.rank_ic_by_date(
        signal=hedged_signal,
        fwd_returns=hedged_fwd_returns,
    )
    summary = performance.summarize_portfolio_performance(
        portfolio_returns=portfolio_returns,
        ic_series=ic_series,
        periods_per_year=252,
    )

    print("\n=== Performance Summary (SPY-Hedged Reversal) ===")
    print(f"Mean IC:           {summary.mean_ic: .4f}")
    print(f"IC Hit Rate:       {summary.ic_hit_rate: .3f}")
    print(f"Cumulative Return: {summary.cumulative_return: .4f}")
    print(f"Sharpe:            {summary.sharpe: .3f}")
    print(f"Max Drawdown:      {summary.max_drawdown: .4f}")

    # --- Save outputs ---
    out_dir = processed_dir / "spy_hedged_reversal"
    out_dir.mkdir(parents=True, exist_ok=True)

    portfolio_returns.to_parquet(out_dir / "portfolio_returns.parquet")
    ic_series.to_parquet(out_dir / "rank_ic.parquet")
    weights.to_parquet(out_dir / "weights.parquet")

    print("\nSaved portfolio_returns, rank_ic, and weights under:", out_dir)


if __name__ == "__main__":
    main()