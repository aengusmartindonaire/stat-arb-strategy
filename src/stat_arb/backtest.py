from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    top_frac: float = 0.1
    bottom_frac: float = 0.1
    cost_bps: float = 0.0
    min_names: int = 10


def _compute_daily_weights(
    signal: pd.Series,
    cfg: BacktestConfig,
) -> pd.Series:
    """
    Cross-sectional long/short weights from a signal on one date:

    - top `top_frac` names: equal-weight long
    - bottom `bottom_frac` names: equal-weight short
    - scaled so total long = +1, total short = -1
    """
    s = signal.dropna()
    if s.empty:
        return pd.Series(dtype=float)

    n = len(s)
    n_long = max(int(np.floor(cfg.top_frac * n)), 0)
    n_short = max(int(np.floor(cfg.bottom_frac * n)), 0)

    if n_long < cfg.min_names or n_short < cfg.min_names:
        return pd.Series(dtype=float)

    ranked = s.sort_values(ascending=False)

    long_names = ranked.index[:n_long]
    short_names = ranked.index[-n_short:]

    w = pd.Series(0.0, index=ranked.index)
    if n_long > 0:
        w.loc[long_names] = 1.0 / n_long
    if n_short > 0:
        w.loc[short_names] = -1.0 / n_short

    return w


def long_short_backtest(
    fwd_returns: pd.DataFrame,
    signal: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Simple long/short backtest:

    Inputs
    ------
    fwd_returns : DataFrame
        Forward returns (index: dates, columns: tickers).
    signal : DataFrame
        Signal with same shape as fwd_returns.
    cfg : BacktestConfig
        Parameters for long/short construction and costs.

    Returns
    -------
    portfolio_returns : Series
        Daily portfolio return.
    weights : DataFrame
        Daily weights (index: dates, columns: tickers).
    """
    if cfg is None:
        cfg = BacktestConfig()

    fwd_returns = fwd_returns.sort_index()
    signal = signal.sort_index()
    idx = fwd_returns.index.intersection(signal.index)
    cols = fwd_returns.columns.intersection(signal.columns)

    fwd_returns = fwd_returns.loc[idx, cols]
    signal = signal.loc[idx, cols]

    weights_list = []
    pnl_list = []
    prev_weights: Optional[pd.Series] = None

    for dt, sig_row in signal.iterrows():
        w = _compute_daily_weights(sig_row, cfg)
        if w.empty:
            weights_list.append(pd.Series(dtype=float, name=dt))
            pnl_list.append(np.nan)
            prev_weights = w
            continue

        # Turnover for costs
        if prev_weights is not None and not prev_weights.empty:
            both = w.index.union(prev_weights.index)
            w_curr = w.reindex(both).fillna(0.0)
            w_prev = prev_weights.reindex(both).fillna(0.0)
            turnover = (w_curr - w_prev).abs().sum()
        else:
            turnover = w.abs().sum()

        daily_ret = (w * fwd_returns.loc[dt, w.index]).sum()

        if cfg.cost_bps != 0.0:
            cost = cfg.cost_bps / 10000.0 * turnover
            daily_ret -= cost

        weights_list.append(w.rename(dt))
        pnl_list.append(daily_ret)
        prev_weights = w

    weights = pd.DataFrame(weights_list).sort_index()
    portfolio_returns = pd.Series(pnl_list, index=fwd_returns.index).sort_index()
    portfolio_returns.name = "portfolio_ret"

    return portfolio_returns, weights