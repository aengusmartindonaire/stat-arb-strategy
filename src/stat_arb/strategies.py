from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------- Reversal-style daily signals ----------


def compute_reversal_signal(
    returns: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Event-driven reversal signal on daily returns.

    For each (date, stock):

        prev_ret = return_t
        if |prev_ret| > threshold:
            signal = -prev_ret
        else:
            signal = NaN (no trade)

    This matches the logic inside your `walk_forward_unhedged_reversal`:
    - you later form trade_ret = fwd_ret * sign(signal)
    - `threshold` is your X% move (e.g. 0.02 for ±2%).
    """
    prev_ret = returns.copy()
    mask = prev_ret.abs() > threshold
    signal = -prev_ret.where(mask)
    return signal


# ---------- 12M-1M momentum rank signal ----------


def compute_12m_1m_momentum_rank(
    returns: pd.DataFrame,
    lookback_days: int = 252,
    gap_days: int = 21,
) -> pd.DataFrame:
    """
    12M-1M momentum score as in the notebook:

    - compute a rolling mean over (lookback_days - gap_days) trading days
    - shift by gap_days (so you exclude the most recent month)
    - for each date, rank cross-sectionally (percentile) and rescale to [-0.5, 0.5]

    In code, the notebook did:

        momentum_scores = returns_window.rolling(252 - 21).mean().shift(21)
        momentum_rank = momentum_scores.rank(pct=True, axis=1) - 0.5
    """
    window = lookback_days - gap_days
    momentum_scores = returns.rolling(window).mean().shift(gap_days)
    momentum_rank = momentum_scores.rank(pct=True, axis=1) - 0.5
    return momentum_rank


# ---------- Hedged returns helper (for SPY/QQQ-hedged strategies) ----------


def compute_hedged_returns(
    stock_returns: pd.DataFrame,
    hedge_returns: pd.Series,
    betas: pd.Series,
) -> pd.DataFrame:
    """
    Compute hedged returns:

        r_hedged_{t,i} = r_stock_{t,i} - beta_i * r_hedge_t

    Inputs
    ------
    stock_returns : DataFrame
        index: dates, columns: tickers
    hedge_returns : Series
        index: dates, single hedge ETF series (e.g. SPY)
    betas : Series
        index: tickers, values: beta vs the hedge ETF

    This matches the `hedged_stock_ret = stockforthisyear - beta * hedge_series.loc[...]`
    pattern you use in the SPY-hedged reversal and momentum strategies.
    """
    stock_returns = stock_returns.sort_index()
    hedge_returns = hedge_returns.sort_index()

    common_dates = stock_returns.index.intersection(hedge_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    hedge_series = hedge_returns.loc[common_dates]

    betas = betas.reindex(stock_returns.columns).fillna(0.0)

    beta_matrix = pd.DataFrame(
        np.tile(betas.to_numpy(), (len(stock_returns), 1)),
        index=stock_returns.index,
        columns=stock_returns.columns,
    )

    hedge_matrix = pd.DataFrame(
        np.tile(hedge_series.to_numpy().reshape(-1, 1), (1, stock_returns.shape[1])),
        index=stock_returns.index,
        columns=stock_returns.columns,
    )

    hedged = stock_returns - beta_matrix * hedge_matrix
    return hedged