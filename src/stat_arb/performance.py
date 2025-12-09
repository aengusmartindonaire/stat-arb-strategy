from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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


# ---------- Cross-sectional rank-IC + portfolio summary ----------

def rank_ic(
    signal: pd.Series,
    fwd_returns: pd.Series,
) -> float:
    """
    Cross-sectional rank-IC between a signal and forward returns on one date.
    """
    df = pd.concat(
        [signal.rename("signal"), fwd_returns.rename("ret")],
        axis=1,
    ).dropna()
    if len(df) < 5:
        return np.nan

    ranks_signal = df["signal"].rank()
    ranks_ret = df["ret"].rank()
    return float(ranks_signal.corr(ranks_ret))


def rank_ic_by_date(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
) -> pd.Series:
    """
    Time series of rank-IC values (one IC per date).

    Both inputs should be DataFrames with:
        index  = dates
        columns = tickers
    """
    signal, fwd_returns = signal.align(fwd_returns, join="inner", axis=0)

    ic_values = {}
    for dt in signal.index:
        ic_values[dt] = rank_ic(signal.loc[dt], fwd_returns.loc[dt])

    ic_series = pd.Series(ic_values).sort_index()
    ic_series.name = "rank_ic"
    return ic_series


@dataclass
class PerformanceSummary:
    mean_ic: float
    ic_hit_rate: float
    cumulative_return: float
    sharpe: float
    max_drawdown: float


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_portfolio_performance(
    portfolio_returns: pd.Series,
    ic_series: pd.Series,
    periods_per_year: int = 252,
) -> PerformanceSummary:
    """
    Summarize portfolio-level stats:

    - mean IC
    - IC hit rate (fraction of IC_t > 0)
    - cumulative return
    - Sharpe ratio (annualized)
    - max drawdown
    """
    rets = portfolio_returns.dropna()
    ic = ic_series.dropna()

    if rets.empty:
        raise ValueError("Portfolio returns are empty.")
    if ic.empty:
        raise ValueError("IC series is empty.")

    mean_ic = float(ic.mean())
    ic_hit_rate = float((ic > 0).mean())

    mean_ret = float(rets.mean())
    vol = float(rets.std())
    sharpe = (mean_ret / vol * np.sqrt(periods_per_year)) if vol > 0 else np.nan

    cumulative_return = float((1 + rets).prod() - 1.0)
    max_dd = _max_drawdown_from_returns(rets)

    return PerformanceSummary(
        mean_ic=mean_ic,
        ic_hit_rate=ic_hit_rate,
        cumulative_return=cumulative_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


def build_summary_table(
    name_to_summary: Dict[str, PerformanceSummary],
) -> pd.DataFrame:
    """
    Turn {strategy_name: PerformanceSummary} into a DataFrame.
    """
    rows = {}
    for name, s in name_to_summary.items():
        rows[name] = {
            "Mean IC": s.mean_ic,
            "IC Hit Rate": s.ic_hit_rate,
            "Cumulative Return": s.cumulative_return,
            "Sharpe": s.sharpe,
            "Max Drawdown": s.max_drawdown,
        }

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Strategy"
    return table


# ---------- Trade-level stats for event strategies ----------


def summarize_trade_level(
    outcomes_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute trade-level statistics for event strategies.

    Expected columns:
        - 'forward_ret'
        - 'signal'
        - 'trade_ret' (e.g. forward_ret * sign(signal))
    """
    if outcomes_df.empty:
        return {
            "IC": np.nan,
            "avg_ret_per_trade": np.nan,
            "win_rate": np.nan,
            "num_trades": 0,
        }

    out: Dict[str, float] = {}

    if "forward_ret" in outcomes_df.columns and "signal" in outcomes_df.columns:
        out["IC"] = float(
            outcomes_df["forward_ret"].corr(outcomes_df["signal"])
        )
    else:
        out["IC"] = np.nan

    if "trade_ret" in outcomes_df.columns:
        tr = outcomes_df["trade_ret"].dropna()
        if len(tr) == 0:
            out["avg_ret_per_trade"] = np.nan
            out["win_rate"] = np.nan
            out["num_trades"] = 0
        else:
            out["avg_ret_per_trade"] = float(tr.mean())
            out["win_rate"] = float((tr > 0).mean())
            out["num_trades"] = int(len(tr))
    else:
        out["avg_ret_per_trade"] = np.nan
        out["win_rate"] = np.nan
        out["num_trades"] = 0

    return out
