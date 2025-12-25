import numpy as np
import pandas as pd

from stat_arb import backtest

def test_compute_daily_weights_balanced_long_short():
    """
    For a simple signal with clear ranking, _compute_daily_weights should
    produce +1 total long exposure and -1 total short exposure when
    top_frac and bottom_frac are symmetric.
    """
    # A > B > C > D, so A is strongest, D is weakest
    signal = pd.Series([4.0, 3.0, 2.0, 1.0], index=list("ABCD"))

    cfg = backtest.BacktestConfig(
        top_frac=0.25,    # top 25% -> 1 name long
        bottom_frac=0.25, # bottom 25% -> 1 name short
        cost_bps=0.0,
        min_names=1,
    )

    w = backtest._compute_daily_weights(signal, cfg)

    # Only A and D should be non-zero
    non_zero = w[w != 0.0]
    assert set(non_zero.index) == {"A", "D"}

    # Long should sum to +1, short to -1, total zero
    long_sum = non_zero[non_zero > 0].sum()
    short_sum = non_zero[non_zero < 0].sum()

    assert np.isclose(long_sum, 1.0)
    assert np.isclose(short_sum, -1.0)
    assert np.isclose(non_zero.sum(), 0.0)


def test_long_short_backtest_long_only_constant_returns():
    """
    In a toy case with:
      - constant +1% forward returns for all stocks and dates
      - signal that ranks stocks but we only go long (no short),
    the portfolio should earn roughly +1% per period and weights should sum to 1.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    tickers = ["A", "B", "C"]

    # Constant 1% forward returns
    fwd_returns = pd.DataFrame(
        0.01,
        index=dates,
        columns=tickers,
    )

    # Signal increases with ticker index, same across dates
    row = pd.Series([1.0, 2.0, 3.0], index=tickers)
    signal = pd.DataFrame(
        [row.values] * len(dates),
        index=dates,
        columns=tickers,
    )

    # Long-only config: top_frac=1, bottom_frac=0, no min_names constraint
    cfg = backtest.BacktestConfig(
        top_frac=1.0,
        bottom_frac=0.0,
        cost_bps=0.0,
        min_names=0,
    )

    portfolio_returns, weights = backtest.long_short_backtest(
        fwd_returns=fwd_returns,
        signal=signal,
        cfg=cfg,
    )

    # Every day we should be 100% long, so portfolio return ~1%
    assert np.allclose(portfolio_returns.dropna().values, 0.01, atol=1e-8)

    # Weights each day should sum to +1
    daily_sums = weights.fillna(0.0).sum(axis=1)
    assert np.allclose(daily_sums.values, 1.0, atol=1e-8)