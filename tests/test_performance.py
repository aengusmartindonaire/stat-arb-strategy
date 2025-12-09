import numpy as np
import pandas as pd

from stat_arb import performance


def test_rank_ic_perfect_correlation():
    """
    If signal and forward returns are perfectly monotonic in the same order,
    rank-IC should be +1.
    """
    signal = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
    fwd_ret = pd.Series([0.1, 0.2, 0.3], index=["A", "B", "C"])

    ic = performance.rank_ic(signal, fwd_ret)
    assert np.isclose(ic, 1.0, atol=1e-8)


def test_rank_ic_by_date_two_periods():
    """
    rank_ic_by_date should compute IC per date when given wide DataFrames.
    In this setup, signal and returns are aligned each day, so IC_t should be +1.
    """
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    cols = ["A", "B", "C"]

    signal_df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
        index=dates,
        columns=cols,
    )
    fwd_df = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]],
        index=dates,
        columns=cols,
    )

    ic_series = performance.rank_ic_by_date(signal_df, fwd_df)
    assert ic_series.shape[0] == 2
    assert np.allclose(ic_series.values, 1.0, atol=1e-8)


def test_summarize_portfolio_performance_constant_returns():
    """
    For constant positive returns:
      - cumulative return should be (1+r)^T - 1
      - max drawdown should be 0 (equity is monotonic)
      - Sharpe should be NaN (zero volatility)
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rets = pd.Series(0.01, index=dates, name="portfolio_ret")

    # Simple IC series with a mix of positive and negative values
    ic_series = pd.Series(
        [0.1, 0.2, 0.0, -0.1],
        index=dates,
        name="rank_ic",
    )

    summary = performance.summarize_portfolio_performance(
        portfolio_returns=rets,
        ic_series=ic_series,
        periods_per_year=252,
    )

    # Mean IC and hit rate
    expected_mean_ic = (0.1 + 0.2 + 0.0 - 0.1) / 4.0
    assert np.isclose(summary.mean_ic, expected_mean_ic, atol=1e-8)
    assert np.isclose(summary.ic_hit_rate, 0.5, atol=1e-8)  # 2/4 > 0

    # Cumulative return: (1.01)^4 - 1
    expected_cum = (1.01**4) - 1.0
    assert np.isclose(summary.cumulative_return, expected_cum, atol=1e-8)

    # With constant returns, volatility is zero => Sharpe should be NaN
    assert np.isnan(summary.sharpe)

    # Monotonic up equity => max drawdown 0
    assert np.isclose(summary.max_drawdown, 0.0, atol=1e-8)


def test_summarize_trade_level_basic():
    """
    summarize_trade_level should:
      - compute IC between forward_ret and signal
      - compute avg_ret_per_trade, win_rate, num_trades from trade_ret
    """
    df = pd.DataFrame(
        {
            "forward_ret": [0.01, -0.02, 0.03],
            "signal": [1.0, -1.0, 1.0],
        }
    )
    # Usual definition: trade_ret = forward_ret * sign(signal)
    df["trade_ret"] = df["forward_ret"] * np.sign(df["signal"])

    stats = performance.summarize_trade_level(df)

    # IC should be positive (signal aligned with forward_ret)
    assert stats["IC"] > 0

    # Average trade return
    expected_avg = df["trade_ret"].mean()
    assert np.isclose(stats["avg_ret_per_trade"], expected_avg, atol=1e-8)

    # All trade_ret are positive in this toy example
    assert np.isclose(stats["win_rate"], 1.0, atol=1e-8)
    assert stats["num_trades"] == 3