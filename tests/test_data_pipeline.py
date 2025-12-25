import numpy as np
import pandas as pd
import yfinance as yf

from stat_arb import data_pipeline


def _download_close_series(tic: str) -> pd.Series:
    """
    Helper: download Close prices from yfinance using the same
    date range as data_pipeline.get_daily_rets / get_daily_returns.
    """
    tic_clean = tic.replace("/", "-")
    df = yf.download(
        tic_clean,
        start="2015-12-31",
        end="2026-01-01",
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise KeyError(f"No 'Close' column for ticker {tic_clean} from yfinance.")

    close = df["Close"].dropna()
    close.name = tic_clean
    return close


def test_get_daily_rets_matches_yfinance_close_pctchange():
    """
    get_daily_rets(tic) should be exactly:
        Close.pct_change().dropna()
    using the same date range and ticker cleaning as in the notebook.
    """
    tic = "SPY"

    # Expected: compute from raw yfinance Close prices
    close = _download_close_series(tic)
    expected = close.pct_change().dropna()

    # Actual: use the library function
    actual = data_pipeline.get_daily_rets(tic)

    # Align on common index before comparison
    common_idx = expected.index.intersection(actual.index)
    expected_aligned = expected.loc[common_idx]
    actual_aligned = actual.loc[common_idx]

    assert actual_aligned.name == expected_aligned.name
    assert np.allclose(actual_aligned.values, expected_aligned.values, atol=1e-5)


def test_get_daily_returns_matches_raw_minus_rf():
    """
    get_daily_returns(tic) should be:
        raw_return_t - rf_daily_month_t

    where:
      - raw_return_t = pct_change of Close from yfinance
      - rf_daily_month_t is pulled from FRED TB3MS, converted to daily, and
        looked up by key 'YYYY-MM-01', exactly as in the notebook.
    """
    tic = "SPY"

    # Raw returns from yfinance
    close = _download_close_series(tic)
    raw = close.pct_change().dropna()

    # Risk-free mapping (daily) from the package helper
    rf_dict = data_pipeline.get_rf_rate_mapping()

    # Map each date to 'YYYY-MM-01' and pull rf; default 0 if missing
    keys = raw.index.map(lambda ts: f"{ts:%Y-%m-01}")
    rf_vals = np.array([rf_dict.get(k, 0.0) for k in keys])

    expected = raw - rf_vals

    # Actual from our helper (which also hits yfinance + FRED)
    actual = data_pipeline.get_daily_returns(tic)

    common_idx = expected.index.intersection(actual.index)
    expected_aligned = expected.loc[common_idx]
    actual_aligned = actual.loc[common_idx]

    assert np.allclose(actual_aligned.values, expected_aligned.values, atol=1e-5)

def test_build_returns_panel_shape_and_columns():
    """
    build_returns_panel(tickers, use_excess=True) should:
      - call get_daily_returns for each ticker
      - return a wide DataFrame with those tickers as columns
      - have a monotonically increasing date index
    """
    tickers = ["SPY", "QQQ"]

    panel = data_pipeline.build_returns_panel(tickers, use_excess=True)

    # At least some data, right shape, right columns
    assert set(panel.columns) == set(tickers)
    assert panel.index.is_monotonic_increasing
    # Should have non-NaN entries
    assert panel.notna().sum().sum() > 0
