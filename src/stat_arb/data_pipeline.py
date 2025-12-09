from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ------------- Bloomberg universe loading -------------


def load_bloomberg_universe_excel(fn: str | Path) -> pd.DataFrame:
    """
    Load a Bloomberg yearly universe export (YYYY0101_US_Port.xlsx).

    - read Excel with header after 2 rows (skiprows=2)
    - filter rows with non-null 'Market Cap'
    - sanitize 'Ticker' (take first token)
    - keep only 'Ticker.1' ending with ' US'
    - restrict tickers to len <= 4
    - drop names whose 'Short Name' contains '-ADR'
    - rename 'PORT US ...' cols to the 3rd token
    - normalize some special column names (ESG, GICS, Employees)
    - strip text after ':' and normalize spaces + slashes
    - set index to 'Ticker', sort by Market_Cap descending,
      take top 1000 rows
    """
    fn = Path(fn)
    df = pd.read_excel(fn, skiprows=2)

    # Keep only rows with market cap
    if "Market Cap" not in df.columns:
        raise KeyError("Expected column 'Market Cap' in Bloomberg file.")
    df = df.loc[df["Market Cap"].notna()]

    # Sanitize Ticker and filter to US common stock universe
    if "Ticker" not in df.columns or "Ticker.1" not in df.columns:
        raise KeyError("Expected columns 'Ticker' and 'Ticker.1' in Bloomberg file.")

    df["Ticker"] = df["Ticker"].map(lambda x: str(x).split()[0])
    df = df.loc[df["Ticker.1"].astype(str).str.endswith(" US")]

    # Ticker length ≤ 4
    df = df.loc[df["Ticker"].map(lambda x: len(str(x)) <= 4)]

    # Drop ADRs
    if "Short Name" in df.columns:
        df = df.loc[~df["Short Name"].astype(str).str.contains("-ADR")]

    # Map PORT US columns to short names (3rd token)
    mapping_dict: Dict[str, str] = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith("PORT US"):
            tokens = col.split()
            if len(tokens) >= 3:
                mapping_dict[col] = tokens[2]

    # Extra explicit mappings as in your notebook
    mapping_dict.update(
        {
            "ESG Disclosure Score (Latest Available) (BLOOMBERG L.P.)": "ESG_Score",
            "GICS_Sector.1": "GICS_Sector_Name",
            "GICS Sector.1": "GICS_Sector_Name",
            "Number of Employees:Y": "EmployeesToday",
            "Number of Employees:Y-5": "Employees5yearsAgo",
        }
    )

    df = df.rename(columns=mapping_dict)

    # Split off any suffix after ":" (e.g. "Metric:Desc") and normalize names
    df.columns = df.columns.map(str)
    df.columns = df.columns.str.split(":").str[0]
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("/", "_")

    # Rename market cap for convenience
    if "Market Cap" in df.columns:
        df = df.rename(columns={"Market Cap": "Market_Cap"})

    # Index by Ticker, sort by market cap, keep top 1000
    df = df.set_index("Ticker")
    if "Market_Cap" in df.columns:
        df = df.sort_values("Market_Cap", ascending=False).head(1000)

    return df


def build_year2blg(
    years: Iterable[int],
    folder: str | Path = ".",
    suffix: str = "0101_US_Port.xlsx",
) -> dict[int, pd.DataFrame]:
    """
    Convenience helper to replicate:

        year2blg = {yr: loadblgexcel(f"{yr}0101_US_Port.xlsx") for yr in range(2020, 2026)}

    but with a bit more structure.
    """
    folder = Path(folder)
    year2blg: dict[int, pd.DataFrame] = {}
    for yr in years:
        fn = folder / f"{yr}{suffix}"
        year2blg[yr] = load_bloomberg_universe_excel(fn)
    return year2blg


# ------------- Daily returns (with and without RF adjustment) -------------


@lru_cache(maxsize=None)
def get_daily_rets(tic: str) -> pd.Series:
    """
    Plain daily percent returns for a ticker from yfinance.

    This mirrors the notebook's `get_daily_rets`:

    - replace '/' with '-' in ticker symbol
    - download from 2015-12-31 to 2026-01-01
    - compute pct_change of Close
    """
    tic_clean = tic.replace("/", "-")
    df = yf.download(tic_clean, start="2015-12-31", end="2026-01-01", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise KeyError(f"No 'Close' column for ticker {tic} from yfinance.")

    dailyrets = df["Close"].pct_change().dropna()
    dailyrets.name = tic_clean
    return dailyrets


_RF_RATE_CACHE: Optional[Dict[str, float]] = None


def _load_rf_rate_from_fred() -> Dict[str, float]:
    """
    Recreate the rf_rate logic from the notebook:

    - pull TB3MS from FRED CSV
    - convert % monthly rate to DAILY rate: (1 + r/100)**(1/252) - 1
    - index by 'YYYY-MM-01' strings
    - manually patch missing 2025-09-01 with 2025-08-01
    - return as a dict mapping 'YYYY-MM-01' -> daily rf
    """
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=TB3MS&scale=left&cosd=1934-01-01&coed=2025-08-01&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-09-19&revision_date=2025-09-19&nd=1934-01-01"
    )
    rf_df = pd.read_csv(url, parse_dates=True, index_col=0)

    # Monthly % to daily decimal
    rf_df = (1 + rf_df * 0.01) ** (1 / 252) - 1

    # Index as 'YYYY-MM-01'
    rf_df.index = rf_df.index.map(lambda ts: f"{ts:%Y-%m-01}")

    # Patch missing 2025-09 if needed (copy 2025-08)
    if "TB3MS" in rf_df.columns and "2025-09-01" not in rf_df.index:
        if "2025-08-01" in rf_df.index:
            rf_df.loc["2025-09-01", "TB3MS"] = rf_df.loc["2025-08-01", "TB3MS"]

    if "TB3MS" not in rf_df.columns:
        raise KeyError("Expected column 'TB3MS' in FRED rf data.")

    rf_series = rf_df["TB3MS"]
    return rf_series.to_dict()


def get_rf_rate_mapping() -> Dict[str, float]:
    """Return cached rf_rate mapping { 'YYYY-MM-01': daily_rf }."""
    global _RF_RATE_CACHE
    if _RF_RATE_CACHE is None:
        _RF_RATE_CACHE = _load_rf_rate_from_fred()
    return _RF_RATE_CACHE


@lru_cache(maxsize=None)
def get_daily_returns(tic: str, rf_rate: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Daily *excess* returns (stock - risk-free) for a ticker.

    This mirrors the notebook's `get_daily_returns`:

    - download daily prices via yfinance
    - compute pct-change of Close
    - subtract a daily risk-free rate looked up by month:
      key = 'YYYY-MM-01' string, value = rf_daily
    """
    if rf_rate is None:
        rf_rate = get_rf_rate_mapping()

    tic_clean = tic.replace("/", "-")
    df = yf.download(tic_clean, start="2015-12-31", end="2026-01-01", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise KeyError(f"No 'Close' column for ticker {tic} from yfinance.")

    dailyrets = df["Close"].pct_change().dropna()
    dailyrets.name = tic_clean

    if len(dailyrets) > 0:
        # Map each date to 'YYYY-MM-01', then to rf_rate dict; default 0 if missing.
        rf_keys = dailyrets.index.map(lambda ts: f"{ts:%Y-%m-01}")
        rf_vals = [rf_rate.get(k, 0.0) for k in rf_keys]
        dailyrets = dailyrets - np.array(rf_vals)

    return dailyrets


def build_returns_panel(
    tickers: Iterable[str],
    use_excess: bool = True,
) -> pd.DataFrame:
    """
    Build a wide DataFrame of daily returns for a list of tickers.

    If use_excess=True, uses `get_daily_returns` (excess returns).
    Otherwise, uses `get_daily_rets` (raw returns).
    """
    series_list: List[pd.Series] = []
    for tic in tickers:
        try:
            if use_excess:
                s = get_daily_returns(tic)
            else:
                s = get_daily_rets(tic)
            series_list.append(s)
        except Exception as e:
            print(f"[build_returns_panel] Skipping {tic}: {e}")

    if not series_list:
        raise ValueError("No valid tickers with returns data.")

    panel = pd.concat(series_list, axis=1).sort_index()
    return panel