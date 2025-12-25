import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add 'scripts' to sys.path so we can import the modules
SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import run_prepare_data
import run_backtest_unhedged
import run_backtest_hedged

# --- Fixtures for Dummy Data ---

@pytest.fixture
def dummy_panel():
    """Create a small random returns panel (10 days, 6 tickers)."""
    # NEEDS AT LEAST 5 TICKERS to pass performance.rank_ic check
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    
    # Make returns large enough to trigger the threshold
    data = np.random.randn(len(dates), len(tickers)) * 0.05
    df = pd.DataFrame(data, index=dates, columns=tickers)
    return df

@pytest.fixture
def dummy_betas():
    """Create a small betas dataframe matching the 6 tickers."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    # Random betas around 1.0
    betas = [1.2, 0.9, 1.1, 1.3, 1.0, 1.5]
    df = pd.DataFrame({"beta": betas}, index=tickers)
    return df

# --- Tests ---

@patch("run_prepare_data.get_paths_config")
@patch("stat_arb.data_pipeline.load_bloomberg_universe_excel")
@patch("stat_arb.data_pipeline.build_returns_panel")
def test_run_prepare_data_smoke(mock_build_panel, mock_load_blg, mock_get_config, tmp_path, dummy_panel):
    """
    Smoke test for run_prepare_data.py
    """
    # 1. Setup Config Mock
    mock_get_config.return_value = {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "bloomberg": {"years": [2023]},
            "panel_name": "test_panel.parquet"
        }
    }

    # 2. Setup Data Mocks
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "20230101_US_Port.xlsx").touch()

    mock_load_blg.return_value = pd.DataFrame(index=["AAPL", "MSFT"]) # content doesn't matter much here
    mock_build_panel.return_value = dummy_panel

    # 3. Run
    run_prepare_data.main()

    # 4. Assert
    out_file = tmp_path / "processed" / "test_panel.parquet"
    assert out_file.exists()


@patch("run_backtest_unhedged.get_paths_config")
@patch("run_backtest_unhedged.get_strategy_config")
def test_run_backtest_unhedged_smoke(mock_strat_cfg, mock_paths_cfg, tmp_path, dummy_panel):
    """
    Smoke test for run_backtest_unhedged.py
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    
    mock_paths_cfg.return_value = {
        "data": {
            "processed_dir": str(processed_dir),
            "panel_name": "test_panel.parquet"
        }
    }
    
    # top_frac=0.2 of 6 tickers = 1.2 -> 1 stock. Safe.
    mock_strat_cfg.return_value = {
        "threshold": 0.0001,
        "top_frac": 0.2,
        "bottom_frac": 0.2,
        "min_names": 1       
    }

    dummy_panel.to_parquet(processed_dir / "test_panel.parquet")

    run_backtest_unhedged.main()

    out_dir = processed_dir / "unhedged_reversal"
    assert (out_dir / "portfolio_returns.parquet").exists()
    assert (out_dir / "rank_ic.parquet").exists()


@patch("run_backtest_hedged.get_paths_config")
@patch("run_backtest_hedged.get_strategy_config")
@patch("stat_arb.data_pipeline.get_daily_returns")
def test_run_backtest_hedged_smoke(mock_get_daily_rets, mock_strat_cfg, mock_paths_cfg, tmp_path, dummy_panel, dummy_betas):
    """
    Smoke test for run_backtest_hedged.py
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)

    mock_paths_cfg.return_value = {
        "data": {
            "processed_dir": str(processed_dir),
            "panel_name": "test_panel.parquet"
        }
    }
    
    mock_strat_cfg.return_value = {
        "betas_file": "betas_test.parquet",
        "hedge_ticker": "SPY",
        "threshold": 0.0001, 
        "min_names": 1,
        "top_frac": 0.2,
        "bottom_frac": 0.2
    }

    dummy_panel.to_parquet(processed_dir / "test_panel.parquet")
    dummy_betas.to_parquet(processed_dir / "betas_test.parquet")

    spy_rets = pd.Series(0.001, index=dummy_panel.index, name="SPY")
    mock_get_daily_rets.return_value = spy_rets

    run_backtest_hedged.main()

    out_dir = processed_dir / "spy_hedged_reversal"
    assert (out_dir / "portfolio_returns.parquet").exists()