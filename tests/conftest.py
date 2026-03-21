"""Shared test fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import init_db, get_session, Base, Position, Trade


@pytest.fixture(autouse=True)
def test_db(tmp_path):
    """Create a fresh test database for each test."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    yield db_path


@pytest.fixture
def session(test_db):
    """Get a database session."""
    return get_session()


@pytest.fixture
def sample_config():
    """Return a sample configuration dict."""
    return {
        "connection": {
            "host": "127.0.0.1",
            "paper_port": 7497,
            "live_port": 7496,
            "client_id": 1,
            "trading_mode": "paper",
            "timeout": 30,
            "max_reconnect_attempts": 3,
            "reconnect_delay": 5,
        },
        "strategies": {
            "covered_call": {
                "enabled": True,
                "target_delta_min": 0.20,
                "target_delta_max": 0.35,
                "min_dte": 30,
                "max_dte": 45,
                "min_iv_rank": 30,
                "earnings_blackout_days": 7,
            },
            "cash_secured_put": {
                "enabled": True,
                "target_delta_min": -0.30,
                "target_delta_max": -0.20,
                "min_dte": 30,
                "max_dte": 45,
                "min_iv_rank": 40,
                "max_buying_power_pct": 0.20,
            },
            "iron_condor": {
                "enabled": True,
                "min_vix": 18,
                "profit_target_pct": 0.33,
            },
            "credit_spread": {
                "enabled": True,
                "max_risk_pct": 0.02,
            },
            "debit_spread": {
                "enabled": True,
                "max_spend_pct": 0.005,
            },
        },
        "risk": {
            "max_portfolio_options_pct": 0.40,
            "max_single_name_pct": 0.10,
            "max_daily_loss_pct": 0.02,
            "margin_buffer_pct": 0.30,
            "earnings_blackout_days": 5,
            "kelly_fraction": 0.25,
            "max_position_count": 30,
        },
        "hedging": {
            "enabled": True,
            "delta_threshold": 0.50,
            "intraday_drop_pct": 0.05,
            "stock_drop_pct": 0.08,
            "gap_down_pct": 0.10,
            "vix_spike_pct": 0.25,
        },
        "reporting": {
            "db_path": "db/test.db",
        },
    }


@pytest.fixture
def account_summary_healthy():
    """Account summary with healthy margins."""
    return {
        "NetLiquidation": 230000.0,
        "BuyingPower": 460000.0,
        "ExcessLiquidity": 150000.0,
        "GrossPositionValue": 50000.0,
        "MaintMarginReq": 80000.0,
        "AvailableFunds": 150000.0,
        "UnrealizedPnL": 1500.0,
        "RealizedPnL": 500.0,
    }


@pytest.fixture
def account_summary_tight_margin():
    """Account summary with tight margins."""
    return {
        "NetLiquidation": 230000.0,
        "BuyingPower": 50000.0,
        "ExcessLiquidity": 30000.0,
        "GrossPositionValue": 180000.0,
        "MaintMarginReq": 200000.0,
        "AvailableFunds": 30000.0,
        "UnrealizedPnL": -3000.0,
        "RealizedPnL": -1000.0,
    }
