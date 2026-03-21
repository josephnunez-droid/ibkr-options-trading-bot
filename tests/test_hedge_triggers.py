"""Unit tests for hedge trigger logic."""

import pytest
from hedge_manager import HedgeManager
from db.models import get_session, Position


class MockDataFeed:
    """Mock data feed for testing."""

    def __init__(self):
        self._vix = 20.0
        self._prices = {}

    def get_vix(self):
        return self._vix

    def get_current_price(self, symbol):
        return self._prices.get(symbol, 100.0)

    def get_option_by_delta(self, *args, **kwargs):
        return {
            "strike": 95.0,
            "expiry": "20260401",
            "mid": 2.50,
            "ask": 2.75,
            "delta": -0.50,
        }

    def get_portfolio_positions(self):
        return []


class TestVIXSpike:
    """Test VIX spike detection and trading pause."""

    def test_no_spike(self, sample_config, test_db):
        data_feed = MockDataFeed()
        data_feed._vix = 20.0

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._vix_baseline = 18.0

        trigger = mgr.check_vix_spike()
        assert trigger is None
        assert mgr.is_paused is False

    def test_spike_triggers_pause(self, sample_config, test_db):
        data_feed = MockDataFeed()
        data_feed._vix = 28.0  # 40% spike from 20

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._vix_baseline = 20.0

        trigger = mgr.check_vix_spike()
        assert trigger is not None
        assert trigger["type"] == "VIX_SPIKE"
        assert trigger["severity"] == "CRITICAL"
        assert mgr.is_paused is True

    def test_spike_below_threshold(self, sample_config, test_db):
        data_feed = MockDataFeed()
        data_feed._vix = 22.0  # 10% spike, below 25% threshold

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._vix_baseline = 20.0

        trigger = mgr.check_vix_spike()
        assert trigger is None
        assert mgr.is_paused is False

    def test_pause_resumes_when_vix_subsides(self, sample_config, test_db):
        data_feed = MockDataFeed()

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._vix_baseline = 20.0
        mgr._trading_paused = True

        data_feed._vix = 21.5  # below half-spike threshold
        trigger = mgr.check_vix_spike()
        assert mgr.is_paused is False


class TestShortOptionHedge:
    """Test delta-based hedge triggers for short options."""

    def test_no_trigger_below_threshold(self, sample_config, session):
        data_feed = MockDataFeed()
        mgr = HedgeManager(sample_config, data_feed=data_feed)

        pos = Position(
            symbol="AAPL",
            strategy="cash_secured_put",
            contract_type="PUT",
            quantity=-1,
            avg_price=5.0,
            delta=-0.30,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_short_option_hedge(pos)
        assert trigger is None

    def test_trigger_above_threshold(self, sample_config, session):
        data_feed = MockDataFeed()
        mgr = HedgeManager(sample_config, data_feed=data_feed)

        pos = Position(
            symbol="AAPL",
            strategy="cash_secured_put",
            contract_type="PUT",
            quantity=-1,
            avg_price=5.0,
            delta=-0.65,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_short_option_hedge(pos)
        assert trigger is not None
        assert trigger["type"] == "DELTA_HEDGE"
        assert trigger["action"] == "BUY_PROTECTIVE_PUT"

    def test_short_call_trigger(self, sample_config, session):
        data_feed = MockDataFeed()
        mgr = HedgeManager(sample_config, data_feed=data_feed)

        pos = Position(
            symbol="AAPL",
            strategy="covered_call",
            contract_type="CALL",
            quantity=-1,
            avg_price=3.0,
            delta=0.60,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_short_option_hedge(pos)
        assert trigger is not None
        assert trigger["action"] == "BUY_PROTECTIVE_CALL"


class TestStockHedge:
    """Test stock position hedge triggers."""

    def test_no_trigger_when_stock_up(self, sample_config, session):
        data_feed = MockDataFeed()
        data_feed._prices["AAPL"] = 160.0

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._price_baselines["AAPL"] = 155.0

        pos = Position(
            symbol="AAPL",
            strategy="covered_call",
            contract_type="STOCK",
            quantity=100,
            avg_price=150.0,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_stock_hedge(pos)
        assert trigger is None

    def test_trigger_on_8pct_drop(self, sample_config, session):
        data_feed = MockDataFeed()
        data_feed._prices["AAPL"] = 137.0  # down 8.7% from entry 150

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._price_baselines["AAPL"] = 150.0

        pos = Position(
            symbol="AAPL",
            strategy="covered_call",
            contract_type="STOCK",
            quantity=100,
            avg_price=150.0,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_stock_hedge(pos)
        assert trigger is not None
        assert trigger["type"] == "STOCK_DROP"
        assert trigger["action"] == "BUY_PROTECTIVE_PUT"

    def test_gap_down_trigger(self, sample_config, session):
        data_feed = MockDataFeed()
        data_feed._prices["AAPL"] = 133.0  # 11.3% gap from baseline

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._price_baselines["AAPL"] = 150.0

        pos = Position(
            symbol="AAPL",
            strategy="covered_call",
            contract_type="STOCK",
            quantity=100,
            avg_price=155.0,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_stock_hedge(pos)
        assert trigger is not None
        assert trigger["type"] in ("GAP_DOWN", "STOCK_DROP")


class TestIntradayDrop:
    """Test intraday drop emergency stop."""

    def test_no_trigger_for_long_positions(self, sample_config, session):
        data_feed = MockDataFeed()
        data_feed._prices["AAPL"] = 140.0

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._price_baselines["AAPL"] = 150.0

        pos = Position(
            symbol="AAPL",
            strategy="test",
            contract_type="PUT",
            quantity=1,  # long
            avg_price=5.0,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_intraday_drop(pos)
        assert trigger is None

    def test_trigger_for_short_on_5pct_drop(self, sample_config, session):
        data_feed = MockDataFeed()
        data_feed._prices["AAPL"] = 141.0  # 6% drop from 150

        mgr = HedgeManager(sample_config, data_feed=data_feed)
        mgr._price_baselines["AAPL"] = 150.0

        pos = Position(
            symbol="AAPL",
            strategy="cash_secured_put",
            contract_type="PUT",
            quantity=-1,  # short
            avg_price=5.0,
            status="OPEN",
        )
        session.add(pos)
        session.commit()

        trigger = mgr._check_intraday_drop(pos)
        assert trigger is not None
        assert trigger["type"] == "INTRADAY_DROP"
        assert trigger["severity"] == "CRITICAL"
        assert trigger["action"] == "EMERGENCY_CLOSE"
