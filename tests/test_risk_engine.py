"""Unit tests for risk engine."""

import pytest
from datetime import datetime, date
from risk_engine import RiskEngine
from db.models import get_session, Position


class TestRiskValidation:
    """Test order validation against risk rules."""

    def test_approve_valid_order(self, sample_config, account_summary_healthy, test_db):
        engine = RiskEngine(sample_config)
        engine.initialize(230000.0)

        approved, reason = engine.validate_order(
            symbol="AAPL",
            strategy="covered_call",
            notional_risk=5000,
            account_summary=account_summary_healthy,
        )
        assert approved is True
        assert reason == "Order approved"

    def test_reject_when_shutdown(self, sample_config, account_summary_healthy, test_db):
        engine = RiskEngine(sample_config)
        engine._shutdown = True

        approved, reason = engine.validate_order(
            symbol="AAPL",
            strategy="covered_call",
            notional_risk=5000,
            account_summary=account_summary_healthy,
        )
        assert approved is False
        assert "SHUTDOWN" in reason

    def test_reject_max_positions(self, sample_config, account_summary_healthy, session):
        sample_config["risk"]["max_position_count"] = 2
        engine = RiskEngine(sample_config)
        engine.initialize(230000.0)

        for i in range(2):
            session.add(Position(
                symbol=f"TEST{i}",
                strategy="test",
                contract_type="PUT",
                quantity=-1,
                avg_price=5.0,
                status="OPEN",
            ))
        session.commit()

        approved, reason = engine.validate_order(
            symbol="AAPL",
            strategy="covered_call",
            notional_risk=5000,
            account_summary=account_summary_healthy,
        )
        assert approved is False
        assert "position count" in reason.lower()

    def test_reject_single_name_exposure(self, sample_config, account_summary_healthy, session):
        engine = RiskEngine(sample_config)
        engine.initialize(230000.0)

        session.add(Position(
            symbol="AAPL",
            strategy="test",
            contract_type="PUT",
            quantity=-10,
            avg_price=200.0,
            status="OPEN",
        ))
        session.commit()

        approved, reason = engine.validate_order(
            symbol="AAPL",
            strategy="cash_secured_put",
            notional_risk=5000,
            account_summary=account_summary_healthy,
        )
        assert approved is False
        assert "exposure" in reason.lower()


class TestDailyLoss:
    """Test daily loss limit."""

    def test_no_breach_with_profit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)
        engine.initialize(230000.0)

        account = {"NetLiquidation": 232000.0}
        approved, reason = engine.check_daily_loss(account)
        assert approved is True

    def test_breach_triggers_shutdown(self, sample_config, test_db):
        engine = RiskEngine(sample_config)
        engine.initialize(230000.0)

        account = {"NetLiquidation": 224000.0}  # down $6K > 2% of $230K
        approved, reason = engine.check_daily_loss(account)
        assert approved is False
        assert engine.is_shutdown is True

    def test_no_breach_within_limit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)
        engine.initialize(230000.0)

        account = {"NetLiquidation": 227000.0}  # down $3K < $4,600
        approved, reason = engine.check_daily_loss(account)
        assert approved is True


class TestMarginBuffer:
    """Test margin buffer checks."""

    def test_healthy_margin(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        account = {
            "NetLiquidation": 230000.0,
            "ExcessLiquidity": 100000.0,
        }
        approved, reason = engine.check_margin_buffer(account)
        assert approved is True

    def test_tight_margin(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        account = {
            "NetLiquidation": 230000.0,
            "ExcessLiquidity": 40000.0,  # ~17% < 30%
        }
        approved, reason = engine.check_margin_buffer(account)
        assert approved is False
        assert "margin buffer" in reason.lower()


class TestPortfolioExposure:
    """Test portfolio options exposure limit."""

    def test_within_limit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        account = {
            "NetLiquidation": 230000.0,
            "GrossPositionValue": 80000.0,
        }
        approved, reason = engine.check_portfolio_exposure(account)
        assert approved is True

    def test_exceeds_limit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        account = {
            "NetLiquidation": 230000.0,
            "GrossPositionValue": 100000.0,  # ~43% > 40%
        }
        approved, reason = engine.check_portfolio_exposure(account)
        assert approved is False
