"""Unit tests for position sizing (Kelly Criterion)."""

import pytest
from risk_engine import RiskEngine


class TestKellyCriterion:
    """Test Kelly-based position sizing."""

    def test_basic_sizing(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="covered_call",
            premium=3.50,
            max_loss=350.0,
            win_rate=0.70,
            account_summary={
                "NetLiquidation": 230000.0,
                "BuyingPower": 460000.0,
            },
        )
        assert contracts >= 1
        assert isinstance(contracts, int)

    def test_negative_kelly_returns_zero(self, sample_config, test_db):
        """If Kelly is negative, don't take the trade."""
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="covered_call",
            premium=1.00,
            max_loss=500.0,
            win_rate=0.30,
            avg_win=100.0,
            avg_loss=500.0,
            account_summary={
                "NetLiquidation": 230000.0,
                "BuyingPower": 460000.0,
            },
        )
        assert contracts == 0

    def test_csp_respects_buying_power_limit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="cash_secured_put",
            premium=5.00,
            max_loss=15000.0,
            win_rate=0.70,
            account_summary={
                "NetLiquidation": 230000.0,
                "BuyingPower": 100000.0,
            },
        )
        # 20% of $100K = $20K BP limit
        assert contracts >= 1
        assert contracts <= 40

    def test_credit_spread_risk_limit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="credit_spread",
            premium=1.50,
            max_loss=350.0,
            win_rate=0.60,
            account_summary={
                "NetLiquidation": 230000.0,
                "BuyingPower": 460000.0,
            },
        )
        # 2% of $230K = $4,600 max risk, $350/contract = ~13 max
        assert contracts >= 1
        assert contracts <= 13

    def test_debit_spread_spend_limit(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="debit_spread",
            premium=2.00,
            max_loss=200.0,
            win_rate=0.45,
            account_summary={
                "NetLiquidation": 230000.0,
                "BuyingPower": 460000.0,
            },
        )
        # 0.5% of $230K = $1,150 max, $200/contract = ~5 max
        assert contracts >= 0
        assert contracts <= 5

    def test_small_account_returns_minimum(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="covered_call",
            premium=3.50,
            max_loss=350.0,
            win_rate=0.70,
            account_summary={
                "NetLiquidation": 10000.0,
                "BuyingPower": 20000.0,
            },
        )
        assert contracts >= 1

    def test_no_account_summary_returns_one(self, sample_config, test_db):
        engine = RiskEngine(sample_config)

        contracts = engine.calculate_position_size(
            strategy="covered_call",
            premium=3.50,
            max_loss=350.0,
        )
        assert contracts == 1
