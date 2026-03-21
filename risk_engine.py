"""Risk management engine: position sizing, exposure limits, Greeks monitoring."""

import logging
import math
from datetime import datetime, date
from typing import Optional, Dict, Tuple
from db.models import get_session, Position, Trade, DailyPnL

logger = logging.getLogger(__name__)


class RiskEngine:
    """Enforces risk limits and calculates position sizes."""

    def __init__(self, config: dict, data_feed=None):
        self.config = config
        self.risk_cfg = config.get("risk", {})
        self.data_feed = data_feed
        self._daily_pnl_start = None
        self._shutdown = False

        self.max_portfolio_options_pct = self.risk_cfg.get("max_portfolio_options_pct", 0.40)
        self.max_single_name_pct = self.risk_cfg.get("max_single_name_pct", 0.10)
        self.max_daily_loss_pct = self.risk_cfg.get("max_daily_loss_pct", 0.02)
        self.margin_buffer_pct = self.risk_cfg.get("margin_buffer_pct", 0.30)
        self.earnings_blackout_days = self.risk_cfg.get("earnings_blackout_days", 5)
        self.kelly_fraction = self.risk_cfg.get("kelly_fraction", 0.25)
        self.max_position_count = self.risk_cfg.get("max_position_count", 30)

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    def initialize(self, portfolio_value: float):
        """Set starting portfolio value for daily loss tracking."""
        self._daily_pnl_start = portfolio_value
        self._shutdown = False
        logger.info(f"Risk engine initialized. Portfolio: ${portfolio_value:,.2f}")

    def validate_order(
        self,
        symbol: str,
        strategy: str,
        notional_risk: float,
        account_summary: dict,
    ) -> Tuple[bool, str]:
        """
        Validate a potential order against all risk rules.
        Returns (approved, reason).
        """
        if self._shutdown:
            return False, "RISK SHUTDOWN: Daily loss limit breached"

        net_liq = account_summary.get("NetLiquidation", 0)
        if net_liq <= 0:
            return False, "Cannot determine portfolio value"

        # Check position count
        session = get_session()
        open_positions = session.query(Position).filter_by(status="OPEN").count()
        session.close()
        if open_positions >= self.max_position_count:
            return False, f"Max position count ({self.max_position_count}) reached"

        # Check portfolio-level options exposure
        approved, msg = self.check_portfolio_exposure(account_summary)
        if not approved:
            return False, msg

        # Check single-name exposure
        approved, msg = self.check_single_name_exposure(symbol, notional_risk, net_liq)
        if not approved:
            return False, msg

        # Check margin buffer
        approved, msg = self.check_margin_buffer(account_summary)
        if not approved:
            return False, msg

        # Check daily loss
        approved, msg = self.check_daily_loss(account_summary)
        if not approved:
            return False, msg

        # Check earnings blackout
        if strategy in ("cash_secured_put", "covered_call", "iron_condor", "credit_spread"):
            approved, msg = self.check_earnings_blackout(symbol)
            if not approved:
                return False, msg

        return True, "Order approved"

    def check_portfolio_exposure(self, account_summary: dict) -> Tuple[bool, str]:
        """Check total options exposure vs portfolio limit."""
        net_liq = account_summary.get("NetLiquidation", 0)
        gross = account_summary.get("GrossPositionValue", 0)

        if net_liq <= 0:
            return True, "OK"

        options_pct = gross / net_liq
        if options_pct > self.max_portfolio_options_pct:
            return False, (
                f"Portfolio options exposure {options_pct:.1%} exceeds "
                f"limit {self.max_portfolio_options_pct:.1%}"
            )

        return True, "OK"

    def check_single_name_exposure(
        self,
        symbol: str,
        additional_risk: float,
        net_liq: float,
    ) -> Tuple[bool, str]:
        """Check single-name exposure limit."""
        session = get_session()
        positions = session.query(Position).filter_by(
            symbol=symbol, status="OPEN"
        ).all()
        session.close()

        existing_exposure = sum(
            abs(p.quantity * p.avg_price * (100 if p.contract_type in ("CALL", "PUT") else 1))
            for p in positions
        )

        total_exposure = existing_exposure + additional_risk
        max_exposure = net_liq * self.max_single_name_pct

        if total_exposure > max_exposure:
            return False, (
                f"{symbol} exposure ${total_exposure:,.0f} would exceed "
                f"limit ${max_exposure:,.0f} ({self.max_single_name_pct:.0%} of portfolio)"
            )

        return True, "OK"

    def check_daily_loss(self, account_summary: dict) -> Tuple[bool, str]:
        """Check daily loss limit. Triggers shutdown if breached."""
        if self._daily_pnl_start is None:
            return True, "OK"

        net_liq = account_summary.get("NetLiquidation", 0)
        daily_pnl = net_liq - self._daily_pnl_start
        max_loss = self._daily_pnl_start * self.max_daily_loss_pct

        if daily_pnl < -max_loss:
            self._shutdown = True
            logger.critical(
                f"DAILY LOSS LIMIT BREACHED: ${daily_pnl:,.2f} "
                f"(limit: -${max_loss:,.2f}). AUTO-SHUTDOWN ACTIVATED."
            )
            return False, (
                f"Daily loss ${daily_pnl:,.2f} exceeds limit "
                f"-${max_loss:,.2f}. Trading halted."
            )

        return True, "OK"

    def check_margin_buffer(self, account_summary: dict) -> Tuple[bool, str]:
        """Ensure adequate margin buffer."""
        net_liq = account_summary.get("NetLiquidation", 0)
        excess = account_summary.get("ExcessLiquidity", 0)

        if net_liq <= 0:
            return True, "OK"

        margin_pct = excess / net_liq
        if margin_pct < self.margin_buffer_pct:
            return False, (
                f"Margin buffer {margin_pct:.1%} below minimum "
                f"{self.margin_buffer_pct:.1%}. "
                f"Excess: ${excess:,.0f}"
            )

        return True, "OK"

    def check_earnings_blackout(self, symbol: str) -> Tuple[bool, str]:
        """Check if symbol is in earnings blackout period."""
        if not self.data_feed:
            return True, "OK"

        days = self.data_feed.days_to_earnings(symbol)
        if days is not None and days <= self.earnings_blackout_days:
            return False, (
                f"{symbol} earnings in {days} days — "
                f"blackout period ({self.earnings_blackout_days} days)"
            )

        return True, "OK"

    def calculate_position_size(
        self,
        strategy: str,
        premium: float,
        max_loss: float,
        win_rate: float = 0.65,
        avg_win: float = None,
        avg_loss: float = None,
        account_summary: dict = None,
    ) -> int:
        """
        Calculate number of contracts using Kelly Criterion.
        Kelly % = W - (1-W)/R where W=win rate, R=win/loss ratio.
        Uses fractional Kelly (quarter-Kelly by default) for safety.
        """
        if not account_summary:
            return 1

        net_liq = account_summary.get("NetLiquidation", 0)
        if net_liq <= 0:
            return 1

        # (default_win_rate, avg_win_mult, avg_loss_mult) — all relative to premium*100
        strategy_defaults = {
            "covered_call": (0.70, 1.0, 1.5),
            "cash_secured_put": (0.70, 1.0, 1.5),
            "wheel": (0.65, 1.0, 1.5),
            "iron_condor": (0.75, 1.0, 2.0),
            "credit_spread": (0.60, 1.0, 1.0),
            "debit_spread": (0.45, 2.0, 1.0),
        }

        defaults = strategy_defaults.get(strategy, (0.60, 1.0, 1.5))
        if win_rate is None:
            win_rate = defaults[0]
        if avg_win is None:
            avg_win = premium * 100 * defaults[1]
        if avg_loss is None:
            avg_loss = premium * 100 * defaults[2]

        if avg_loss <= 0:
            return 1

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio

        if kelly_pct <= 0:
            logger.warning(f"Negative Kelly for {strategy}: W={win_rate}, R={win_loss_ratio:.2f}")
            return 0

        adj_kelly = kelly_pct * self.kelly_fraction
        risk_budget = net_liq * adj_kelly

        risk_per_contract = max_loss if max_loss > 0 else premium * 100
        if risk_per_contract <= 0:
            return 1

        contracts = max(1, int(risk_budget / risk_per_contract))

        # Strategy-specific caps
        if strategy == "cash_secured_put":
            max_bp = self.config.get("strategies", {}).get(
                "cash_secured_put", {}
            ).get("max_buying_power_pct", 0.20)
            bp = account_summary.get("BuyingPower", 0)
            max_by_bp = int(bp * max_bp / (premium * 100)) if premium > 0 else 1
            contracts = min(contracts, max(1, max_by_bp))

        elif strategy == "credit_spread":
            max_risk_pct = self.config.get("strategies", {}).get(
                "credit_spread", {}
            ).get("max_risk_pct", 0.02)
            max_risk = net_liq * max_risk_pct
            if max_loss > 0:
                contracts = min(contracts, max(1, int(max_risk / max_loss)))

        elif strategy == "debit_spread":
            max_spend_pct = self.config.get("strategies", {}).get(
                "debit_spread", {}
            ).get("max_spend_pct", 0.005)
            max_spend = net_liq * max_spend_pct
            contracts = min(contracts, max(1, int(max_spend / (premium * 100))))

        logger.info(
            f"Position size for {strategy}: {contracts} contracts "
            f"(Kelly={kelly_pct:.2%}, Adj={adj_kelly:.2%}, "
            f"Budget=${risk_budget:,.0f})"
        )

        return contracts

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Aggregate portfolio Greeks from open positions."""
        session = get_session()
        positions = session.query(Position).filter_by(status="OPEN").all()
        session.close()

        greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        for pos in positions:
            multiplier = 100 if pos.contract_type in ("CALL", "PUT") else 1
            qty = pos.quantity
            greeks["delta"] += pos.delta * qty * multiplier
            greeks["gamma"] += pos.gamma * qty * multiplier
            greeks["theta"] += pos.theta * qty * multiplier
            greeks["vega"] += pos.vega * qty * multiplier

        return greeks

    def daily_risk_report(self, account_summary: dict) -> Dict:
        """Generate daily risk metrics report."""
        net_liq = account_summary.get("NetLiquidation", 0)
        excess = account_summary.get("ExcessLiquidity", 0)
        gross = account_summary.get("GrossPositionValue", 0)
        unrealized = account_summary.get("UnrealizedPnL", 0)
        realized = account_summary.get("RealizedPnL", 0)

        greeks = self.get_portfolio_greeks()

        session = get_session()
        open_count = session.query(Position).filter_by(status="OPEN").count()
        session.close()

        report = {
            "timestamp": datetime.now().isoformat(),
            "net_liquidation": net_liq,
            "excess_liquidity": excess,
            "margin_usage_pct": (1 - excess / net_liq) if net_liq > 0 else 0,
            "gross_exposure": gross,
            "options_exposure_pct": gross / net_liq if net_liq > 0 else 0,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "daily_pnl": (net_liq - self._daily_pnl_start) if self._daily_pnl_start else 0,
            "open_positions": open_count,
            "portfolio_delta": greeks["delta"],
            "portfolio_gamma": greeks["gamma"],
            "portfolio_theta": greeks["theta"],
            "portfolio_vega": greeks["vega"],
            "shutdown_active": self._shutdown,
        }

        try:
            session = get_session()
            daily = DailyPnL(
                date=date.today(),
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                total_pnl=realized + unrealized,
                portfolio_value=net_liq,
                margin_used=net_liq - excess,
                excess_margin=excess,
                delta=greeks["delta"],
                theta=greeks["theta"],
                vega=greeks["vega"],
                gamma=greeks["gamma"],
                positions_count=open_count,
            )
            session.merge(daily)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to save daily PnL: {e}")

        return report
