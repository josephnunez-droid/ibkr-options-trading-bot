"""Wheel Strategy: CSP -> Assignment -> Covered Call -> Exit."""

import logging
from datetime import date, datetime
from typing import List, Dict
from strategies.base import BaseStrategy
from db.models import get_session, Position, WheelTracker

logger = logging.getLogger(__name__)


class WheelStrategy(BaseStrategy):
    """
    Wheel strategy combining CSPs and covered calls.

    Lifecycle:
    1. Sell CSP -> collect premium
    2. If assigned -> buy shares, track cost basis
    3. Sell covered calls -> collect premium
    4. If called away -> exit with profit
    5. Exit when total credits >= 5% of stock cost
    """

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        super().__init__("wheel", config, data_feed, executor, risk_engine)
        self.exit_profit_pct = self.strategy_config.get("exit_profit_pct", 0.05)
        self.prefer_symbols = self.strategy_config.get("prefer_symbols", [])

    def scan(self, universe: List[str]) -> List[Dict]:
        """Scan for new wheel opportunities or manage existing ones."""
        signals = []
        if not self.data_feed:
            return signals

        session = get_session()
        active_wheels = session.query(WheelTracker).filter(
            WheelTracker.status.in_(["CSP_ACTIVE", "ASSIGNED", "CC_ACTIVE"])
        ).all()
        active_symbols = {w.symbol for w in active_wheels}

        for wheel in active_wheels:
            if wheel.status == "ASSIGNED":
                signals.append({
                    "symbol": wheel.symbol,
                    "wheel_id": wheel.id,
                    "action": "WRITE_CC",
                    "cost_basis": wheel.cost_basis,
                    "premium_collected": wheel.total_premium_collected,
                    "score": 100,
                    "reason": f"Assigned - write covered call (basis: ${wheel.cost_basis:.2f})",
                })

        session.close()

        candidates = [s for s in universe if s in self.prefer_symbols and s not in active_symbols]

        for symbol in candidates:
            csp_cfg = self.config.get("strategies", {}).get("cash_secured_put", {})
            if not self._check_earnings(symbol, csp_cfg.get("earnings_blackout_days", 5)):
                continue
            if not self._check_iv_rank(symbol, csp_cfg.get("min_iv_rank", 40)):
                continue

            iv_rank = self.data_feed.get_iv_rank(symbol)
            signals.append({
                "symbol": symbol,
                "action": "OPEN_CSP",
                "iv_rank": iv_rank,
                "score": iv_rank,
                "reason": f"New wheel entry, IV rank {iv_rank:.0f}",
            })

        return signals

    def enter(self, signal: Dict) -> bool:
        """Execute wheel entry or transition."""
        action = signal.get("action", "OPEN_CSP")

        if action == "OPEN_CSP":
            return self._open_csp(signal)
        elif action == "WRITE_CC":
            return self._write_covered_call(signal)

        return False

    def _open_csp(self, signal: Dict) -> bool:
        """Open a CSP as wheel entry."""
        if not self.data_feed or not self.executor:
            return False

        symbol = signal["symbol"]
        csp_cfg = self.config.get("strategies", {}).get("cash_secured_put", {})
        delta_min = csp_cfg.get("target_delta_min", -0.30)
        delta_max = csp_cfg.get("target_delta_max", -0.20)
        target_delta = (delta_min + delta_max) / 2

        option = self.data_feed.get_option_by_delta(
            symbol, "P", target_delta,
            min_dte=csp_cfg.get("min_dte", 30),
            max_dte=csp_cfg.get("max_dte", 45),
        )
        if not option:
            return False

        expiry = option["expiry"]
        expiry_str = (
            expiry.strftime("%Y%m%d")
            if hasattr(expiry, "strftime")
            else str(expiry).replace("-", "")
        )

        trade = self.executor.place_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=option["strike"],
            right="P",
            action="SELL",
            quantity=1,
            limit_price=round(option["mid"], 2),
            strategy="wheel",
            trade_action="OPEN",
            notes="Wheel CSP entry",
        )

        if trade:
            session = get_session()
            wheel = WheelTracker(
                symbol=symbol,
                status="CSP_ACTIVE",
                cost_basis=option["strike"] * 100,
                total_premium_collected=option["mid"] * 100,
                csp_entries=1,
                current_strike=option["strike"],
                current_expiry=expiry if isinstance(expiry, date) else None,
            )
            session.add(wheel)
            session.commit()
            session.close()

            logger.info(f"Wheel started: {symbol} CSP @ {option['strike']}")
            return True

        return False

    def _write_covered_call(self, signal: Dict) -> bool:
        """Write covered call after assignment."""
        if not self.data_feed or not self.executor:
            return False

        symbol = signal["symbol"]
        cost_basis = signal.get("cost_basis", 0)
        premium_collected = signal.get("premium_collected", 0)

        adjusted_basis = (cost_basis - premium_collected) / 100
        cc_cfg = self.config.get("strategies", {}).get("covered_call", {})
        target_delta = (
            cc_cfg.get("target_delta_min", 0.20) + cc_cfg.get("target_delta_max", 0.35)
        ) / 2

        option = self.data_feed.get_option_by_delta(
            symbol, "C", target_delta,
            min_dte=cc_cfg.get("min_dte", 30),
            max_dte=cc_cfg.get("max_dte", 45),
        )
        if not option:
            return False

        if option["strike"] < adjusted_basis:
            logger.info(
                f"Wheel CC: strike ${option['strike']:.2f} < basis ${adjusted_basis:.2f}, "
                f"looking for higher strike"
            )
            chain = self.data_feed.get_options_chain(
                symbol, cc_cfg.get("min_dte", 30), cc_cfg.get("max_dte", 45), "C"
            )
            if not chain.empty:
                above = chain[chain["strike"] >= adjusted_basis]
                if not above.empty:
                    option = above.iloc[0].to_dict()
                else:
                    return False
            else:
                return False

        expiry = option["expiry"]
        expiry_str = (
            expiry.strftime("%Y%m%d")
            if hasattr(expiry, "strftime")
            else str(expiry).replace("-", "")
        )

        trade = self.executor.place_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=option["strike"],
            right="C",
            action="SELL",
            quantity=1,
            limit_price=round(option["mid"], 2),
            strategy="wheel",
            trade_action="OPEN",
            notes=f"Wheel CC, adjusted basis=${adjusted_basis:.2f}",
        )

        if trade:
            session = get_session()
            wheel = session.query(WheelTracker).filter_by(
                symbol=symbol, status="ASSIGNED"
            ).first()
            if wheel:
                wheel.status = "CC_ACTIVE"
                wheel.total_premium_collected += option["mid"] * 100
                wheel.cc_entries += 1
                wheel.current_strike = option["strike"]
                wheel.current_expiry = expiry if isinstance(expiry, date) else None
                session.commit()
            session.close()

            logger.info(f"Wheel CC: {symbol} @ {option['strike']}")
            return True

        return False

    def manage(self) -> List[Dict]:
        """Manage wheel lifecycle transitions."""
        actions = []

        session = get_session()
        wheels = session.query(WheelTracker).filter(
            WheelTracker.status.in_(["CSP_ACTIVE", "CC_ACTIVE"])
        ).all()

        for wheel in wheels:
            if wheel.status == "CSP_ACTIVE":
                positions = self.data_feed.get_portfolio_positions() if self.data_feed else []
                stock_pos = [
                    p for p in positions
                    if p["symbol"] == wheel.symbol and p["secType"] == "STK" and p["quantity"] >= 100
                ]
                if stock_pos:
                    wheel.status = "ASSIGNED"
                    wheel.shares_held = int(stock_pos[0]["quantity"])
                    session.commit()
                    actions.append({
                        "type": "ASSIGNMENT",
                        "symbol": wheel.symbol,
                        "wheel_id": wheel.id,
                        "reason": f"CSP assigned, {wheel.shares_held} shares",
                    })

            elif wheel.status == "CC_ACTIVE":
                positions = self.data_feed.get_portfolio_positions() if self.data_feed else []
                stock_pos = [
                    p for p in positions
                    if p["symbol"] == wheel.symbol and p["secType"] == "STK"
                ]
                if not stock_pos or stock_pos[0]["quantity"] < 100:
                    profit_pct = (
                        wheel.total_premium_collected / wheel.cost_basis
                        if wheel.cost_basis > 0 else 0
                    )
                    if profit_pct >= self.exit_profit_pct:
                        wheel.status = "COMPLETE"
                        wheel.completed_at = datetime.utcnow()
                    else:
                        wheel.status = "CSP_ACTIVE"
                        wheel.shares_held = 0
                    session.commit()
                    actions.append({
                        "type": "CALLED_AWAY",
                        "symbol": wheel.symbol,
                        "profit_pct": profit_pct,
                    })

        session.close()
        return actions

    def exit(self) -> List[Dict]:
        """Check for wheel completion."""
        exits = []
        session = get_session()
        wheels = session.query(WheelTracker).filter(
            WheelTracker.status != "COMPLETE"
        ).all()

        for wheel in wheels:
            if wheel.cost_basis > 0:
                profit_pct = wheel.total_premium_collected / wheel.cost_basis
                if profit_pct >= self.exit_profit_pct:
                    exits.append({
                        "type": "WHEEL_COMPLETE",
                        "symbol": wheel.symbol,
                        "profit_pct": profit_pct,
                        "total_premium": wheel.total_premium_collected,
                        "reason": (
                            f"Premium collected {profit_pct:.1%} >= "
                            f"{self.exit_profit_pct:.1%} target"
                        ),
                    })

        session.close()
        return exits
