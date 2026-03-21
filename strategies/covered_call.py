"""Covered Call Writer strategy."""

import logging
from datetime import date, datetime
from typing import List, Dict
from strategies.base import BaseStrategy
from db.models import get_session, Position

logger = logging.getLogger(__name__)


class CoveredCallStrategy(BaseStrategy):
    """
    Writes covered calls on existing stock positions.

    Rules:
    - Target delta: 0.20-0.35 OTM
    - Prefer 30-45 DTE
    - Skip if IV rank < 30 or earnings within 7 days
    - Auto-roll when approaching expiry or assignment risk
    """

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        super().__init__("covered_call", config, data_feed, executor, risk_engine)
        self.delta_min = self.strategy_config.get("target_delta_min", 0.20)
        self.delta_max = self.strategy_config.get("target_delta_max", 0.35)
        self.min_dte = self.strategy_config.get("min_dte", 30)
        self.max_dte = self.strategy_config.get("max_dte", 45)
        self.min_iv_rank = self.strategy_config.get("min_iv_rank", 30)
        self.earnings_blackout = self.strategy_config.get("earnings_blackout_days", 7)
        self.roll_dte_threshold = self.strategy_config.get("roll_dte_threshold", 7)
        self.roll_itm_threshold = self.strategy_config.get("roll_itm_threshold", 0.01)

    def scan(self, universe: List[str]) -> List[Dict]:
        """Find stock positions without covered calls."""
        signals = []
        if not self.data_feed:
            return signals

        positions = self.data_feed.get_portfolio_positions()
        stock_positions = {
            p["symbol"]: p for p in positions
            if p["secType"] == "STK" and p["quantity"] >= 100
        }

        if not stock_positions:
            return signals

        session = get_session()
        existing_calls = session.query(Position).filter_by(
            strategy="covered_call", contract_type="CALL", status="OPEN"
        ).all()
        covered_symbols = {p.symbol for p in existing_calls}
        session.close()

        for symbol, pos in stock_positions.items():
            if symbol not in universe:
                continue
            if symbol in covered_symbols:
                continue
            if not self._check_earnings(symbol, self.earnings_blackout):
                continue
            if not self._check_iv_rank(symbol, self.min_iv_rank):
                continue

            shares = int(pos["quantity"])
            contracts = shares // 100
            if contracts < 1:
                continue

            iv_rank = self.data_feed.get_iv_rank(symbol)
            signals.append({
                "symbol": symbol,
                "contracts": contracts,
                "shares": shares,
                "avg_cost": pos["avgCost"],
                "iv_rank": iv_rank,
                "score": iv_rank,
                "reason": f"Stock position ({shares} shares), IV rank {iv_rank:.0f}",
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

    def enter(self, signal: Dict) -> bool:
        """Write covered call for a signal."""
        if not self.data_feed or not self.executor:
            return False

        symbol = signal["symbol"]
        contracts = signal["contracts"]
        target_delta = (self.delta_min + self.delta_max) / 2

        option = self.data_feed.get_option_by_delta(
            symbol, "C", target_delta, self.min_dte, self.max_dte
        )
        if not option:
            logger.warning(f"No suitable call found for {symbol}")
            return False

        expiry = option["expiry"]
        if hasattr(expiry, "strftime"):
            expiry_str = expiry.strftime("%Y%m%d")
        else:
            expiry_str = str(expiry).replace("-", "")

        trade = self.executor.place_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=option["strike"],
            right="C",
            action="SELL",
            quantity=contracts,
            limit_price=round(option["mid"], 2),
            strategy="covered_call",
            trade_action="OPEN",
            notes=f"CC on {signal['shares']} shares, delta={option['delta']:.3f}",
        )

        if trade:
            logger.info(
                f"Covered call: SELL {contracts} {symbol} "
                f"{option['strike']}C {expiry_str} @ ${option['mid']:.2f}"
            )
            return True
        return False

    def manage(self) -> List[Dict]:
        """Check for roll opportunities on existing covered calls."""
        actions = []
        if not self.data_feed:
            return actions

        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="covered_call", contract_type="CALL", status="OPEN"
        ).all()
        session.close()

        for pos in positions:
            if not pos.expiry:
                continue

            dte = (pos.expiry - date.today()).days
            current_price = self.data_feed.get_current_price(pos.symbol)

            if dte <= self.roll_dte_threshold:
                actions.append({
                    "type": "ROLL",
                    "symbol": pos.symbol,
                    "position_id": pos.id,
                    "reason": f"DTE={dte}, below threshold {self.roll_dte_threshold}",
                    "current_strike": pos.strike,
                })
            elif pos.strike and current_price > pos.strike * (1 + self.roll_itm_threshold):
                actions.append({
                    "type": "ROLL",
                    "symbol": pos.symbol,
                    "position_id": pos.id,
                    "reason": f"ITM: price ${current_price:.2f} > strike ${pos.strike:.2f}",
                    "current_strike": pos.strike,
                })

        return actions

    def exit(self) -> List[Dict]:
        """Check for exit conditions on covered calls."""
        exits = []
        if not self.data_feed:
            return exits

        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="covered_call", contract_type="CALL", status="OPEN"
        ).all()
        session.close()

        for pos in positions:
            if pos.avg_price > 0 and pos.current_price > 0:
                profit_pct = (pos.avg_price - pos.current_price) / pos.avg_price
                if profit_pct >= 0.80:
                    exits.append({
                        "type": "CLOSE",
                        "symbol": pos.symbol,
                        "position_id": pos.id,
                        "reason": f"Profit target hit: {profit_pct:.0%}",
                    })

        return exits

    def roll(self, position_id: int) -> bool:
        """Roll a covered call to next cycle."""
        if not self.data_feed or not self.executor:
            return False

        session = get_session()
        pos = session.query(Position).get(position_id)
        session.close()

        if not pos:
            return False

        expiry_str = pos.expiry.strftime("%Y%m%d") if pos.expiry else ""
        self.executor.place_option_order(
            symbol=pos.symbol,
            expiry=expiry_str,
            strike=pos.strike,
            right="C",
            action="BUY",
            quantity=abs(pos.quantity),
            limit_price=round(pos.current_price * 1.02, 2),
            strategy="covered_call",
            trade_action="ROLL",
            notes="Close leg of roll",
        )

        target_delta = (self.delta_min + self.delta_max) / 2
        new_option = self.data_feed.get_option_by_delta(
            pos.symbol, "C", target_delta, self.min_dte, self.max_dte
        )

        if new_option:
            new_expiry = new_option["expiry"]
            if hasattr(new_expiry, "strftime"):
                new_expiry_str = new_expiry.strftime("%Y%m%d")
            else:
                new_expiry_str = str(new_expiry).replace("-", "")

            self.executor.place_option_order(
                symbol=pos.symbol,
                expiry=new_expiry_str,
                strike=new_option["strike"],
                right="C",
                action="SELL",
                quantity=abs(pos.quantity),
                limit_price=round(new_option["mid"], 2),
                strategy="covered_call",
                trade_action="ROLL",
                notes=f"New leg: {new_option['strike']}C",
            )
            return True

        return False
