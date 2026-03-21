"""Debit Spread strategy for earnings plays."""

import logging
from datetime import date
from typing import List, Dict
from strategies.base import BaseStrategy
from db.models import get_session, Position

logger = logging.getLogger(__name__)


class DebitSpreadStrategy(BaseStrategy):
    """
    Debit spreads for earnings plays.

    Rules:
    - Enter 5-7 days before earnings
    - ATM or slightly ITM long leg
    - Max 0.5% of portfolio per play
    - Exit 1 day before earnings or on profit target
    """

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        super().__init__("debit_spread", config, data_feed, executor, risk_engine)
        self.days_before = self.strategy_config.get("days_before_earnings", 7)
        self.exit_days = self.strategy_config.get("exit_days_before_earnings", 1)
        self.max_spend_pct = self.strategy_config.get("max_spend_pct", 0.005)

    def scan(self, universe: List[str]) -> List[Dict]:
        """Scan for earnings plays."""
        signals = []
        if not self.data_feed:
            return signals

        for symbol in universe:
            days = self.data_feed.days_to_earnings(symbol)
            if days is None:
                continue

            if not (self.exit_days < days <= self.days_before):
                continue

            session = get_session()
            existing = session.query(Position).filter_by(
                symbol=symbol, strategy="debit_spread", status="OPEN"
            ).count()
            session.close()
            if existing > 0:
                continue

            price = self.data_feed.get_current_price(symbol)
            iv_rank = self.data_feed.get_iv_rank(symbol)

            signals.append({
                "symbol": symbol,
                "price": price,
                "days_to_earnings": days,
                "iv_rank": iv_rank,
                "score": 100 - days,
                "reason": f"Earnings in {days} days, IV rank {iv_rank:.0f}",
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

    def enter(self, signal: Dict) -> bool:
        """Place debit spread for earnings."""
        if not self.data_feed or not self.executor:
            return False

        symbol = signal["symbol"]
        price = signal["price"]

        sma_20 = self.data_feed.get_sma(symbol, 20)
        if sma_20 and price > sma_20:
            right = "C"
            long_strike_target = price
            short_strike_target = price * 1.05
        else:
            right = "P"
            long_strike_target = price
            short_strike_target = price * 0.95

        chain = self.data_feed.get_options_chain(
            symbol, min_dte=5, max_dte=21, right=right
        )
        if chain.empty:
            return False

        earnings_date = self.data_feed.get_earnings_date(symbol)
        if earnings_date:
            valid = chain[chain["expiry"] >= earnings_date]
            if not valid.empty:
                chain = valid

        expiry = chain["expiry"].iloc[0]
        expiry_str = (
            expiry.strftime("%Y%m%d")
            if hasattr(expiry, "strftime")
            else str(expiry).replace("-", "")
        )

        exp_chain = chain[chain["expiry"] == expiry]
        strikes = sorted(exp_chain["strike"].unique())

        long_strike = min(strikes, key=lambda s: abs(s - long_strike_target))
        short_strike = min(strikes, key=lambda s: abs(s - short_strike_target))

        if long_strike == short_strike:
            return False

        long_row = exp_chain[exp_chain["strike"] == long_strike]
        short_row = exp_chain[exp_chain["strike"] == short_strike]

        if long_row.empty or short_row.empty:
            return False

        debit = long_row["mid"].values[0] - short_row["mid"].values[0]
        if debit <= 0:
            return False

        max_loss = debit * 100
        account = self.data_feed.get_account_values()
        contracts = (
            self.risk_engine.calculate_position_size(
                "debit_spread", debit, max_loss, account_summary=account
            ) if self.risk_engine else 1
        )

        if contracts <= 0:
            return False

        if right == "C":
            legs = [
                {"expiry": expiry_str, "strike": long_strike, "right": "C", "action": "BUY"},
                {"expiry": expiry_str, "strike": short_strike, "right": "C", "action": "SELL"},
            ]
        else:
            legs = [
                {"expiry": expiry_str, "strike": long_strike, "right": "P", "action": "BUY"},
                {"expiry": expiry_str, "strike": short_strike, "right": "P", "action": "SELL"},
            ]

        trade = self.executor.place_spread_order(
            symbol=symbol,
            legs=legs,
            action="BUY",
            quantity=contracts,
            net_price=round(debit, 2),
            strategy="debit_spread",
            trade_action="OPEN",
            notes=f"Earnings play: {signal['days_to_earnings']}d out",
        )

        if trade:
            logger.info(
                f"Debit spread: {symbol} {long_strike}/{short_strike}{right} "
                f"x{contracts} @ ${debit:.2f}"
            )
            return True

        return False

    def manage(self) -> List[Dict]:
        """Monitor earnings debit spreads."""
        return []

    def exit(self) -> List[Dict]:
        """Exit before earnings or on profit."""
        exits = []
        if not self.data_feed:
            return exits

        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="debit_spread", status="OPEN"
        ).all()
        session.close()

        symbols = set(p.symbol for p in positions)
        for symbol in symbols:
            days = self.data_feed.days_to_earnings(symbol)

            if days is not None and days <= self.exit_days:
                exits.append({
                    "type": "CLOSE",
                    "symbol": symbol,
                    "reason": f"Earnings in {days} day(s) — exit per rule",
                })
                continue

            sym_pos = [p for p in positions if p.symbol == symbol]
            total_debit = sum(abs(p.avg_price * p.quantity) for p in sym_pos)
            total_current = sum(abs(p.current_price * p.quantity) for p in sym_pos)

            if total_debit > 0:
                profit_pct = (total_current - total_debit) / total_debit
                if profit_pct >= 1.0:
                    exits.append({
                        "type": "CLOSE",
                        "symbol": symbol,
                        "profit_pct": profit_pct,
                        "reason": f"100% profit target: {profit_pct:.0%}",
                    })

        return exits
