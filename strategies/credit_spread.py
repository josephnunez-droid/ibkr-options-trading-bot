"""Credit Spread strategy (Bull Put / Bear Call)."""

import logging
from datetime import date
from typing import List, Dict
from strategies.base import BaseStrategy
from db.models import get_session, Position

logger = logging.getLogger(__name__)


class CreditSpreadStrategy(BaseStrategy):
    """
    Directional credit spreads using SMA crossover for bias.

    Rules:
    - Bull put spread when 20 SMA > 50 SMA (bullish)
    - Bear call spread when 20 SMA < 50 SMA (bearish)
    - Width: $5 for stocks < $200, $10 for $200+
    - Max 2% of portfolio risk per spread
    """

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        super().__init__("credit_spread", config, data_feed, executor, risk_engine)
        self.min_dte = self.strategy_config.get("min_dte", 30)
        self.max_dte = self.strategy_config.get("max_dte", 45)
        self.sma_fast = self.strategy_config.get("sma_fast", 20)
        self.sma_slow = self.strategy_config.get("sma_slow", 50)
        self.width_under_200 = self.strategy_config.get("width_under_200", 5)
        self.width_over_200 = self.strategy_config.get("width_over_200", 10)
        self.max_risk_pct = self.strategy_config.get("max_risk_pct", 0.02)

    def scan(self, universe: List[str]) -> List[Dict]:
        """Scan for credit spread opportunities."""
        signals = []
        if not self.data_feed:
            return signals

        for symbol in universe:
            if not self._check_earnings(symbol, 5):
                continue

            session = get_session()
            existing = session.query(Position).filter_by(
                symbol=symbol, strategy="credit_spread", status="OPEN"
            ).count()
            session.close()
            if existing > 0:
                continue

            sma_fast = self.data_feed.get_sma(symbol, self.sma_fast)
            sma_slow = self.data_feed.get_sma(symbol, self.sma_slow)

            if sma_fast is None or sma_slow is None:
                continue

            price = self.data_feed.get_current_price(symbol)
            iv_rank = self.data_feed.get_iv_rank(symbol)

            if sma_fast > sma_slow:
                direction = "BULL"
                spread_type = "Bull Put Spread"
            else:
                direction = "BEAR"
                spread_type = "Bear Call Spread"

            signals.append({
                "symbol": symbol,
                "price": price,
                "direction": direction,
                "spread_type": spread_type,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "iv_rank": iv_rank,
                "score": iv_rank,
                "reason": (
                    f"{spread_type}: SMA{self.sma_fast}=${sma_fast:.2f} "
                    f"vs SMA{self.sma_slow}=${sma_slow:.2f}"
                ),
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

    def enter(self, signal: Dict) -> bool:
        """Place credit spread."""
        if not self.data_feed or not self.executor:
            return False

        symbol = signal["symbol"]
        price = signal["price"]
        direction = signal["direction"]
        width = self.width_under_200 if price < 200 else self.width_over_200

        chain = self.data_feed.get_options_chain(symbol, self.min_dte, self.max_dte)
        if chain.empty:
            return False

        expiry = chain["expiry"].iloc[0]
        expiry_str = (
            expiry.strftime("%Y%m%d")
            if hasattr(expiry, "strftime")
            else str(expiry).replace("-", "")
        )

        strikes = sorted(chain["strike"].unique())

        if direction == "BULL":
            target_short = price * 0.95
            short_strike = min(strikes, key=lambda s: abs(s - target_short))
            long_strike = min(strikes, key=lambda s: abs(s - (short_strike - width)))

            short_row = chain[(chain["strike"] == short_strike) & (chain["right"] == "P")]
            long_row = chain[(chain["strike"] == long_strike) & (chain["right"] == "P")]

            if short_row.empty or long_row.empty:
                return False

            credit = short_row["mid"].values[0] - long_row["mid"].values[0]
            legs = [
                {"expiry": expiry_str, "strike": long_strike, "right": "P", "action": "BUY"},
                {"expiry": expiry_str, "strike": short_strike, "right": "P", "action": "SELL"},
            ]
        else:
            target_short = price * 1.05
            short_strike = min(strikes, key=lambda s: abs(s - target_short))
            long_strike = min(strikes, key=lambda s: abs(s - (short_strike + width)))

            short_row = chain[(chain["strike"] == short_strike) & (chain["right"] == "C")]
            long_row = chain[(chain["strike"] == long_strike) & (chain["right"] == "C")]

            if short_row.empty or long_row.empty:
                return False

            credit = short_row["mid"].values[0] - long_row["mid"].values[0]
            legs = [
                {"expiry": expiry_str, "strike": short_strike, "right": "C", "action": "SELL"},
                {"expiry": expiry_str, "strike": long_strike, "right": "C", "action": "BUY"},
            ]

        if credit <= 0:
            return False

        max_loss = (width - credit) * 100
        account = self.data_feed.get_account_values()
        contracts = (
            self.risk_engine.calculate_position_size(
                "credit_spread", credit, max_loss, account_summary=account
            ) if self.risk_engine else 1
        )

        if contracts <= 0:
            return False

        trade = self.executor.place_spread_order(
            symbol=symbol,
            legs=legs,
            action="SELL",
            quantity=contracts,
            net_price=round(credit, 2),
            strategy="credit_spread",
            trade_action="OPEN",
            notes=f"{signal['spread_type']} ${width} wide @ ${credit:.2f}",
        )

        if trade:
            logger.info(
                f"Credit spread: {signal['spread_type']} {symbol} "
                f"x{contracts} @ ${credit:.2f}"
            )
            return True

        return False

    def manage(self) -> List[Dict]:
        """Monitor credit spreads."""
        return []

    def exit(self) -> List[Dict]:
        """Check for exit: 50% profit target."""
        exits = []

        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="credit_spread", status="OPEN"
        ).all()
        session.close()

        symbols = set(p.symbol for p in positions)
        for symbol in symbols:
            sym_pos = [p for p in positions if p.symbol == symbol]
            total_credit = sum(p.avg_price * p.quantity for p in sym_pos)
            total_current = sum(p.current_price * p.quantity for p in sym_pos)

            if total_credit != 0:
                profit_pct = (total_credit - total_current) / abs(total_credit)
                if profit_pct >= 0.50:
                    exits.append({
                        "type": "CLOSE",
                        "symbol": symbol,
                        "profit_pct": profit_pct,
                        "reason": f"50% profit target: {profit_pct:.0%}",
                    })

        return exits
