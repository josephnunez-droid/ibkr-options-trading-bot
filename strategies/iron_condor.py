"""Iron Condor strategy for range-bound, high-IV underlyings."""

import logging
from datetime import date
from typing import List, Dict
import numpy as np
from strategies.base import BaseStrategy
from db.models import get_session, Position

logger = logging.getLogger(__name__)


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor on high-IV, range-bound underlyings.

    Rules:
    - Wings at ~1 std dev OTM on each side
    - 30-45 DTE
    - Only enter when VIX >= 18
    - Close at 1/3 max profit
    """

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        super().__init__("iron_condor", config, data_feed, executor, risk_engine)
        self.min_dte = self.strategy_config.get("min_dte", 30)
        self.max_dte = self.strategy_config.get("max_dte", 45)
        self.wing_std_dev = self.strategy_config.get("wing_std_dev", 1.0)
        self.min_vix = self.strategy_config.get("min_vix", 18)
        self.profit_target_pct = self.strategy_config.get("profit_target_pct", 0.33)
        self.target_symbols = self.strategy_config.get("symbols", ["SPY", "QQQ", "IWM"])

    def scan(self, universe: List[str]) -> List[Dict]:
        """Scan for iron condor opportunities."""
        signals = []
        if not self.data_feed:
            return signals

        vix = self.data_feed.get_vix()
        if vix < self.min_vix:
            logger.info(f"VIX {vix:.1f} < {self.min_vix} — skipping iron condors")
            return signals

        candidates = [s for s in self.target_symbols if s in universe]

        for symbol in candidates:
            if not self._check_earnings(symbol, 5):
                continue

            session = get_session()
            existing = session.query(Position).filter_by(
                symbol=symbol, strategy="iron_condor", status="OPEN"
            ).count()
            session.close()
            if existing > 0:
                continue

            iv_rank = self.data_feed.get_iv_rank(symbol)
            price = self.data_feed.get_current_price(symbol)

            hist = self.data_feed.get_historical_data(symbol, "30 D", "1 day")
            if hist.empty:
                continue

            close_col = "Close" if "Close" in hist.columns else "close"
            returns = np.log(hist[close_col] / hist[close_col].shift(1)).dropna()
            daily_std = returns.std()
            period_std = daily_std * np.sqrt(35)
            std_move = price * period_std * self.wing_std_dev

            signals.append({
                "symbol": symbol,
                "price": price,
                "std_move": std_move,
                "iv_rank": iv_rank,
                "vix": vix,
                "score": iv_rank + vix,
                "reason": f"VIX={vix:.1f}, IV rank={iv_rank:.0f}, 1s=${std_move:.2f}",
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

    def enter(self, signal: Dict) -> bool:
        """Place iron condor."""
        if not self.data_feed or not self.executor:
            return False

        symbol = signal["symbol"]
        price = signal["price"]
        std_move = signal["std_move"]

        put_short_strike = self._round_strike(price - std_move)
        put_long_strike = self._round_strike(put_short_strike - self._wing_width(price))
        call_short_strike = self._round_strike(price + std_move)
        call_long_strike = self._round_strike(call_short_strike + self._wing_width(price))

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
        put_short = min(strikes, key=lambda s: abs(s - put_short_strike))
        put_long = min(strikes, key=lambda s: abs(s - put_long_strike))
        call_short = min(strikes, key=lambda s: abs(s - call_short_strike))
        call_long = min(strikes, key=lambda s: abs(s - call_long_strike))

        puts = chain[chain["right"] == "P"]
        calls = chain[chain["right"] == "C"]

        ps_row = puts[puts["strike"] == put_short]
        pl_row = puts[puts["strike"] == put_long]
        cs_row = calls[calls["strike"] == call_short]
        cl_row = calls[calls["strike"] == call_long]

        if ps_row.empty or pl_row.empty or cs_row.empty or cl_row.empty:
            logger.warning(f"Could not find all IC strikes for {symbol}")
            return False

        credit = (
            ps_row["mid"].values[0] - pl_row["mid"].values[0] +
            cs_row["mid"].values[0] - cl_row["mid"].values[0]
        )

        if credit <= 0:
            logger.warning(f"Negative credit for {symbol} IC: ${credit:.2f}")
            return False

        max_loss = max(
            (put_short - put_long) * 100 - credit * 100,
            (call_long - call_short) * 100 - credit * 100,
        )
        account = self.data_feed.get_account_values()
        contracts = (
            self.risk_engine.calculate_position_size(
                "iron_condor", credit, max_loss, account_summary=account
            ) if self.risk_engine else 1
        )

        if contracts <= 0:
            return False

        legs = [
            {"expiry": expiry_str, "strike": put_long, "right": "P", "action": "BUY"},
            {"expiry": expiry_str, "strike": put_short, "right": "P", "action": "SELL"},
            {"expiry": expiry_str, "strike": call_short, "right": "C", "action": "SELL"},
            {"expiry": expiry_str, "strike": call_long, "right": "C", "action": "BUY"},
        ]

        trade = self.executor.place_spread_order(
            symbol=symbol,
            legs=legs,
            action="SELL",
            quantity=contracts,
            net_price=round(credit, 2),
            strategy="iron_condor",
            trade_action="OPEN",
            notes=f"IC: {put_long}/{put_short}P {call_short}/{call_long}C @ ${credit:.2f}",
        )

        if trade:
            logger.info(
                f"Iron Condor: {symbol} {put_long}/{put_short}P "
                f"{call_short}/{call_long}C x{contracts} @ ${credit:.2f}"
            )
            return True

        return False

    def manage(self) -> List[Dict]:
        """Monitor iron condors for management."""
        return []

    def exit(self) -> List[Dict]:
        """Check for profit target exit."""
        exits = []

        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="iron_condor", status="OPEN"
        ).all()
        session.close()

        symbols = set(p.symbol for p in positions)

        for symbol in symbols:
            symbol_positions = [p for p in positions if p.symbol == symbol]
            total_credit = sum(p.avg_price * p.quantity for p in symbol_positions)
            total_current = sum(p.current_price * p.quantity for p in symbol_positions)

            if total_credit != 0:
                profit_pct = (total_credit - total_current) / abs(total_credit)
                if profit_pct >= self.profit_target_pct:
                    exits.append({
                        "type": "CLOSE",
                        "symbol": symbol,
                        "profit_pct": profit_pct,
                        "reason": f"Profit target {profit_pct:.0%} >= {self.profit_target_pct:.0%}",
                        "positions": [p.id for p in symbol_positions],
                    })

        return exits

    def _round_strike(self, price: float) -> float:
        """Round to nearest standard strike."""
        if price < 50:
            return round(price)
        elif price < 200:
            return round(price / 5) * 5
        else:
            return round(price / 10) * 10

    def _wing_width(self, price: float) -> float:
        """Determine wing width based on price."""
        if price < 200:
            return 5.0
        return 10.0
