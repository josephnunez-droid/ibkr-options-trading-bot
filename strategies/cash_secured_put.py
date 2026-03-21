"""Cash-Secured Put Writer strategy."""

import logging
from datetime import date
from typing import List, Dict
from strategies.base import BaseStrategy
from db.models import get_session, Position

logger = logging.getLogger(__name__)


class CashSecuredPutStrategy(BaseStrategy):
    """
    Writes cash-secured puts on approved universe.

    Rules:
    - Target delta: -0.20 to -0.30
    - Prefer 30-45 DTE
    - Require IV rank >= 40
    - Max 20% of buying power per position
    """

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        super().__init__("cash_secured_put", config, data_feed, executor, risk_engine)
        self.delta_min = self.strategy_config.get("target_delta_min", -0.30)
        self.delta_max = self.strategy_config.get("target_delta_max", -0.20)
        self.min_dte = self.strategy_config.get("min_dte", 30)
        self.max_dte = self.strategy_config.get("max_dte", 45)
        self.min_iv_rank = self.strategy_config.get("min_iv_rank", 40)
        self.max_bp_pct = self.strategy_config.get("max_buying_power_pct", 0.20)
        self.earnings_blackout = self.strategy_config.get("earnings_blackout_days", 5)

    def scan(self, universe: List[str]) -> List[Dict]:
        """Scan universe for CSP opportunities."""
        signals = []
        if not self.data_feed:
            return signals

        for symbol in universe:
            if not self._check_earnings(symbol, self.earnings_blackout):
                continue
            if not self._check_iv_rank(symbol, self.min_iv_rank):
                continue

            session = get_session()
            existing = session.query(Position).filter_by(
                symbol=symbol, strategy="cash_secured_put",
                contract_type="PUT", status="OPEN"
            ).count()
            session.close()

            if existing > 0:
                continue

            iv_rank = self.data_feed.get_iv_rank(symbol)
            price = self.data_feed.get_current_price(symbol)

            signals.append({
                "symbol": symbol,
                "price": price,
                "iv_rank": iv_rank,
                "score": iv_rank,
                "reason": f"IV rank {iv_rank:.0f}, price ${price:.2f}",
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

    def enter(self, signal: Dict) -> bool:
        """Write a cash-secured put."""
        if not self.data_feed or not self.executor or not self.risk_engine:
            return False

        symbol = signal["symbol"]
        target_delta = (self.delta_min + self.delta_max) / 2

        option = self.data_feed.get_option_by_delta(
            symbol, "P", target_delta, self.min_dte, self.max_dte
        )
        if not option:
            logger.warning(f"No suitable put found for {symbol}")
            return False

        premium = option["mid"]
        max_loss = option["strike"] * 100
        account = self.data_feed.get_account_values()
        contracts = self.risk_engine.calculate_position_size(
            "cash_secured_put", premium, max_loss,
            account_summary=account
        )

        if contracts <= 0:
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
            right="P",
            action="SELL",
            quantity=contracts,
            limit_price=round(premium, 2),
            strategy="cash_secured_put",
            trade_action="OPEN",
            notes=f"CSP delta={option['delta']:.3f}, IV rank={signal['iv_rank']:.0f}",
        )

        if trade:
            logger.info(
                f"CSP: SELL {contracts} {symbol} "
                f"{option['strike']}P {expiry_str} @ ${premium:.2f}"
            )
            return True
        return False

    def manage(self) -> List[Dict]:
        """Check open CSPs for management actions."""
        actions = []
        if not self.data_feed:
            return actions

        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="cash_secured_put", contract_type="PUT", status="OPEN"
        ).all()
        session.close()

        for pos in positions:
            if not pos.expiry:
                continue

            dte = (pos.expiry - date.today()).days
            price = self.data_feed.get_current_price(pos.symbol)

            if dte <= 7 and price > pos.strike:
                actions.append({
                    "type": "ROLL",
                    "symbol": pos.symbol,
                    "position_id": pos.id,
                    "reason": f"DTE={dte}, OTM - roll forward",
                })
            elif pos.strike and price <= pos.strike * 1.02:
                actions.append({
                    "type": "ALERT",
                    "symbol": pos.symbol,
                    "position_id": pos.id,
                    "reason": f"Approaching ITM: ${price:.2f} near ${pos.strike:.2f}",
                })

        return actions

    def exit(self) -> List[Dict]:
        """Check for CSP exit conditions."""
        exits = []
        session = get_session()
        positions = session.query(Position).filter_by(
            strategy="cash_secured_put", contract_type="PUT", status="OPEN"
        ).all()
        session.close()

        for pos in positions:
            if pos.avg_price > 0 and pos.current_price > 0:
                profit_pct = (pos.avg_price - pos.current_price) / pos.avg_price
                if profit_pct >= 0.50:
                    exits.append({
                        "type": "CLOSE",
                        "symbol": pos.symbol,
                        "position_id": pos.id,
                        "reason": f"50% profit target: {profit_pct:.0%}",
                    })

        return exits
