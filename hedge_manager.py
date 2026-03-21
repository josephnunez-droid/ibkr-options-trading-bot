"""Hedge manager: monitors positions and applies protective hedges."""

import logging
from datetime import datetime
from typing import Optional, List, Dict
from db.models import get_session, Position

logger = logging.getLogger(__name__)


class HedgeManager:
    """Monitors positions and triggers protective hedges."""

    def __init__(self, config: dict, data_feed=None, executor=None, risk_engine=None):
        self.config = config
        self.hedge_cfg = config.get("hedging", {})
        self.data_feed = data_feed
        self.executor = executor
        self.risk_engine = risk_engine
        self.enabled = self.hedge_cfg.get("enabled", True)

        self.delta_threshold = self.hedge_cfg.get("delta_threshold", 0.50)
        self.intraday_drop_pct = self.hedge_cfg.get("intraday_drop_pct", 0.05)
        self.stock_drop_pct = self.hedge_cfg.get("stock_drop_pct", 0.08)
        self.gap_down_pct = self.hedge_cfg.get("gap_down_pct", 0.10)
        self.vix_spike_pct = self.hedge_cfg.get("vix_spike_pct", 0.25)
        self.auto_convert = self.hedge_cfg.get("auto_convert_to_cc", True)

        self._vix_baseline: Optional[float] = None
        self._price_baselines: Dict[str, float] = {}
        self._trading_paused = False

    @property
    def is_paused(self) -> bool:
        return self._trading_paused

    def initialize(self):
        """Set VIX and price baselines at start of day."""
        if not self.data_feed:
            return

        try:
            self._vix_baseline = self.data_feed.get_vix()
            logger.info(f"VIX baseline set: {self._vix_baseline:.2f}")
        except Exception as e:
            logger.error(f"Failed to get VIX baseline: {e}")
            self._vix_baseline = 20.0

        session = get_session()
        positions = session.query(Position).filter_by(status="OPEN").all()
        symbols = set(p.symbol for p in positions)
        session.close()

        for symbol in symbols:
            try:
                price = self.data_feed.get_current_price(symbol)
                self._price_baselines[symbol] = price
            except Exception:
                pass

    def check_all_hedges(self) -> List[Dict]:
        """Run all hedge checks. Returns list of triggered actions."""
        if not self.enabled or not self.data_feed or not self.executor:
            return []

        triggers = []

        vix_trigger = self.check_vix_spike()
        if vix_trigger:
            triggers.append(vix_trigger)

        session = get_session()
        positions = session.query(Position).filter_by(status="OPEN").all()
        session.close()

        for pos in positions:
            if pos.contract_type in ("CALL", "PUT") and pos.quantity < 0:
                trigger = self._check_short_option_hedge(pos)
                if trigger:
                    triggers.append(trigger)

            if pos.contract_type == "STOCK" and pos.quantity > 0:
                trigger = self._check_stock_hedge(pos)
                if trigger:
                    triggers.append(trigger)

            trigger = self._check_intraday_drop(pos)
            if trigger:
                triggers.append(trigger)

        return triggers

    def check_vix_spike(self) -> Optional[Dict]:
        """Check for VIX spike and pause trading if needed."""
        if not self._vix_baseline:
            return None

        try:
            current_vix = self.data_feed.get_vix()
            spike_pct = (current_vix - self._vix_baseline) / self._vix_baseline

            if spike_pct >= self.vix_spike_pct:
                self._trading_paused = True
                logger.critical(
                    f"VIX SPIKE: {self._vix_baseline:.1f} -> {current_vix:.1f} "
                    f"({spike_pct:.1%}). Trading PAUSED."
                )
                return {
                    "type": "VIX_SPIKE",
                    "severity": "CRITICAL",
                    "message": (
                        f"VIX spiked {spike_pct:.1%} "
                        f"({self._vix_baseline:.1f} -> {current_vix:.1f}). "
                        f"New entries paused."
                    ),
                    "action": "PAUSE_TRADING",
                }
            else:
                if self._trading_paused and spike_pct < self.vix_spike_pct * 0.5:
                    self._trading_paused = False
                    logger.info("VIX spike subsided. Trading resumed.")

        except Exception as e:
            logger.error(f"VIX check failed: {e}")

        return None

    def _check_short_option_hedge(self, pos: Position) -> Optional[Dict]:
        """Check if a short option needs delta hedging."""
        if abs(pos.delta) < self.delta_threshold:
            return None

        logger.warning(
            f"HEDGE TRIGGER: {pos.symbol} {pos.contract_type} "
            f"delta={pos.delta:.3f} exceeds threshold {self.delta_threshold}"
        )

        if pos.contract_type == "PUT" and pos.quantity < 0:
            return {
                "type": "DELTA_HEDGE",
                "severity": "WARNING",
                "symbol": pos.symbol,
                "message": (
                    f"Short put {pos.symbol} {pos.strike} delta={pos.delta:.3f} "
                    f"exceeds {self.delta_threshold}. Buy protective put."
                ),
                "action": "BUY_PROTECTIVE_PUT",
                "position_id": pos.id,
            }
        elif pos.contract_type == "CALL" and pos.quantity < 0:
            return {
                "type": "DELTA_HEDGE",
                "severity": "WARNING",
                "symbol": pos.symbol,
                "message": (
                    f"Short call {pos.symbol} {pos.strike} delta={pos.delta:.3f}. "
                    f"Buy protective call."
                ),
                "action": "BUY_PROTECTIVE_CALL",
                "position_id": pos.id,
            }

        return None

    def _check_stock_hedge(self, pos: Position) -> Optional[Dict]:
        """Check if stock position needs protective put."""
        if pos.symbol not in self._price_baselines:
            return None

        try:
            current = self.data_feed.get_current_price(pos.symbol)
            entry = pos.avg_price
            drop_from_entry = (entry - current) / entry

            if drop_from_entry >= self.stock_drop_pct:
                logger.warning(
                    f"HEDGE TRIGGER: {pos.symbol} stock down "
                    f"{drop_from_entry:.1%} from entry"
                )
                return {
                    "type": "STOCK_DROP",
                    "severity": "WARNING",
                    "symbol": pos.symbol,
                    "message": (
                        f"{pos.symbol} down {drop_from_entry:.1%} from "
                        f"entry ${entry:.2f}. Buy protective put."
                    ),
                    "action": "BUY_PROTECTIVE_PUT",
                    "position_id": pos.id,
                }

            baseline = self._price_baselines.get(pos.symbol)
            if baseline:
                gap = (baseline - current) / baseline
                if gap >= self.gap_down_pct:
                    logger.critical(
                        f"GAP DOWN: {pos.symbol} gapped {gap:.1%}"
                    )
                    return {
                        "type": "GAP_DOWN",
                        "severity": "CRITICAL",
                        "symbol": pos.symbol,
                        "message": (
                            f"{pos.symbol} gapped down {gap:.1%}. "
                            f"Close covered call and evaluate exit."
                        ),
                        "action": "EVALUATE_EXIT",
                        "position_id": pos.id,
                    }

        except Exception as e:
            logger.error(f"Stock hedge check failed for {pos.symbol}: {e}")

        return None

    def _check_intraday_drop(self, pos: Position) -> Optional[Dict]:
        """Check for intraday drops triggering emergency stop."""
        if pos.symbol not in self._price_baselines:
            return None

        try:
            current = self.data_feed.get_current_price(pos.symbol)
            baseline = self._price_baselines[pos.symbol]
            drop = (baseline - current) / baseline

            if drop >= self.intraday_drop_pct and pos.quantity < 0:
                logger.critical(
                    f"EMERGENCY: {pos.symbol} down {drop:.1%} intraday "
                    f"with short position"
                )
                return {
                    "type": "INTRADAY_DROP",
                    "severity": "CRITICAL",
                    "symbol": pos.symbol,
                    "message": (
                        f"{pos.symbol} down {drop:.1%} intraday. "
                        f"Emergency stop on short position."
                    ),
                    "action": "EMERGENCY_CLOSE",
                    "position_id": pos.id,
                }

        except Exception:
            pass

        return None

    def execute_hedge(self, trigger: Dict):
        """Execute a hedge action based on trigger."""
        action = trigger.get("action")
        symbol = trigger.get("symbol")

        if action == "PAUSE_TRADING":
            self._trading_paused = True
            logger.warning("Trading paused due to hedge trigger")

        elif action == "BUY_PROTECTIVE_PUT" and symbol:
            self._buy_protective_option(symbol, "P")

        elif action == "BUY_PROTECTIVE_CALL" and symbol:
            self._buy_protective_option(symbol, "C")

        elif action == "EMERGENCY_CLOSE":
            self._emergency_close_position(trigger.get("position_id"))

        elif action == "EVALUATE_EXIT":
            logger.warning(
                f"MANUAL REVIEW NEEDED: {symbol} - {trigger.get('message')}"
            )

    def _buy_protective_option(self, symbol: str, right: str):
        """Buy a protective option for hedging."""
        if not self.data_feed or not self.executor:
            return

        try:
            target_delta = 0.50 if right == "C" else -0.50

            option = self.data_feed.get_option_by_delta(
                symbol, right, target_delta, min_dte=14, max_dte=30
            )

            if option:
                expiry = option["expiry"]
                if hasattr(expiry, "strftime"):
                    expiry_str = expiry.strftime("%Y%m%d")
                else:
                    expiry_str = str(expiry).replace("-", "")

                self.executor.place_option_order(
                    symbol=symbol,
                    expiry=expiry_str,
                    strike=option["strike"],
                    right=right,
                    action="BUY",
                    quantity=1,
                    limit_price=option["ask"],
                    strategy="hedge",
                    trade_action="HEDGE",
                    notes=f"Protective {right} hedge for {symbol}",
                )
                logger.info(
                    f"Protective {right} ordered for {symbol} "
                    f"@ {option['strike']} exp {option['expiry']}"
                )

        except Exception as e:
            logger.error(f"Failed to place protective option for {symbol}: {e}")

    def _emergency_close_position(self, position_id: int):
        """Emergency close a specific position using market order."""
        if not self.executor:
            return

        session = get_session()
        pos = session.query(Position).get(position_id)
        session.close()

        if not pos:
            return

        action = "BUY" if pos.quantity < 0 else "SELL"
        qty = abs(pos.quantity)

        if pos.contract_type in ("CALL", "PUT"):
            expiry_str = pos.expiry.strftime("%Y%m%d") if pos.expiry else ""
            right = "C" if pos.contract_type == "CALL" else "P"
            self.executor.place_option_order(
                symbol=pos.symbol,
                expiry=expiry_str,
                strike=pos.strike,
                right=right,
                action=action,
                quantity=qty,
                limit_price=pos.current_price * 1.05,
                strategy="hedge",
                trade_action="HEDGE",
                notes="Emergency close",
            )
        else:
            self.executor.place_stock_order(
                symbol=pos.symbol,
                action=action,
                quantity=qty,
                limit_price=0,
                strategy="hedge",
                trade_action="HEDGE",
                use_market=True,
                notes="Emergency close",
            )

        logger.critical(f"EMERGENCY CLOSE sent for position {position_id}")
