"""Order execution layer: routing, fill tracking, bracket orders."""

import logging
import time
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple
from ib_insync import (
    IB, Stock, Option, LimitOrder, MarketOrder, Order,
    Trade as IBTrade, OrderStatus
)
from db.models import get_session, Trade, Position

logger = logging.getLogger(__name__)


class Executor:
    """Handles order placement, tracking, and position management."""

    def __init__(self, ib: IB, config: dict, risk_engine=None):
        self.ib = ib
        self.config = config
        self.exec_cfg = config.get("execution", {})
        self.risk_engine = risk_engine

        self.order_type = self.exec_cfg.get("order_type", "LMT")
        self.aggressive_retry_seconds = self.exec_cfg.get("aggressive_retry_seconds", 120)
        self.aggressive_offset = self.exec_cfg.get("aggressive_offset", 0.02)
        self.max_retries = self.exec_cfg.get("max_retries", 3)
        self.cancel_timeout = self.exec_cfg.get("cancel_timeout", 300)

        # Register fill handler
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details

        self._pending_orders: Dict[int, dict] = {}

    def place_option_order(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        action: str,
        quantity: int,
        limit_price: float,
        strategy: str,
        trade_action: str = "OPEN",
        notes: str = None,
    ) -> Optional[IBTrade]:
        """
        Place a limit order for an option contract.

        Args:
            symbol: Underlying symbol
            expiry: Expiry date string (YYYYMMDD)
            strike: Strike price
            right: "C" or "P"
            action: "BUY" or "SELL"
            quantity: Number of contracts
            limit_price: Limit price per contract
            strategy: Strategy name for tracking
            trade_action: OPEN, CLOSE, ROLL, HEDGE
            notes: Optional notes
        """
        contract = Option(symbol, expiry, strike, right, "SMART", currency="USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            logger.error(f"Could not qualify option: {symbol} {expiry} {strike}{right}")
            return None

        contract = qualified[0]

        # Risk validation
        if self.risk_engine:
            notional = abs(quantity * limit_price * 100)
            account = (
                self.risk_engine.data_feed.get_account_values()
                if self.risk_engine.data_feed else {}
            )
            approved, reason = self.risk_engine.validate_order(
                symbol, strategy, notional, account
            )
            if not approved:
                logger.warning(f"Order REJECTED by risk engine: {reason}")
                return None

        order = LimitOrder(action, quantity, limit_price)
        order.tif = "DAY"
        order.outsideRth = False

        trade = self.ib.placeOrder(contract, order)

        self._pending_orders[order.orderId] = {
            "symbol": symbol,
            "strategy": strategy,
            "trade_action": trade_action,
            "contract_type": "CALL" if right == "C" else "PUT",
            "strike": strike,
            "expiry": expiry,
            "notes": notes,
            "placed_at": datetime.now(),
        }

        logger.info(
            f"Order placed: {action} {quantity} {symbol} "
            f"{strike}{right} {expiry} @ ${limit_price:.2f} "
            f"[{strategy}/{trade_action}]"
        )

        return trade

    def place_stock_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
        strategy: str,
        trade_action: str = "OPEN",
        use_market: bool = False,
        notes: str = None,
    ) -> Optional[IBTrade]:
        """Place a stock order."""
        contract = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            logger.error(f"Could not qualify stock: {symbol}")
            return None
        contract = qualified[0]

        if use_market:
            order = MarketOrder(action, quantity)
            logger.warning(f"MARKET ORDER: {action} {quantity} {symbol}")
        else:
            order = LimitOrder(action, quantity, limit_price)

        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)

        self._pending_orders[order.orderId] = {
            "symbol": symbol,
            "strategy": strategy,
            "trade_action": trade_action,
            "contract_type": "STOCK",
            "strike": None,
            "expiry": None,
            "notes": notes,
            "placed_at": datetime.now(),
        }

        logger.info(
            f"Stock order: {action} {quantity} {symbol} "
            f"@ {'MKT' if use_market else f'${limit_price:.2f}'} "
            f"[{strategy}/{trade_action}]"
        )

        return trade

    def place_spread_order(
        self,
        symbol: str,
        legs: List[Dict],
        action: str,
        quantity: int,
        net_price: float,
        strategy: str,
        trade_action: str = "OPEN",
        notes: str = None,
    ) -> Optional[IBTrade]:
        """
        Place a combo/spread order.
        legs: List of dicts with keys: expiry, strike, right, action
        """
        from ib_insync import Contract, ComboLeg

        combo_legs = []
        for leg in legs:
            opt = Option(symbol, leg["expiry"], leg["strike"], leg["right"], "SMART")
            qualified = self.ib.qualifyContracts(opt)
            if not qualified:
                logger.error(f"Could not qualify leg: {leg}")
                return None

            combo_leg = ComboLeg(
                conId=qualified[0].conId,
                ratio=1,
                action=leg["action"],
                exchange="SMART",
            )
            combo_legs.append(combo_leg)

        combo = Contract(
            symbol=symbol,
            secType="BAG",
            currency="USD",
            exchange="SMART",
            comboLegs=combo_legs,
        )

        order = LimitOrder(action, quantity, net_price)
        order.tif = "DAY"
        trade = self.ib.placeOrder(combo, order)

        self._pending_orders[order.orderId] = {
            "symbol": symbol,
            "strategy": strategy,
            "trade_action": trade_action,
            "contract_type": "SPREAD",
            "strike": None,
            "expiry": legs[0]["expiry"] if legs else None,
            "notes": notes,
            "placed_at": datetime.now(),
        }

        logger.info(
            f"Spread order: {action} {quantity}x {symbol} {strategy} "
            f"@ ${net_price:.2f} [{trade_action}]"
        )

        return trade

    def wait_for_fill(
        self,
        trade: IBTrade,
        timeout: int = None,
    ) -> bool:
        """Wait for order fill with optional aggressive retry."""
        if timeout is None:
            timeout = self.cancel_timeout

        start = time.time()
        retry_done = False

        while time.time() - start < timeout:
            self.ib.sleep(1)

            if trade.orderStatus.status == "Filled":
                logger.info(f"Order filled: {trade.order.orderId}")
                return True

            if trade.orderStatus.status in ("Cancelled", "ApiCancelled"):
                logger.warning(f"Order cancelled: {trade.order.orderId}")
                return False

            elapsed = time.time() - start
            if (
                not retry_done
                and elapsed > self.aggressive_retry_seconds
                and isinstance(trade.order, LimitOrder)
            ):
                old_price = trade.order.lmtPrice
                if trade.order.action == "BUY":
                    new_price = round(old_price + self.aggressive_offset, 2)
                else:
                    new_price = round(old_price - self.aggressive_offset, 2)

                trade.order.lmtPrice = new_price
                self.ib.placeOrder(trade.contract, trade.order)
                retry_done = True

                logger.info(
                    f"Aggressive retry: {trade.order.orderId} "
                    f"price {old_price} -> {new_price}"
                )

        logger.warning(f"Order timeout, cancelling: {trade.order.orderId}")
        self.ib.cancelOrder(trade.order)
        return False

    def cancel_order(self, order_id: int):
        """Cancel a specific order."""
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {order_id}")
                return
        logger.warning(f"Order {order_id} not found in open trades")

    def cancel_all_orders(self):
        """Cancel all open orders."""
        open_trades = self.ib.openTrades()
        count = len(open_trades)
        for trade in open_trades:
            self.ib.cancelOrder(trade.order)
        self.ib.sleep(2)
        logger.warning(f"Cancelled {count} open orders")

    def close_all_positions(self):
        """Close all options positions at market (emergency)."""
        positions = self.ib.positions()
        closed = 0

        for pos in positions:
            if pos.position == 0:
                continue

            contract = pos.contract
            action = "SELL" if pos.position > 0 else "BUY"
            qty = abs(pos.position)

            order = MarketOrder(action, qty)
            self.ib.placeOrder(contract, order)
            closed += 1

            logger.warning(
                f"EMERGENCY CLOSE: {action} {qty} "
                f"{contract.symbol} {contract.secType}"
            )

        self.ib.sleep(2)
        logger.warning(f"Emergency close: {closed} positions sent to market")

    def get_open_orders(self) -> List[dict]:
        """Get all open orders."""
        orders = []
        for trade in self.ib.openTrades():
            orders.append({
                "orderId": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "secType": trade.contract.secType,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "price": getattr(trade.order, "lmtPrice", "MKT"),
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
            })
        return orders

    def get_positions(self) -> List[dict]:
        """Get all current positions."""
        positions = []
        for pos in self.ib.positions():
            if pos.position == 0:
                continue
            c = pos.contract
            positions.append({
                "symbol": c.symbol,
                "secType": c.secType,
                "strike": getattr(c, "strike", None),
                "right": getattr(c, "right", None),
                "expiry": getattr(c, "lastTradeDateOrContractMonth", None),
                "quantity": pos.position,
                "avgCost": pos.avgCost,
            })
        return positions

    def _on_order_status(self, trade: IBTrade):
        """Handle order status updates."""
        status = trade.orderStatus
        logger.debug(
            f"Order {trade.order.orderId} status: {status.status} "
            f"(filled: {status.filled}, remaining: {status.remaining})"
        )

    def _on_exec_details(self, trade: IBTrade, fill):
        """Record fill in database."""
        order_id = trade.order.orderId
        meta = self._pending_orders.get(order_id, {})

        try:
            session = get_session()
            db_trade = Trade(
                timestamp=datetime.now(),
                symbol=meta.get("symbol", trade.contract.symbol),
                strategy=meta.get("strategy", "unknown"),
                action=meta.get("trade_action", "UNKNOWN"),
                direction=trade.order.action,
                contract_type=meta.get("contract_type", trade.contract.secType),
                strike=meta.get("strike", getattr(trade.contract, "strike", None)),
                expiry=(
                    datetime.strptime(meta["expiry"], "%Y%m%d").date()
                    if meta.get("expiry") else None
                ),
                quantity=int(fill.execution.shares),
                price=fill.execution.price,
                commission=(
                    fill.commissionReport.commission
                    if fill.commissionReport else 0
                ),
                order_id=order_id,
                fill_id=fill.execution.execId,
                notes=meta.get("notes"),
            )
            session.add(db_trade)
            session.commit()
            session.close()

            logger.info(
                f"FILL recorded: {trade.order.action} {fill.execution.shares} "
                f"{trade.contract.symbol} @ ${fill.execution.price:.2f} "
                f"[{meta.get('strategy', '?')}]"
            )

        except Exception as e:
            logger.error(f"Failed to record fill: {e}")

    def update_position_db(self, positions: List[dict]):
        """Sync IBKR positions to database."""
        session = get_session()

        session.query(Position).filter_by(status="OPEN").update(
            {"status": "PENDING_CHECK"}
        )

        for pos in positions:
            existing = session.query(Position).filter_by(
                symbol=pos["symbol"],
                contract_type=pos["secType"],
                strike=pos.get("strike"),
                status="PENDING_CHECK",
            ).first()

            if existing:
                existing.quantity = int(pos["quantity"])
                existing.avg_price = pos["avgCost"]
                existing.status = "OPEN"
            else:
                new_pos = Position(
                    symbol=pos["symbol"],
                    strategy="synced",
                    contract_type=pos["secType"],
                    strike=pos.get("strike"),
                    quantity=int(pos["quantity"]),
                    avg_price=pos["avgCost"],
                    status="OPEN",
                )
                session.add(new_pos)

        session.query(Position).filter_by(status="PENDING_CHECK").update({
            "status": "CLOSED",
            "closed_at": datetime.now(),
        })

        session.commit()
        session.close()
