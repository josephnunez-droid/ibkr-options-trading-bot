"""Database models for trade tracking and position management."""

from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Date, Boolean, Text, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class Trade(Base):
    """Record of every order fill."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    action = Column(String(10), nullable=False)  # OPEN, CLOSE, ROLL, HEDGE
    direction = Column(String(4), nullable=False)  # BUY, SELL
    contract_type = Column(String(10), nullable=False)  # STOCK, CALL, PUT
    strike = Column(Float, nullable=True)
    expiry = Column(Date, nullable=True)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    order_id = Column(Integer, nullable=True)
    fill_id = Column(String(50), nullable=True)
    pnl = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    def __repr__(self):
        return (f"<Trade {self.id}: {self.direction} {self.quantity} "
                f"{self.symbol} {self.contract_type} @ {self.price}>")


class Position(Base):
    """Current open positions."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    contract_type = Column(String(10), nullable=False)
    strike = Column(Float, nullable=True)
    expiry = Column(Date, nullable=True)
    quantity = Column(Integer, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    delta = Column(Float, default=0.0)
    theta = Column(Float, default=0.0)
    vega = Column(Float, default=0.0)
    gamma = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    status = Column(String(10), default="OPEN")  # OPEN, CLOSED

    def __repr__(self):
        return (f"<Position {self.id}: {self.quantity} {self.symbol} "
                f"{self.contract_type} {self.status}>")


class DailyPnL(Base):
    """Daily portfolio snapshot."""
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    portfolio_value = Column(Float, default=0.0)
    margin_used = Column(Float, default=0.0)
    excess_margin = Column(Float, default=0.0)
    delta = Column(Float, default=0.0)
    theta = Column(Float, default=0.0)
    vega = Column(Float, default=0.0)
    gamma = Column(Float, default=0.0)
    positions_count = Column(Integer, default=0)
    trades_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<DailyPnL {self.date}: ${self.total_pnl:+.2f}>"


class WheelTracker(Base):
    """Track wheel strategy lifecycle per symbol."""
    __tablename__ = "wheel_tracker"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    status = Column(String(20), default="CSP_ACTIVE")
    # CSP_ACTIVE, ASSIGNED, CC_ACTIVE, COMPLETE
    cost_basis = Column(Float, default=0.0)
    total_premium_collected = Column(Float, default=0.0)
    shares_held = Column(Integer, default=0)
    csp_entries = Column(Integer, default=0)
    cc_entries = Column(Integer, default=0)
    current_strike = Column(Float, nullable=True)
    current_expiry = Column(Date, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Wheel {self.symbol}: {self.status}>"


class AlertLog(Base):
    """Log of all alerts sent."""
    __tablename__ = "alert_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(10), nullable=False)  # INFO, WARNING, CRITICAL
    category = Column(String(30), nullable=False)
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)


# --- Database initialization ---

_engine = None
_Session = None


def init_db(db_path: str = "db/trades.db") -> Session:
    """Initialize the database and return a session factory."""
    global _engine, _Session

    _engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Enable WAL mode for better concurrent access
    @event.listens_for(_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine)

    logger.info(f"Database initialized at {db_path}")
    return _Session


def get_session() -> Session:
    """Get a new database session."""
    if _Session is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _Session()
