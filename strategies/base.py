"""Abstract base class for all trading strategies."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from datetime import date

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Base class for options trading strategies.

    Each strategy implements:
    - scan(): Find new opportunities
    - enter(): Execute entry trades
    - manage(): Monitor and manage open positions
    - exit(): Close positions meeting exit criteria
    """

    def __init__(self, name: str, config: dict, data_feed=None, executor=None, risk_engine=None):
        self.name = name
        self.config = config
        self.strategy_config = config.get("strategies", {}).get(name, {})
        self.data_feed = data_feed
        self.executor = executor
        self.risk_engine = risk_engine
        self.enabled = self.strategy_config.get("enabled", True)

    @abstractmethod
    def scan(self, universe: List[str]) -> List[Dict]:
        """Scan universe for trade opportunities."""
        pass

    @abstractmethod
    def enter(self, signal: Dict) -> bool:
        """Execute entry trade based on a signal."""
        pass

    @abstractmethod
    def manage(self) -> List[Dict]:
        """Check open positions for this strategy."""
        pass

    @abstractmethod
    def exit(self) -> List[Dict]:
        """Check for exit conditions."""
        pass

    def is_enabled(self) -> bool:
        return self.enabled

    def _check_earnings(self, symbol: str, blackout_days: int) -> bool:
        """Returns True if safe (no upcoming earnings within blackout)."""
        if not self.data_feed:
            return True
        days = self.data_feed.days_to_earnings(symbol)
        if days is not None and days <= blackout_days:
            logger.info(f"{symbol} earnings in {days} days - blackout")
            return False
        return True

    def _check_iv_rank(self, symbol: str, min_rank: float) -> bool:
        """Returns True if IV rank meets minimum."""
        if not self.data_feed:
            return True
        rank = self.data_feed.get_iv_rank(symbol)
        if rank < min_rank:
            logger.debug(f"{symbol} IV rank {rank:.0f} < {min_rank}")
            return False
        return True
