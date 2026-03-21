"""IBKR TWS/Gateway connection manager with auto-reconnect."""

import time
import logging
import socket
from typing import Optional
from ib_insync import IB, util
import yaml

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages connection to Interactive Brokers TWS or IB Gateway."""

    def __init__(self, config: dict):
        self.config = config["connection"]
        self.ib = IB()
        self._connected = False
        self._reconnect_count = 0

        # Determine port based on trading mode
        if self.config["trading_mode"] == "live":
            self.port = self.config["live_port"]
        else:
            self.port = self.config["paper_port"]

        self.host = self.config["host"]
        self.client_id = self.config["client_id"]
        self.timeout = self.config["timeout"]
        self.max_reconnects = self.config["max_reconnect_attempts"]
        self.reconnect_delay = self.config["reconnect_delay"]

        # Register disconnect handler (use lambda to keep strong reference)
        self._disconnect_handler = lambda: self._on_disconnect()
        self.ib.disconnectedEvent += self._disconnect_handler

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to TWS/Gateway."""
        return self.ib.isConnected()

    @property
    def trading_mode(self) -> str:
        return self.config["trading_mode"]

    def connect(self) -> bool:
        """Establish connection to TWS/Gateway."""
        if self.is_connected:
            logger.info("Already connected to IBKR")
            return True

        # Detect WSL2 gateway IP if needed
        host = self._resolve_host()

        try:
            self.ib.connect(
                host=host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=False,
            )
            self._connected = True
            self._reconnect_count = 0

            mode = self.config["trading_mode"].upper()
            logger.info(
                f"Connected to IBKR ({mode}) at {host}:{self.port} "
                f"(client {self.client_id})"
            )

            # Log account info
            accounts = self.ib.managedAccounts()
            logger.info(f"Managed accounts: {accounts}")

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Cleanly disconnect from TWS/Gateway."""
        if self.is_connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        while self._reconnect_count < self.max_reconnects:
            self._reconnect_count += 1
            delay = self.reconnect_delay * self._reconnect_count

            logger.warning(
                f"Reconnect attempt {self._reconnect_count}/"
                f"{self.max_reconnects} in {delay}s..."
            )
            time.sleep(delay)

            try:
                self.disconnect()
                if self.connect():
                    return True
            except Exception as e:
                logger.error(f"Reconnect attempt failed: {e}")

        logger.critical(
            f"Failed to reconnect after {self.max_reconnects} attempts"
        )
        return False

    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if self.is_connected:
            return True
        return self.reconnect()

    def _on_disconnect(self):
        """Handle unexpected disconnection."""
        logger.warning("Disconnected from IBKR unexpectedly")
        self._connected = False

    def _resolve_host(self) -> str:
        """Resolve host, auto-detecting WSL2 gateway if needed."""
        host = self.host

        if host == "127.0.0.1":
            # Try localhost first, then WSL2 gateway
            if not self._can_reach(host, self.port):
                wsl_host = self._get_wsl_gateway()
                if wsl_host and self._can_reach(wsl_host, self.port):
                    logger.info(f"Using WSL2 gateway: {wsl_host}")
                    return wsl_host

        return host

    def _get_wsl_gateway(self) -> Optional[str]:
        """Get the WSL2 host gateway IP."""
        try:
            with open("/etc/resolv.conf", "r") as f:
                for line in f:
                    if line.startswith("nameserver"):
                        ip = line.split()[1].strip()
                        logger.debug(f"WSL2 gateway candidate: {ip}")
                        return ip
        except (FileNotFoundError, IndexError):
            pass
        return None

    def _can_reach(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a host:port is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def get_account_summary(self) -> dict:
        """Get key account metrics."""
        if not self.ensure_connected():
            return {}

        summary = {}
        account_values = self.ib.accountSummary()

        keys_of_interest = [
            "NetLiquidation", "TotalCashValue", "BuyingPower",
            "MaintMarginReq", "AvailableFunds", "ExcessLiquidity",
            "GrossPositionValue", "UnrealizedPnL", "RealizedPnL",
        ]

        for av in account_values:
            if av.tag in keys_of_interest and av.currency == "USD":
                summary[av.tag] = float(av.value)

        return summary

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
