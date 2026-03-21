#!/usr/bin/env python3
"""
IBKR Automated Options Trading Bot - Main Entry Point

Usage:
    python main.py                    # Run bot in scheduled mode
    python main.py --dashboard        # Show terminal dashboard only
    python main.py --scan             # Run one-time scan
    python main.py --report           # Generate daily report
    python main.py --kill             # Emergency: cancel all orders & close positions

Default mode is PAPER TRADING. Change in config.yaml to go live.
"""

import sys
import os
import asyncio
import argparse
import logging
import signal
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Ensure an event loop exists before importing ib_insync (required for Python 3.14+)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console
from rich.panel import Panel

from connection import ConnectionManager
from data_feeds import DataFeed
from screener import Screener
from risk_engine import RiskEngine
from executor import Executor
from hedge_manager import HedgeManager
from reporter import Reporter
from db.models import init_db

from strategies.covered_call import CoveredCallStrategy
from strategies.cash_secured_put import CashSecuredPutStrategy
from strategies.wheel import WheelStrategy
from strategies.iron_condor import IronCondorStrategy
from strategies.credit_spread import CreditSpreadStrategy
from strategies.debit_spread import DebitSpreadStrategy

console = Console()

_bot = None


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Configure rotating file and console logging."""
    report_cfg = config.get("reporting", {})
    log_dir = Path(report_cfg.get("logs_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, report_cfg.get("log_level", "INFO"))

    root = logging.getLogger()
    root.setLevel(level)

    file_handler = RotatingFileHandler(
        log_dir / "trading_bot.log",
        maxBytes=report_cfg.get("log_max_bytes", 10_000_000),
        backupCount=report_cfg.get("log_backup_count", 5),
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
    ))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console_handler)

    logging.getLogger("ib_insync").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.conn = ConnectionManager(config)
        self.scheduler = BlockingScheduler()
        self._running = False

        self.data_feed = None
        self.screener = None
        self.risk_engine = None
        self.executor = None
        self.hedge_mgr = None
        self.reporter = None
        self.strategies = {}
        self.universe = []

    def initialize(self) -> bool:
        """Connect to IBKR and initialize all components."""
        if self.config["connection"]["trading_mode"] == "live":
            console.print("[bold red]WARNING: LIVE TRADING MODE[/bold red]")
            console.print("You are about to trade with REAL MONEY.")
            confirm = input("Type 'CONFIRM LIVE' to proceed: ")
            if confirm != "CONFIRM LIVE":
                console.print("Aborted. Switch to paper mode in config.yaml.")
                return False

        if not self.conn.connect():
            logger.error("Failed to connect to IBKR")
            return False

        db_path = self.config.get("reporting", {}).get("db_path", "db/trades.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        init_db(db_path)

        ib = self.conn.ib
        self.data_feed = DataFeed(ib, self.config)
        self.screener = Screener(self.config, self.data_feed)
        self.risk_engine = RiskEngine(self.config, self.data_feed)
        self.executor = Executor(ib, self.config, self.risk_engine)
        self.hedge_mgr = HedgeManager(
            self.config, self.data_feed, self.executor, self.risk_engine
        )
        self.reporter = Reporter(self.config, self.data_feed, self.risk_engine)

        account = self.conn.get_account_summary()
        net_liq = account.get("NetLiquidation", 0)
        self.risk_engine.initialize(net_liq)

        strategy_args = dict(
            config=self.config,
            data_feed=self.data_feed,
            executor=self.executor,
            risk_engine=self.risk_engine,
        )

        self.strategies = {
            "covered_call": CoveredCallStrategy(**strategy_args),
            "cash_secured_put": CashSecuredPutStrategy(**strategy_args),
            "wheel": WheelStrategy(**strategy_args),
            "iron_condor": IronCondorStrategy(**strategy_args),
            "credit_spread": CreditSpreadStrategy(**strategy_args),
            "debit_spread": DebitSpreadStrategy(**strategy_args),
        }

        mode = self.config["connection"]["trading_mode"].upper()
        console.print(f"\n[bold green]Bot initialized ({mode} mode)[/bold green]")
        console.print(f"  Net Liquidation: ${net_liq:,.2f}")
        enabled = [s for s, obj in self.strategies.items() if obj.is_enabled()]
        console.print(f"  Strategies: {', '.join(enabled)}")

        return True

    # ---- Scheduled Tasks ----

    def pre_market_scan(self):
        """9:00 AM ET - Screen universe, check IV, earnings."""
        logger.info("=" * 60)
        logger.info("PRE-MARKET SCAN")
        logger.info("=" * 60)

        if not self.conn.ensure_connected():
            return

        self.universe = self.screener.build_universe()
        logger.info(f"Universe: {len(self.universe)} symbols")

        self.hedge_mgr.initialize()

        for name, strategy in self.strategies.items():
            if not strategy.is_enabled():
                continue
            try:
                signals = strategy.scan(self.universe)
                logger.info(f"[{name}] {len(signals)} signals found")
                for sig in signals[:5]:
                    logger.info(f"  {sig['symbol']}: {sig.get('reason', '')}")
            except Exception as e:
                logger.error(f"[{name}] scan error: {e}")

    def market_open_execute(self):
        """9:35 AM ET - Execute entries from scan signals."""
        logger.info("=" * 60)
        logger.info("MARKET OPEN - EXECUTING ENTRIES")
        logger.info("=" * 60)

        if not self.conn.ensure_connected():
            return

        if self.risk_engine.is_shutdown:
            logger.warning("Risk shutdown active - no new entries")
            return

        if self.hedge_mgr.is_paused:
            logger.warning("Trading paused (VIX spike) - no new entries")
            return

        for name, strategy in self.strategies.items():
            if not strategy.is_enabled():
                continue

            try:
                signals = strategy.scan(self.universe)
                for signal in signals[:3]:
                    try:
                        success = strategy.enter(signal)
                        if success:
                            self.reporter.send_alert(
                                f"Entry: {name} - {signal['symbol']} "
                                f"({signal.get('reason', '')})",
                                level="INFO",
                                category="fills",
                            )
                    except Exception as e:
                        logger.error(f"[{name}] entry error for {signal['symbol']}: {e}")
            except Exception as e:
                logger.error(f"[{name}] execution error: {e}")

    def midday_check(self):
        """12:00 PM ET - Review positions, check hedges."""
        logger.info("=" * 60)
        logger.info("MIDDAY CHECK")
        logger.info("=" * 60)

        if not self.conn.ensure_connected():
            return

        positions = self.executor.get_positions()
        self.executor.update_position_db(positions)

        account = self.conn.get_account_summary()
        self.risk_engine.check_daily_loss(account)

        triggers = self.hedge_mgr.check_all_hedges()
        for trigger in triggers:
            self.reporter.send_alert(
                trigger["message"],
                level=trigger["severity"],
                category="hedge_triggers",
            )
            if trigger["severity"] == "CRITICAL":
                self.hedge_mgr.execute_hedge(trigger)

        for name, strategy in self.strategies.items():
            if not strategy.is_enabled():
                continue
            try:
                actions = strategy.manage()
                for action in actions:
                    logger.info(f"[{name}] {action['type']}: {action.get('reason', '')}")
            except Exception as e:
                logger.error(f"[{name}] manage error: {e}")

    def power_hour(self):
        """3:30 PM ET - Evaluate rolls and closes."""
        logger.info("=" * 60)
        logger.info("POWER HOUR - ROLLS & CLOSES")
        logger.info("=" * 60)

        if not self.conn.ensure_connected():
            return

        for name, strategy in self.strategies.items():
            if not strategy.is_enabled():
                continue

            try:
                exits = strategy.exit()
                for exit_sig in exits:
                    logger.info(
                        f"[{name}] EXIT: {exit_sig.get('symbol', '')} "
                        f"- {exit_sig.get('reason', '')}"
                    )
                    self.reporter.send_alert(
                        f"Exit signal: {name} - {exit_sig.get('symbol', '')} "
                        f"({exit_sig.get('reason', '')})",
                        level="INFO",
                        category="fills",
                    )

                actions = strategy.manage()
                for action in actions:
                    if action["type"] == "ROLL" and hasattr(strategy, "roll"):
                        strategy.roll(action.get("position_id"))
            except Exception as e:
                logger.error(f"[{name}] power hour error: {e}")

    def after_hours_report(self):
        """4:30 PM ET - Generate daily report."""
        logger.info("=" * 60)
        logger.info("AFTER HOURS - DAILY REPORT")
        logger.info("=" * 60)

        if not self.conn.ensure_connected():
            return

        account = self.conn.get_account_summary()
        self.risk_engine.daily_risk_report(account)

        report_path = self.reporter.generate_daily_report(account)
        logger.info(f"Daily report: {report_path}")

        net_liq = account.get("NetLiquidation", 0)
        daily_pnl = net_liq - (self.risk_engine._daily_pnl_start or net_liq)
        self.reporter.send_alert(
            f"Daily Summary: P&L ${daily_pnl:+,.2f} | NAV ${net_liq:,.2f}",
            level="INFO",
            category="fills",
        )

    # ---- Modes ----

    def run_scheduled(self):
        """Run bot with scheduled tasks."""
        tz = self.config.get("schedule", {}).get("timezone", "US/Eastern")

        schedules = {
            "pre_market_scan": ("09", "00"),
            "market_open_execute": ("09", "35"),
            "midday_check": ("12", "00"),
            "power_hour": ("15", "30"),
            "after_hours_report": ("16", "30"),
        }

        for task_name, (hour, minute) in schedules.items():
            override = self.config.get("schedule", {}).get(task_name, f"{hour}:{minute}")
            h, m = override.split(":")
            self.scheduler.add_job(
                getattr(self, task_name),
                CronTrigger(
                    hour=int(h), minute=int(m),
                    day_of_week="mon-fri",
                    timezone=tz,
                ),
                id=task_name,
                name=task_name,
            )
            logger.info(f"Scheduled: {task_name} at {h}:{m} ET (Mon-Fri)")

        self._running = True
        console.print("\n[bold green]Bot is running. Press Ctrl+C to stop.[/bold green]\n")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()

    def run_scan_once(self):
        """Run a single scan cycle."""
        self.pre_market_scan()
        self.reporter.display_dashboard(self.conn.get_account_summary())

    def run_dashboard(self):
        """Display terminal dashboard."""
        account = self.conn.get_account_summary()
        self.reporter.display_dashboard(account)

    def run_report(self):
        """Generate daily report."""
        account = self.conn.get_account_summary()
        path = self.reporter.generate_daily_report(account)
        console.print(f"[green]Report saved: {path}[/green]")

    def kill_switch(self):
        """Emergency: cancel all orders and close all positions."""
        console.print("[bold red]KILL SWITCH ACTIVATED[/bold red]")
        console.print("This will cancel ALL open orders and close ALL positions.")
        confirm = input("Type 'KILL' to confirm: ")
        if confirm != "KILL":
            console.print("Aborted.")
            return

        self.executor.cancel_all_orders()
        self.executor.close_all_positions()

        self.reporter.send_alert(
            "KILL SWITCH: All orders cancelled, positions closing at market",
            level="CRITICAL",
            category="errors",
        )
        console.print("[bold red]Kill switch executed.[/bold red]")

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False

        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

        self.conn.disconnect()
        console.print("[yellow]Bot stopped.[/yellow]")


def handle_signal(signum, frame):
    """Handle SIGINT/SIGTERM."""
    if _bot:
        _bot.shutdown()
    sys.exit(0)


def main():
    global _bot

    parser = argparse.ArgumentParser(
        description="IBKR Automated Options Trading Bot"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dashboard", action="store_true", help="Show terminal dashboard")
    parser.add_argument("--scan", action="store_true", help="Run one-time scan")
    parser.add_argument("--report", action="store_true", help="Generate daily report")
    parser.add_argument("--kill", action="store_true", help="Emergency kill switch")

    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"

    config = load_config(str(config_path))

    # Each mode gets a unique client ID so they can run simultaneously
    client_id_offsets = {"dashboard": 1, "scan": 2, "report": 3, "kill": 4}
    for mode, offset in client_id_offsets.items():
        if getattr(args, mode, False):
            config["connection"]["client_id"] = config["connection"].get("client_id", 1) + offset
            break

    setup_logging(config)

    console.print(Panel(
        "[bold cyan]IBKR Automated Options Trading Bot[/bold cyan]\n"
        f"[dim]Mode: {config['connection']['trading_mode'].upper()}[/dim]",
        style="cyan",
    ))

    bot = TradingBot(config)
    _bot = bot

    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    if not bot.initialize():
        sys.exit(1)

    if args.kill:
        bot.kill_switch()
    elif args.dashboard:
        bot.run_dashboard()
    elif args.scan:
        bot.run_scan_once()
    elif args.report:
        bot.run_report()
    else:
        bot.run_scheduled()


if __name__ == "__main__":
    main()
