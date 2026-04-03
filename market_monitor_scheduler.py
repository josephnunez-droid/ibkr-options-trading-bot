#!/usr/bin/env python3
"""
Standalone Market Monitor Scheduler

Runs automatically on weekdays, scanning top 100 NYSE stocks multiple times
per trading day. No IBKR connection required — uses yfinance for all data.

Analyzes: Greeks, IV Rank, 20/50/200 day MA, RSI, MACD, Stochastics
Notifies: Desktop notifications, Telegram, and/or email on significant changes.

Usage:
    python market_monitor_scheduler.py              # Run scheduler (stays alive all day)
    python market_monitor_scheduler.py --once       # Run one scan and exit
    python market_monitor_scheduler.py --daemon     # Run as background daemon

Setup (auto-start on weekday mornings):
    # Option 1: cron (runs scheduler at 8:25 AM ET, Mon-Fri)
    crontab -e
    25 8 * * 1-5 cd /path/to/ibkr-options-trading-bot && python market_monitor_scheduler.py

    # Option 2: systemd (see setup_monitor_service.sh)
"""

import logging
import signal
import sys
import os
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from market_monitor import MarketMonitor

# ---- Logging Setup ----

def setup_logging(config: dict):
    log_dir = Path(config.get("reporting", {}).get("logs_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_dir / "market_monitor.log",
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
    ))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console_handler)

    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


logger = logging.getLogger("monitor_scheduler")

# ---- Notification Helpers ----

def send_desktop_notification(title: str, body: str):
    """Send OS desktop notification (Linux/macOS/Windows)."""
    try:
        system = platform.system()
        if system == "Linux":
            subprocess.run(
                ["notify-send", "--urgency=normal", "--app-name=IBKR Monitor", title, body],
                timeout=5, check=False,
            )
        elif system == "Darwin":
            script = f'display notification "{body}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], timeout=5, check=False)
        elif system == "Windows":
            ps = (
                f'[System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms") | Out-Null; '
                f'$n = New-Object System.Windows.Forms.NotifyIcon; '
                f'$n.Icon = [System.Drawing.SystemIcons]::Information; '
                f'$n.Visible = $true; '
                f'$n.ShowBalloonTip(5000, "{title}", "{body}", "Info")'
            )
            subprocess.run(["powershell", "-Command", ps], timeout=5, check=False)
    except Exception as e:
        logger.debug(f"Desktop notification failed: {e}")


def send_telegram(config: dict, message: str):
    """Send Telegram notification."""
    import requests
    tg = config.get("notifications", {}).get("telegram", {})
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


def send_email(config: dict, subject: str, body: str):
    """Send email notification."""
    import smtplib
    from email.mime.text import MIMEText

    email_cfg = config.get("notifications", {}).get("email", {})
    if not email_cfg.get("smtp_server"):
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = email_cfg["sender"]
        msg["To"] = email_cfg["recipient"]
        with smtplib.SMTP(email_cfg["smtp_server"], email_cfg["smtp_port"]) as server:
            server.starttls()
            server.login(email_cfg["sender"], email_cfg["password"])
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Email send failed: {e}")


# ---- Scan Job ----

def run_scheduled_scan(config: dict, session_name: str):
    """Execute a single market monitor scan and dispatch notifications."""
    now = datetime.now()
    logger.info(f"{'='*60}")
    logger.info(f"MARKET MONITOR — {session_name} ({now.strftime('%H:%M ET')})")
    logger.info(f"{'='*60}")

    reports_dir = config.get("reporting", {}).get("reports_dir", "reports")
    monitor = MarketMonitor(config, reports_dir)

    try:
        text_summary, html_report, changes = monitor.run_scan_and_report()
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        send_desktop_notification("IBKR Monitor ERROR", str(e))
        return

    # Log summary header
    for line in text_summary.split("\n")[:5]:
        logger.info(line)
    logger.info(f"Changes detected: {len(changes)}")

    # ---- Dispatch notifications ----
    monitor_cfg = config.get("market_monitor", {})
    notify_cfg = config.get("notifications", {})
    has_changes = len(changes) > 0
    should_notify = (
        monitor_cfg.get("notify_always", False)
        or (monitor_cfg.get("notify_on_changes", True) and has_changes)
    )

    # Desktop notification (always if enabled)
    if monitor_cfg.get("desktop_notifications", True):
        if has_changes:
            top_changes = ", ".join(c["symbol"] for c in changes[:5])
            body = f"{session_name}: {len(changes)} changes — {top_changes}"
            send_desktop_notification("IBKR Market Monitor", body)
        elif monitor_cfg.get("notify_always", False):
            send_desktop_notification("IBKR Market Monitor", f"{session_name} complete — no significant changes.")

    # Telegram / Email (only on changes or if notify_always)
    if should_notify and notify_cfg.get("enabled", False):
        method = notify_cfg.get("method", "telegram")
        short_summary = f"*IBKR Market Monitor — {session_name}*\n"
        if has_changes:
            short_summary += f"{len(changes)} changes detected:\n"
            for c in changes[:15]:
                short_summary += f"• [{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}\n"
        else:
            short_summary += "No significant changes.\n"

        if method == "telegram":
            send_telegram(config, short_summary)
        elif method == "email":
            send_email(config, f"Market Monitor: {session_name}", text_summary)

    # Save extra text report in monitor subdirectory
    monitor_reports = Path(reports_dir) / "monitor"
    monitor_reports.mkdir(parents=True, exist_ok=True)
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    session_slug = session_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    text_path = monitor_reports / f"monitor_{date_str}_{session_slug}_{time_str}.txt"
    text_path.write_text(text_summary)

    logger.info(f"Report saved: {text_path}")
    logger.info(f"{'='*60}\n")


# ---- Scheduler ----

SESSION_NAMES = {
    "08:30": "Pre-Market Overview",
    "10:00": "Post-Open Analysis",
    "12:00": "Midday Check",
    "14:00": "Afternoon Update",
    "15:45": "Closing Summary",
}


def build_scheduler(config: dict) -> BlockingScheduler:
    """Build APScheduler with market monitor scan jobs."""
    scheduler = BlockingScheduler()
    tz = config.get("schedule", {}).get("timezone", "US/Eastern")

    monitor_cfg = config.get("market_monitor", {})
    scan_times = monitor_cfg.get("scan_times", list(SESSION_NAMES.keys()))

    for i, time_str in enumerate(scan_times):
        h, m = time_str.split(":")
        session_name = SESSION_NAMES.get(time_str, f"Scan {time_str}")

        scheduler.add_job(
            run_scheduled_scan,
            CronTrigger(
                hour=int(h), minute=int(m),
                day_of_week="mon-fri",
                timezone=tz,
            ),
            args=[config, session_name],
            id=f"monitor_{i}",
            name=f"market_monitor_{time_str}",
            misfire_grace_time=300,  # Allow 5 min late start
        )
        logger.info(f"Scheduled: {session_name} at {time_str} ET (Mon-Fri)")

    return scheduler


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Market Monitor Scheduler")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon (detach from terminal)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(config)

    # Daemon mode: fork to background
    if args.daemon:
        pid = os.fork()
        if pid > 0:
            print(f"Market monitor daemon started (PID {pid})")
            print(f"Logs: {Path(config.get('reporting', {}).get('logs_dir', 'logs')) / 'market_monitor.log'}")
            sys.exit(0)
        os.setsid()

    # One-shot mode
    if args.once:
        now = datetime.now()
        session = f"Manual Scan ({now.strftime('%H:%M')})"
        run_scheduled_scan(config, session)
        return

    # Scheduled mode
    logger.info("=" * 60)
    logger.info("IBKR MARKET MONITOR SCHEDULER")
    logger.info("Monitoring top 100 NYSE stocks — weekdays only")
    logger.info("Indicators: Greeks, IV Rank, MA(20/50/200), RSI, MACD, Stochastics")
    logger.info("=" * 60)

    scheduler = build_scheduler(config)

    def graceful_shutdown(signum, frame):
        logger.info("Shutting down market monitor scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    logger.info("Scheduler running. Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Market monitor scheduler stopped.")


if __name__ == "__main__":
    main()
