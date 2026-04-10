#!/usr/bin/env python3
"""
Scheduled Market Monitor — Standalone weekday daemon.

Runs the top 100 NYSE stock scanner multiple times per trading day
(Mon-Fri) and sends notifications via Telegram and/or email when
significant changes are detected.

NO IBKR connection required — uses yfinance for all market data.

Usage:
    python scheduled_monitor.py              # Start the scheduled daemon
    python scheduled_monitor.py --once       # Run one scan and exit
    python scheduled_monitor.py --test-notify  # Send a test notification

Configure scan times and notification settings in config.yaml under
the 'market_monitor' and 'notifications' sections.
"""

import argparse
import logging
import signal
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from market_monitor import MarketMonitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config: dict):
    """Configure rotating file + console logging."""
    report_cfg = config.get("reporting", {})
    log_dir = Path(report_cfg.get("logs_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_dir / "market_monitor.log",
        maxBytes=report_cfg.get("log_max_bytes", 10_000_000),
        backupCount=report_cfg.get("log_backup_count", 5),
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
    ))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    ))
    root.addHandler(console_handler)

    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


logger = logging.getLogger("scheduled_monitor")

# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

def send_telegram(token: str, chat_id: str, text: str, html: str = None):
    """Send a Telegram message. Falls back to truncated text for long messages."""
    if not token or not chat_id:
        logger.warning("Telegram not configured (missing bot_token or chat_id)")
        return False

    # Telegram max message length is 4096 chars
    MAX_LEN = 4000
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN] + "\n\n... (truncated, see full report in reports/)"

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        if resp.status_code == 200:
            logger.info("Telegram notification sent")
            return True
        else:
            logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def send_email(smtp_cfg: dict, subject: str, text_body: str, html_body: str = None):
    """Send an email notification with optional HTML attachment."""
    server = smtp_cfg.get("smtp_server", "")
    port = smtp_cfg.get("smtp_port", 587)
    sender = smtp_cfg.get("sender", "")
    password = smtp_cfg.get("password", "")
    recipient = smtp_cfg.get("recipient", "")

    if not all([server, sender, password, recipient]):
        logger.warning("Email not configured (missing smtp settings)")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient

        msg.attach(MIMEText(text_body, "plain"))
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(server, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())

        logger.info(f"Email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return False


def notify(config: dict, subject: str, text_body: str, html_body: str = None):
    """Send notification using all configured methods."""
    notify_cfg = config.get("notifications", {})
    if not notify_cfg.get("enabled", False):
        logger.info("Notifications disabled in config — skipping")
        return

    method = notify_cfg.get("method", "telegram")

    if method in ("telegram", "both"):
        tg = notify_cfg.get("telegram", {})
        send_telegram(tg.get("bot_token", ""), tg.get("chat_id", ""), text_body, html_body)

    if method in ("email", "both"):
        email_cfg = notify_cfg.get("email", {})
        send_email(email_cfg, subject, text_body, html_body)


# ---------------------------------------------------------------------------
# Scan + notify
# ---------------------------------------------------------------------------

# Keep a module-level monitor instance so state persists across scans
_monitor_instance: MarketMonitor = None


def _get_monitor(config: dict) -> MarketMonitor:
    """Get or create the monitor singleton for the session."""
    global _monitor_instance
    if _monitor_instance is None:
        reports_dir = config.get("reporting", {}).get("reports_dir", "reports")
        _monitor_instance = MarketMonitor(config, reports_dir)
    return _monitor_instance


def run_scan_and_notify(config: dict, session_label: str = "",
                         is_morning_briefing: bool = False):
    """Run the full market monitor scan and send notifications.

    Args:
        config: Application configuration dict.
        session_label: Human-readable label for this scan.
        is_morning_briefing: If True, sends the comprehensive morning
            briefing instead of a standard summary.  Also refreshes
            the top-100 stock rankings for the day.
    """
    monitor = _get_monitor(config)

    logger.info(f"{'=' * 60}")
    logger.info(f"MARKET MONITOR — {session_label or 'Scheduled Scan'}")
    logger.info(f"{'=' * 60}")

    # Refresh top-100 rankings at the start of each trading day
    if is_morning_briefing:
        logger.info("Morning briefing — refreshing top 100 rankings...")
        monitor.refresh_top_100()

    text_summary, html_report, changes = monitor.run_scan_and_report(
        is_morning_briefing=is_morning_briefing,
    )

    # Save session-specific reports
    reports_dir = config.get("reporting", {}).get("reports_dir", "reports")
    session_dir = Path(reports_dir) / "monitor"
    session_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    (session_dir / f"scan_{ts}.txt").write_text(text_summary)
    (session_dir / f"scan_{ts}.html").write_text(html_report)

    # Log a short status
    logger.info(f"Scan complete — {len(changes)} changes detected")
    for line in text_summary.split("\n")[:15]:
        if line.strip():
            logger.info(line)

    # Decide whether to notify
    mon_cfg = config.get("market_monitor", {})
    should_notify = (
        is_morning_briefing  # Always notify for morning briefing
        or mon_cfg.get("notify_always", False)
        or (mon_cfg.get("notify_on_changes", True) and changes)
    )

    if should_notify:
        now = datetime.now().strftime("%H:%M ET")
        change_count = len(changes)

        # Build notification text
        if is_morning_briefing:
            header = f"<b>DAILY MORNING BRIEFING — {datetime.now().strftime('%A, %B %d')}</b>\n"
            header += f"Top 100 NYSE stocks analyzed\n\n"
        else:
            header = f"<b>Market Monitor — {session_label or now}</b>\n"
            header += f"Changes detected: {change_count}\n\n"

        if mon_cfg.get("notify_full_summary", True):
            notify_text = header + text_summary
        else:
            change_lines = []
            for c in changes[:25]:
                change_lines.append(
                    f"[{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}"
                )
            notify_text = header + "\n".join(change_lines)

        if is_morning_briefing:
            subject = f"Morning Briefing: Top 100 NYSE — {datetime.now().strftime('%A %m/%d')}"
        else:
            subject = f"Market Monitor: {change_count} changes — {now}"
        notify(config, subject, notify_text, html_report)
    else:
        logger.info("No significant changes — notification skipped")

    return text_summary, changes


# ---------------------------------------------------------------------------
# Session labels for each scan time
# ---------------------------------------------------------------------------

SCAN_SESSION_LABELS = {
    "08:00": "Morning Briefing",
    "08:30": "Pre-Market Overview",
    "09:15": "Early Morning Scan",
    "09:45": "Post-Open Analysis",
    "10:30": "Mid-Morning Update",
    "11:30": "Late Morning Update",
    "12:00": "Midday Analysis",
    "12:30": "Midday Analysis",
    "13:30": "Early Afternoon Check",
    "14:00": "Afternoon Update",
    "14:30": "Afternoon Update",
    "15:00": "Power Hour Scan",
    "15:45": "End-of-Day Summary",
    "16:15": "After-Hours Review",
}

# The first scan of the day is always the morning briefing
MORNING_BRIEFING_TIMES = {"08:00", "08:30"}


def get_session_label(time_str: str) -> str:
    """Get a friendly label for a scan time."""
    return SCAN_SESSION_LABELS.get(time_str, f"Scan ({time_str} ET)")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def start_scheduled(config: dict):
    """Start the APScheduler daemon with all configured scan times.

    The first scan of the day (typically 08:00 or 08:30) is treated as the
    *morning briefing* — it refreshes the top-100 rankings and sends a
    comprehensive overview.  All subsequent scans send incremental updates.
    """
    tz = config.get("schedule", {}).get("timezone", "US/Eastern")
    mon_cfg = config.get("market_monitor", {})
    scan_times = mon_cfg.get("scan_times", [
        "08:00", "09:45", "12:00", "14:00", "15:45",
    ])

    scheduler = BlockingScheduler()

    for i, time_str in enumerate(scan_times):
        h, m = time_str.split(":")
        label = get_session_label(time_str)
        is_morning = time_str in MORNING_BRIEFING_TIMES or i == 0

        scheduler.add_job(
            run_scan_and_notify,
            CronTrigger(
                hour=int(h), minute=int(m),
                day_of_week="mon-fri",
                timezone=tz,
            ),
            args=[config, label],
            kwargs={"is_morning_briefing": is_morning},
            id=f"monitor_scan_{i}",
            name=f"market_monitor_{time_str}",
        )
        tag = " [MORNING BRIEFING]" if is_morning else ""
        logger.info(f"Scheduled: {label} at {time_str} ET (Mon-Fri){tag}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("MARKET MONITOR DAEMON RUNNING")
    logger.info(f"  Scans per day: {len(scan_times)}")
    logger.info(f"  Schedule: {', '.join(scan_times)} ET")
    logger.info(f"  Days: Monday - Friday")
    logger.info(f"  Morning briefing: {scan_times[0]} ET")
    notify_cfg = config.get("notifications", {})
    if notify_cfg.get("enabled"):
        logger.info(f"  Notifications: {notify_cfg.get('method', 'telegram')}")
    else:
        logger.info("  Notifications: DISABLED (enable in config.yaml)")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop.\n")

    def handle_signal(signum, frame):
        logger.info("Shutting down market monitor daemon...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Market monitor daemon stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scheduled Market Monitor — Top 100 NYSE stocks, weekdays"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one scan immediately and exit",
    )
    parser.add_argument(
        "--morning", action="store_true",
        help="Run the morning briefing scan (refreshes top 100 rankings)",
    )
    parser.add_argument(
        "--test-notify", action="store_true",
        help="Send a test notification and exit",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(config)

    if args.test_notify:
        logger.info("Sending test notification...")
        test_msg = (
            "<b>Market Monitor — Test Notification</b>\n\n"
            "If you see this, notifications are working correctly.\n"
            f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}"
        )
        notify(config, "Market Monitor: Test Notification", test_msg)
        logger.info("Done. Check your Telegram / email.")
        return

    if args.morning:
        run_scan_and_notify(config, "Morning Briefing", is_morning_briefing=True)
        return

    if args.once:
        run_scan_and_notify(config, "Manual Scan")
        return

    start_scheduled(config)


if __name__ == "__main__":
    main()
