#!/usr/bin/env python3
"""
Scheduled Market Monitor — Enhanced weekday daemon.

Runs the top 100 NYSE stock scanner 8 times per trading day
(Mon-Fri) and sends notifications via Telegram and/or email when
significant changes are detected.  Includes a dedicated Pre-Market
Morning Briefing (07:00 ET) and End-of-Day Summary (15:50 ET) that
always send notifications regardless of detected changes.

NO IBKR connection required — uses yfinance for all market data.

Usage:
    python scheduled_monitor.py              # Start the scheduled daemon
    python scheduled_monitor.py --once       # Run one scan and exit
    python scheduled_monitor.py --morning    # Run morning briefing and exit
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
# Session labels for each scan time (8 scans per day)
# ---------------------------------------------------------------------------

SCAN_SESSIONS = [
    ("07:00", "Pre-Market Briefing"),
    ("09:15", "Market Open Snapshot"),
    ("10:00", "Post-Open Analysis"),
    ("11:30", "Late Morning Update"),
    ("12:30", "Midday Check"),
    ("14:00", "Afternoon Update"),
    ("15:00", "Power Hour Alert"),
    ("15:50", "End-of-Day Summary"),
]

# Quick lookup by time string
SCAN_SESSION_LABELS = {t: label for t, label in SCAN_SESSIONS}

# Times that always send notifications (morning briefing + EOD summary)
ALWAYS_NOTIFY_TIMES = {"07:00", "15:50"}

TOTAL_SCANS = len(SCAN_SESSIONS)


def get_session_label(time_str: str) -> str:
    """Get a friendly label for a scan time."""
    return SCAN_SESSION_LABELS.get(time_str, f"Scan ({time_str} ET)")


def get_session_number(time_str: str) -> int:
    """Get the 1-based session number for a scan time."""
    for i, (t, _) in enumerate(SCAN_SESSIONS):
        if t == time_str:
            return i + 1
    return 0


# ---------------------------------------------------------------------------
# Scan + notify
# ---------------------------------------------------------------------------

def run_scan_and_notify(config: dict, session_label: str = "",
                        is_briefing: bool = False, is_eod: bool = False,
                        scan_number: int = 0, total_scans: int = TOTAL_SCANS):
    """Run the full market monitor scan and send notifications.

    Parameters
    ----------
    config : dict
        Application config loaded from config.yaml.
    session_label : str
        Human-readable label for this scan session.
    is_briefing : bool
        If True, call monitor.generate_morning_briefing() for a
        comprehensive morning overview and always send the notification.
    is_eod : bool
        If True, this is the end-of-day summary; always send notification.
    scan_number : int
        Current scan number (1-based) within the day's schedule.
    total_scans : int
        Total number of scans scheduled per day.
    """
    reports_dir = config.get("reporting", {}).get("reports_dir", "reports")
    monitor = MarketMonitor(config, reports_dir)

    scan_tag = f"Scan {scan_number}/{total_scans}" if scan_number else ""
    notify_priority = "HIGH" if (is_briefing or is_eod) else "NORMAL"

    logger.info(f"{'=' * 60}")
    logger.info(
        f"MARKET MONITOR — {session_label or 'Scheduled Scan'}"
        + (f"  [{scan_tag}]" if scan_tag else "")
        + f"  [Priority: {notify_priority}]"
    )
    logger.info(f"{'=' * 60}")

    # ---- Morning Briefing mode ----
    if is_briefing:
        try:
            briefing_text = monitor.generate_morning_briefing()
        except AttributeError:
            logger.warning(
                "monitor.generate_morning_briefing() not available — "
                "falling back to standard scan"
            )
            briefing_text = None

        if briefing_text:
            # Save briefing report
            session_dir = Path(reports_dir) / "monitor"
            session_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            (session_dir / f"briefing_{ts}.txt").write_text(briefing_text)

            logger.info("Morning briefing generated")
            for line in briefing_text.split("\n")[:20]:
                if line.strip():
                    logger.info(line)

            # Always send morning briefing notification
            now = datetime.now().strftime("%H:%M ET")
            header = (
                f"<b>Pre-Market Briefing — {now}</b>\n"
                f"[{scan_tag}] | Priority: {notify_priority}\n\n"
            )
            notify_text = header + briefing_text
            subject = f"Morning Briefing — {datetime.now().strftime('%Y-%m-%d')}"
            notify(config, subject, notify_text)
            return briefing_text, []

    # ---- Standard scan (including EOD) ----
    text_summary, html_report, changes = monitor.run_scan_and_report()

    # Save session-specific reports
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
    always_send = is_briefing or is_eod
    mon_cfg = config.get("market_monitor", {})
    should_notify = (
        always_send
        or mon_cfg.get("notify_always", False)
        or (mon_cfg.get("notify_on_changes", True) and changes)
    )

    if should_notify:
        now = datetime.now().strftime("%H:%M ET")
        change_count = len(changes)

        # Build notification text
        header = f"<b>Market Monitor — {session_label or now}</b>\n"
        if scan_tag:
            header += f"[{scan_tag}] | Priority: {notify_priority}\n"
        header += f"Changes detected: {change_count}\n\n"

        if is_eod:
            header = f"<b>End-of-Day Summary — {now}</b>\n"
            if scan_tag:
                header += f"[{scan_tag}] | Priority: {notify_priority}\n"
            header += f"Changes detected: {change_count}\n\n"

        if mon_cfg.get("notify_full_summary", True):
            # Send the full text summary
            notify_text = header + text_summary
        else:
            # Send just the changes
            change_lines = []
            for c in changes[:25]:
                change_lines.append(
                    f"[{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}"
                )
            notify_text = header + "\n".join(change_lines)

        if is_eod:
            subject = f"EOD Summary: {change_count} changes — {now}"
        else:
            subject = f"Market Monitor: {change_count} changes — {now}"
        notify(config, subject, notify_text, html_report)
    else:
        logger.info("No significant changes — notification skipped")

    return text_summary, changes


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def log_heartbeat():
    """Log a heartbeat message so operators can verify the daemon is alive."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[HEARTBEAT] Daemon alive at {now}")


# ---------------------------------------------------------------------------
# Weekend handler
# ---------------------------------------------------------------------------

def log_weekend_status():
    """Log a message indicating the market is closed on weekends."""
    day_name = datetime.now().strftime("%A")
    logger.info(
        f"Market is closed today ({day_name}). "
        "No scans scheduled. Next scan on Monday."
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def start_scheduled(config: dict):
    """Start the APScheduler daemon with all configured scan times."""
    tz = config.get("schedule", {}).get("timezone", "US/Eastern")

    scheduler = BlockingScheduler()

    # --- 8 scan sessions per day (Mon-Fri) ---
    for i, (time_str, label) in enumerate(SCAN_SESSIONS):
        h, m = time_str.split(":")
        is_briefing = (time_str == "07:00")
        is_eod = (time_str == "15:50")
        scan_number = i + 1

        scheduler.add_job(
            run_scan_and_notify,
            CronTrigger(
                hour=int(h), minute=int(m),
                day_of_week="mon-fri",
                timezone=tz,
            ),
            args=[config, label],
            kwargs={
                "is_briefing": is_briefing,
                "is_eod": is_eod,
                "scan_number": scan_number,
                "total_scans": TOTAL_SCANS,
            },
            id=f"monitor_scan_{i}",
            name=f"market_monitor_{time_str}",
        )
        priority = "HIGH" if (is_briefing or is_eod) else "NORMAL"
        logger.info(
            f"Scheduled: {label} at {time_str} ET (Mon-Fri) "
            f"[{scan_number}/{TOTAL_SCANS}] priority={priority}"
        )

    # --- Heartbeat every hour on weekdays (06:00 - 17:00 ET) ---
    scheduler.add_job(
        log_heartbeat,
        CronTrigger(
            hour="6-17", minute=0,
            day_of_week="mon-fri",
            timezone=tz,
        ),
        id="heartbeat",
        name="hourly_heartbeat",
    )
    logger.info("Scheduled: Heartbeat every hour 06:00-17:00 ET (Mon-Fri)")

    # --- Weekend status log (once on Sat and Sun mornings at 08:00 ET) ---
    scheduler.add_job(
        log_weekend_status,
        CronTrigger(
            hour=8, minute=0,
            day_of_week="sat,sun",
            timezone=tz,
        ),
        id="weekend_status",
        name="weekend_market_closed",
    )
    logger.info("Scheduled: Weekend status log at 08:00 ET (Sat-Sun)")

    scan_times = [t for t, _ in SCAN_SESSIONS]
    logger.info("")
    logger.info("=" * 60)
    logger.info("MARKET MONITOR DAEMON RUNNING (Enhanced)")
    logger.info(f"  Scans per day: {TOTAL_SCANS}")
    logger.info(f"  Schedule: {', '.join(scan_times)} ET")
    logger.info(f"  Always-notify sessions: Pre-Market Briefing (07:00), EOD Summary (15:50)")
    logger.info(f"  Heartbeat: Every hour 06:00-17:00 ET")
    logger.info(f"  Days: Monday - Friday")
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
        description="Scheduled Market Monitor — Top 100 NYSE stocks, weekdays (8 scans/day)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one scan immediately and exit",
    )
    parser.add_argument(
        "--morning", action="store_true",
        help="Run only the morning briefing scan and exit",
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
        logger.info("Running morning briefing scan...")
        run_scan_and_notify(
            config,
            session_label="Pre-Market Briefing",
            is_briefing=True,
            scan_number=1,
            total_scans=TOTAL_SCANS,
        )
        return

    if args.once:
        run_scan_and_notify(config, "Manual Scan")
        return

    start_scheduled(config)


if __name__ == "__main__":
    main()
