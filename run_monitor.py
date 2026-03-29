#!/usr/bin/env python3
"""
Standalone Market Monitor - runs without IBKR connection.

Uses yfinance to analyze the top 100 NYSE stocks with full technical analysis:
  - Greeks (Delta, Gamma, Theta, Vega) for ATM options
  - IV Rank (52-week implied volatility ranking)
  - Moving Averages: 20, 50, 200-day SMA
  - RSI (14-period)
  - MACD (12, 26, 9)
  - Stochastic Oscillator (14, 3, 3)

Modes:
    python run_monitor.py                    # One-shot scan, JSON to stdout
    python run_monitor.py --top 20           # Only top 20 in output
    python run_monitor.py --email            # One-shot scan + send HTML email
    python run_monitor.py --daemon           # Run as scheduled daemon (5x weekday scans + email)
    python run_monitor.py --alerts-only      # Only output if alerts detected
"""

import sys
import os
import json
import logging
import smtplib
from datetime import datetime, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import yaml

from market_monitor import MarketMonitor

logger = logging.getLogger("run_monitor")


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(__file__).parent / path
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def send_email(config: dict, subject: str, html_body: str):
    """Send an HTML email using config.yaml email settings."""
    email_cfg = config.get("notifications", {}).get("email", {})
    smtp_server = email_cfg.get("smtp_server")
    smtp_port = email_cfg.get("smtp_port", 587)
    sender = email_cfg.get("sender")
    password = email_cfg.get("password")
    recipient = email_cfg.get("recipient")

    if not all([smtp_server, sender, password, recipient]):
        logger.warning("Email not configured in config.yaml notifications.email section")
        logger.info("To enable email: set smtp_server, sender, password, recipient in config.yaml")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        logger.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


SCAN_LABELS = {
    "08:30": "Pre-Market Overview",
    "10:00": "Post-Open Analysis",
    "12:00": "Midday Check",
    "14:00": "Afternoon Update",
    "15:45": "End-of-Day Summary",
}


def run_scan_and_email(config: dict, scan_label: str = "Scan", force_email: bool = False):
    """Run a scan and optionally send the email report."""
    reports_dir = Path(config.get("reporting", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    monitor = MarketMonitor(config, reporter=None)
    scan_time = datetime.now()
    df = monitor.run_scan()

    if df.empty:
        logger.warning("Scan returned no results")
        return None, 0

    alerts = monitor._detect_changes(df)
    alert_count = len(alerts)

    # Always email for pre-market and EOD; others only if alerts exist
    should_email = force_email or alert_count > 0
    if scan_label in ("Pre-Market Overview", "End-of-Day Summary"):
        should_email = True

    if should_email:
        html = monitor.build_email_html(df, scan_time, scan_label)
        today_str = date.today().strftime("%Y-%m-%d")
        subject = f"[Market Monitor] {scan_label} - {today_str}"
        send_email(config, subject, html)

        # Also save HTML report
        html_path = reports_dir / f"email_{scan_time.strftime('%Y%m%d_%H%M%S')}.html"
        html_path.write_text(html)

    return df, alert_count


def run_daemon(config: dict):
    """Run as a scheduled daemon using APScheduler - scans 5x per weekday."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()
    tz = config.get("schedule", {}).get("timezone", "US/Eastern")
    scan_times = config.get("market_monitor", {}).get("scan_times", [
        "08:30", "10:00", "12:00", "14:00", "15:45",
    ])

    for time_str in scan_times:
        h, m = time_str.split(":")
        label = SCAN_LABELS.get(time_str, f"Scan at {time_str}")

        def _job(lbl=label):
            logger.info(f"Starting scheduled scan: {lbl}")
            run_scan_and_email(config, scan_label=lbl)

        scheduler.add_job(
            _job,
            CronTrigger(hour=int(h), minute=int(m), day_of_week="mon-fri", timezone=tz),
            id=f"monitor_{time_str}",
            name=f"Market Monitor {time_str}",
        )
        logger.info(f"Scheduled: {label} at {time_str} ET (Mon-Fri)")

    logger.info("Market Monitor daemon started. Press Ctrl+C to stop.")
    print(f"Market Monitor daemon running — {len(scan_times)} scans/day (Mon-Fri)", file=sys.stderr)
    print(f"Scan times (ET): {', '.join(scan_times)}", file=sys.stderr)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Daemon stopped.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Market Monitor")
    parser.add_argument("--top", type=int, default=100, help="Number of top stocks to include")
    parser.add_argument("--alerts-only", action="store_true", help="Only output if alerts exist")
    parser.add_argument("--email", action="store_true", help="Send HTML email report after scan")
    parser.add_argument("--daemon", action="store_true", help="Run as scheduled daemon (5x weekday scans)")
    parser.add_argument("--label", type=str, default="Manual Scan", help="Scan label for email subject")
    args = parser.parse_args()

    log_level = logging.INFO if (args.daemon or args.email) else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stderr,
    )

    config = load_config()

    if args.daemon:
        run_daemon(config)
        return

    if args.email:
        df, alert_count = run_scan_and_email(config, scan_label=args.label, force_email=True)
        if df is not None:
            print(json.dumps({
                "status": "ok",
                "scan_time": datetime.now().isoformat(),
                "symbols_scanned": len(df),
                "alert_count": alert_count,
                "email_sent": True,
            }, indent=2))
        return

    # Default: one-shot JSON output
    reports_dir = Path(config.get("reporting", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    monitor = MarketMonitor(config, reporter=None)
    df = monitor.run_scan()

    if df.empty:
        output = {"status": "error", "message": "Scan returned no results", "scan_time": datetime.now().isoformat()}
        print(json.dumps(output, indent=2))
        sys.exit(1)

    top_df = df.head(args.top)
    alerts = monitor._detect_changes(df)

    if args.alerts_only and not alerts:
        output = {"status": "no_alerts", "scan_time": datetime.now().isoformat(), "symbols_scanned": len(df)}
        print(json.dumps(output, indent=2))
        return

    overbought = df[df["rsi"] >= monitor.rsi_overbought]["symbol"].tolist() if "rsi" in df.columns else []
    oversold = df[df["rsi"] <= monitor.rsi_oversold]["symbol"].tolist() if "rsi" in df.columns else []
    high_iv = df[df["iv_rank"] >= monitor.iv_rank_high]["symbol"].tolist()
    macd_crosses = df[df["macd_cross"].notna()][["symbol", "macd_cross"]].to_dict("records") if "macd_cross" in df.columns else []
    sma_crosses = df[df["sma_cross"].notna()][["symbol", "sma_cross"]].to_dict("records") if "sma_cross" in df.columns else []

    output = {
        "status": "ok",
        "scan_time": datetime.now().isoformat(),
        "scan_number": monitor._scan_count,
        "symbols_scanned": len(df),
        "top_stocks": top_df.to_dict("records"),
        "signals": {
            "overbought_rsi": overbought,
            "oversold_rsi": oversold,
            "high_iv_rank": high_iv,
            "macd_crossovers": macd_crosses,
            "sma_crossovers": sma_crosses,
        },
        "alerts": alerts,
        "alert_count": len(alerts),
    }

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
