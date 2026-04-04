"""
Monitor Runner: Entry point for standalone market monitor scans.

Runs a full technical analysis scan of the top 100 NYSE stocks using
the MarketMonitor class. No IBKR connection required (uses yfinance).

Usage:
    python monitor_runner.py                          # Run midday scan
    python monitor_runner.py --session pre_market      # Run specific session
    python monitor_runner.py --session closing          # End-of-day summary
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from market_monitor import MarketMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("monitor_runner")

SESSION_LABELS = {
    "pre_market": "Pre-Market Scan (8:30 AM ET)",
    "morning": "Morning Update (10:00 AM ET)",
    "midday": "Midday Analysis (12:00 PM ET)",
    "afternoon": "Afternoon Check (2:00 PM ET)",
    "closing": "Closing Summary (3:45 PM ET)",
}


def main():
    parser = argparse.ArgumentParser(description="IBKR Market Monitor Runner")
    parser.add_argument(
        "--session",
        choices=list(SESSION_LABELS.keys()),
        default="midday",
        help="Which session to run",
    )
    args = parser.parse_args()

    session_label = SESSION_LABELS[args.session]
    logger.info(f"Running market monitor: {session_label}")

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    reports_dir = config.get("reporting", {}).get("reports_dir", "reports")
    monitor = MarketMonitor(config, reports_dir)
    text_summary, html_report, changes = monitor.run_scan_and_report()

    # Save session-specific reports
    session_reports_dir = Path(reports_dir) / "monitor"
    session_reports_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    text_path = session_reports_dir / f"monitor_{date_str}_{args.session}_{time_str}.txt"
    html_path = session_reports_dir / f"monitor_{date_str}_{args.session}_{time_str}.html"

    text_path.write_text(text_summary)
    html_path.write_text(html_report)

    logger.info(f"Reports saved: {text_path}, {html_path}")

    # Print the text summary
    print("\n" + text_summary)

    if changes:
        print(f"\n[!] {len(changes)} significant changes detected.")


if __name__ == "__main__":
    main()
