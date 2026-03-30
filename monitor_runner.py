"""
Monitor Runner: Entry point for the scheduled market monitor trigger.

This script is invoked by Claude Code scheduled triggers to:
1. Run the full market scan (top 100 NYSE stocks)
2. Analyze Greeks, IV Rank, 20/50/200 SMA, RSI, MACD, Stochastics
3. Save reports locally
4. Print summary for the trigger to email via Gmail

Usage:
    python monitor_runner.py [--session pre_market|midday|power_hour|closing]
    python monitor_runner.py --session pre_market --no-options
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from market_monitor import run_scan

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
    "morning": "Morning Update (10:30 AM ET)",
    "midday": "Midday Analysis (12:30 PM ET)",
    "afternoon": "Afternoon Check (2:30 PM ET)",
    "closing": "Closing Summary (4:15 PM ET)",
}


def main():
    parser = argparse.ArgumentParser(description="IBKR Market Monitor Runner")
    parser.add_argument(
        "--session",
        choices=list(SESSION_LABELS.keys()),
        default="midday",
        help="Which session to run",
    )
    parser.add_argument(
        "--no-options",
        action="store_true",
        help="Skip options data fetch (faster)",
    )
    args = parser.parse_args()

    session_label = SESSION_LABELS[args.session]
    include_options = not args.no_options

    logger.info(f"Running market monitor: {session_label}")

    df, text_summary, html_report = run_scan(
        session_label=session_label,
        include_options=include_options,
    )

    # Save reports
    reports_dir = Path("reports/monitor")
    reports_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    text_path = reports_dir / f"monitor_{date_str}_{args.session}_{time_str}.txt"
    html_path = reports_dir / f"monitor_{date_str}_{args.session}_{time_str}.html"

    text_path.write_text(text_summary)
    html_path.write_text(html_report)

    logger.info(f"Reports saved: {text_path}, {html_path}")

    # Print the text summary to stdout - this is what the trigger will capture
    print("\n" + text_summary)

    # Print a short status line for the trigger
    if not df.empty:
        advancing = len(df[df["daily_change_pct"] > 0])
        declining = len(df[df["daily_change_pct"] < 0])
        avg_change = df["daily_change_pct"].mean()
        print(f"\nSTATUS: {advancing} advancing, {declining} declining, "
              f"avg change {avg_change:+.2f}%")
    else:
        print("\nSTATUS: No data available")


if __name__ == "__main__":
    main()
