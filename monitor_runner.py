"""
Monitor Runner: Entry point for scheduled and one-off market monitor scans.

Runs independently of IBKR — uses yfinance for all market data.
Analyzes top 100 NYSE stocks: Greeks, IV Rank, 20/50/200 SMA, RSI, MACD, Stochastics.

Usage:
    python monitor_runner.py                              # Run midday scan
    python monitor_runner.py --session pre_market          # Run specific session
    python monitor_runner.py --session closing --notify    # Scan + send notifications
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from market_monitor import MarketMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("monitor_runner")

SESSION_LABELS = {
    "pre_market": "Pre-Market Scan (8:30 AM ET)",
    "morning": "Morning Update (10:00 AM ET)",
    "midday": "Midday Analysis (12:00 PM ET)",
    "afternoon": "Afternoon Check (2:00 PM ET)",
    "closing": "Closing Summary (3:45 PM ET)",
}


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def send_desktop_notification(title: str, body: str):
    """Send OS desktop notification (Linux/macOS/Windows)."""
    try:
        import subprocess
        import platform

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
            # PowerShell toast notification
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


def main():
    parser = argparse.ArgumentParser(description="IBKR Market Monitor Runner")
    parser.add_argument(
        "--session",
        choices=list(SESSION_LABELS.keys()),
        default="midday",
        help="Which session to run",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send desktop notification with results",
    )
    args = parser.parse_args()

    session_label = SESSION_LABELS[args.session]
    config = load_config()

    logger.info(f"Running market monitor: {session_label}")

    reports_dir = config.get("reporting", {}).get("reports_dir", "reports")
    monitor = MarketMonitor(config, reports_dir)
    text_summary, html_report, changes = monitor.run_scan_and_report()

    # Save reports
    reports_dir_path = Path(reports_dir) / "monitor"
    reports_dir_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    text_path = reports_dir_path / f"monitor_{date_str}_{args.session}_{time_str}.txt"
    text_path.write_text(text_summary)
    logger.info(f"Text report saved: {text_path}")

    # Print the text summary
    print("\n" + text_summary)

    # Summary status
    print(f"\nSESSION: {session_label}")
    if changes:
        print(f"CHANGES: {len(changes)} significant changes detected")
        for c in changes[:10]:
            print(f"  [{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}")
    else:
        print("CHANGES: No significant changes since last scan")

    # Desktop notification
    use_notify = args.notify or config.get("market_monitor", {}).get("desktop_notifications", False)
    if use_notify:
        if changes:
            body = f"{len(changes)} changes detected. Top: " + ", ".join(
                c["symbol"] for c in changes[:5]
            )
            send_desktop_notification("IBKR Market Monitor", body)
        elif config.get("market_monitor", {}).get("notify_always", False):
            send_desktop_notification("IBKR Market Monitor", f"{session_label} complete — no changes.")


if __name__ == "__main__":
    main()
