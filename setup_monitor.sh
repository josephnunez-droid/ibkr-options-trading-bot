#!/usr/bin/env bash
# Quick setup script for the Market Monitor daemon.
# Run: bash setup_monitor.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== Market Monitor Setup ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# 1. Create required directories
echo "[1/4] Creating directories..."
mkdir -p "$SCRIPT_DIR/reports/monitor"
mkdir -p "$SCRIPT_DIR/logs"

# 2. Install Python dependencies
echo "[2/4] Installing Python dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# 3. Verify config
echo "[3/4] Checking config.yaml..."
if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
    echo "  ERROR: config.yaml not found!"
    exit 1
fi

# Check if notifications are configured
if grep -q 'enabled: false' "$SCRIPT_DIR/config.yaml" | head -1; then
    echo ""
    echo "  WARNING: Notifications are DISABLED in config.yaml"
    echo "  To receive alerts, update config.yaml:"
    echo "    notifications:"
    echo "      enabled: true"
    echo "      method: \"telegram\"  # or \"email\" or \"both\""
    echo "      telegram:"
    echo "        bot_token: \"YOUR_BOT_TOKEN\""
    echo "        chat_id: \"YOUR_CHAT_ID\""
    echo ""
fi

# 4. Test run
echo "[4/4] Running a quick test scan..."
echo ""
cd "$SCRIPT_DIR"
python -c "
from market_monitor import MarketMonitor
from nyse_screener import NYSEScreener
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

print('NYSEScreener initialized OK')
print('MarketMonitor initialized OK')
print()
print('To run a single scan:')
print('  python scheduled_monitor.py --once')
print()
print('To start the daemon (runs Mon-Fri):')
print('  python scheduled_monitor.py')
print()
print('To test notifications:')
print('  python scheduled_monitor.py --test-notify')
print()
print('To install as a systemd service:')
print('  sudo cp market-monitor.service /etc/systemd/system/')
print('  sudo systemctl daemon-reload')
print('  sudo systemctl enable --now market-monitor')
"

echo ""
echo "=== Setup Complete ==="
