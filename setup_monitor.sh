#!/bin/bash
# Setup script for the Market Monitor daemon.
#
# Installs the systemd user service so the monitor starts
# automatically on login and runs weekday scans.
#
# Usage:
#   chmod +x setup_monitor.sh
#   ./setup_monitor.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="market-monitor"
SERVICE_FILE="${SCRIPT_DIR}/market-monitor.service"
USER_SERVICE_DIR="${HOME}/.config/systemd/user"

echo "=== Market Monitor Setup ==="
echo ""
echo "Working directory: ${SCRIPT_DIR}"
echo ""

# 1. Install Python dependencies
echo "[1/4] Checking Python dependencies..."
pip3 install -q -r "${SCRIPT_DIR}/requirements.txt" 2>/dev/null || {
    echo "  pip install failed — install dependencies manually:"
    echo "  pip3 install -r ${SCRIPT_DIR}/requirements.txt"
}
echo "  Done."

# 2. Copy service file
echo "[2/4] Installing systemd user service..."
mkdir -p "${USER_SERVICE_DIR}"

# Update WorkingDirectory and ExecStart paths in the service file
sed \
    -e "s|WorkingDirectory=.*|WorkingDirectory=${SCRIPT_DIR}|" \
    -e "s|ExecStart=.*|ExecStart=$(which python3) ${SCRIPT_DIR}/scheduled_monitor.py|" \
    "${SERVICE_FILE}" > "${USER_SERVICE_DIR}/${SERVICE_NAME}.service"

echo "  Installed to ${USER_SERVICE_DIR}/${SERVICE_NAME}.service"

# 3. Reload and enable
echo "[3/4] Enabling service..."
systemctl --user daemon-reload
systemctl --user enable "${SERVICE_NAME}"
echo "  Service enabled (starts on login)."

# 4. Start
echo "[4/4] Starting service..."
systemctl --user start "${SERVICE_NAME}"
echo "  Service started."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: Configure notifications in config.yaml:"
echo "  1. Set notifications.enabled: true"
echo "  2. Set notifications.email.sender to your Gmail address"
echo "  3. Set notifications.email.password to a Gmail App Password"
echo "     (create at https://myaccount.google.com/apppasswords)"
echo "  4. Set notifications.email.recipient to where you want alerts"
echo "  5. Test with: python3 scheduled_monitor.py --test-notify"
echo ""
echo "Schedule (Mon-Fri, ET):"
echo "  08:00  Morning Briefing (top 100 refresh + full analysis)"
echo "  09:45  Post-Open Analysis"
echo "  10:45  Mid-Morning Update"
echo "  11:45  Late Morning Update"
echo "  13:00  Early Afternoon Check"
echo "  14:30  Afternoon Update"
echo "  15:45  End-of-Day Summary"
echo ""
echo "Commands:"
echo "  Status:   systemctl --user status ${SERVICE_NAME}"
echo "  Logs:     journalctl --user -u ${SERVICE_NAME} -f"
echo "  Stop:     systemctl --user stop ${SERVICE_NAME}"
echo "  Restart:  systemctl --user restart ${SERVICE_NAME}"
echo "  Disable:  systemctl --user disable ${SERVICE_NAME}"
echo ""
echo "Manual runs:"
echo "  python3 scheduled_monitor.py                # Start daemon"
echo "  python3 scheduled_monitor.py --morning      # Morning briefing"
echo "  python3 scheduled_monitor.py --once          # Single scan"
echo "  python3 scheduled_monitor.py --test-notify   # Test notifications"
