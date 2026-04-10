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
echo ""
echo "Configure notifications in config.yaml under 'notifications'."
