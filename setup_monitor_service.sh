#!/usr/bin/env bash
#
# Setup script for IBKR Market Monitor auto-scheduling.
#
# This creates EITHER a systemd user service OR a cron job to run
# the market monitor scheduler automatically on weekday mornings.
#
# Usage:
#   ./setup_monitor_service.sh systemd    # Install systemd user service
#   ./setup_monitor_service.sh cron       # Install cron job
#   ./setup_monitor_service.sh remove     # Remove both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(command -v python3 || command -v python)"
SERVICE_NAME="ibkr-market-monitor"
CRON_MARKER="# IBKR_MARKET_MONITOR"

usage() {
    echo "Usage: $0 {systemd|cron|remove}"
    echo ""
    echo "  systemd  - Install as a systemd user service (starts at 8:25 AM ET, Mon-Fri)"
    echo "  cron     - Install as a cron job (starts at 8:25 AM ET, Mon-Fri)"
    echo "  remove   - Remove both systemd service and cron job"
    exit 1
}

install_systemd() {
    local service_dir="$HOME/.config/systemd/user"
    mkdir -p "$service_dir"

    # Main service file
    cat > "$service_dir/${SERVICE_NAME}.service" << EOF
[Unit]
Description=IBKR Market Monitor - Top 100 NYSE Stock Scanner
Documentation=file://${SCRIPT_DIR}/market_monitor_scheduler.py

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON} ${SCRIPT_DIR}/market_monitor_scheduler.py
Restart=on-failure
RestartSec=60
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

    # Timer to start at 8:25 AM ET on weekdays and stop at 4:30 PM ET
    cat > "$service_dir/${SERVICE_NAME}.timer" << EOF
[Unit]
Description=Start IBKR Market Monitor on weekday mornings

[Timer]
OnCalendar=Mon..Fri 08:25
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Stop timer at end of trading day
    cat > "$service_dir/${SERVICE_NAME}-stop.service" << EOF
[Unit]
Description=Stop IBKR Market Monitor after trading hours

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl --user stop ${SERVICE_NAME}.service
EOF

    cat > "$service_dir/${SERVICE_NAME}-stop.timer" << EOF
[Unit]
Description=Stop IBKR Market Monitor after market close

[Timer]
OnCalendar=Mon..Fri 16:30
Persistent=false

[Install]
WantedBy=timers.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}.timer"
    systemctl --user enable "${SERVICE_NAME}-stop.timer"
    systemctl --user start "${SERVICE_NAME}.timer"
    systemctl --user start "${SERVICE_NAME}-stop.timer"

    echo ""
    echo "Systemd service installed successfully!"
    echo ""
    echo "  Service: ${SERVICE_NAME}.service"
    echo "  Timer:   ${SERVICE_NAME}.timer (8:25 AM ET, Mon-Fri)"
    echo "  Stop:    ${SERVICE_NAME}-stop.timer (4:30 PM ET, Mon-Fri)"
    echo ""
    echo "Commands:"
    echo "  Start now:    systemctl --user start ${SERVICE_NAME}.service"
    echo "  Stop:         systemctl --user stop ${SERVICE_NAME}.service"
    echo "  Status:       systemctl --user status ${SERVICE_NAME}.service"
    echo "  Logs:         journalctl --user -u ${SERVICE_NAME}.service -f"
    echo "  Disable:      systemctl --user disable ${SERVICE_NAME}.timer"
}

install_cron() {
    # Remove existing entry first
    crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | crontab - 2>/dev/null || true

    # Add new cron job: 8:25 AM Mon-Fri (ET)
    (
        crontab -l 2>/dev/null || true
        echo "25 8 * * 1-5 cd ${SCRIPT_DIR} && ${PYTHON} ${SCRIPT_DIR}/market_monitor_scheduler.py >> ${SCRIPT_DIR}/logs/market_monitor_cron.log 2>&1 ${CRON_MARKER}"
    ) | crontab -

    echo ""
    echo "Cron job installed successfully!"
    echo ""
    echo "  Schedule: 8:25 AM, Monday-Friday"
    echo "  Log:      ${SCRIPT_DIR}/logs/market_monitor_cron.log"
    echo ""
    echo "  View:     crontab -l"
    echo "  Remove:   $0 remove"
    echo ""
    echo "NOTE: The scheduler will run all 5 daily scans (8:30, 10:00, 12:00, 2:00, 3:45 PM ET)"
    echo "and exit after the last scan completes. Cron restarts it the next morning."
}

remove_all() {
    # Remove systemd
    if systemctl --user is-enabled "${SERVICE_NAME}.timer" 2>/dev/null; then
        systemctl --user stop "${SERVICE_NAME}.timer" 2>/dev/null || true
        systemctl --user stop "${SERVICE_NAME}-stop.timer" 2>/dev/null || true
        systemctl --user stop "${SERVICE_NAME}.service" 2>/dev/null || true
        systemctl --user disable "${SERVICE_NAME}.timer" 2>/dev/null || true
        systemctl --user disable "${SERVICE_NAME}-stop.timer" 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/${SERVICE_NAME}"*.{service,timer}
        systemctl --user daemon-reload
        echo "Systemd service removed."
    fi

    # Remove cron
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | crontab -
        echo "Cron job removed."
    fi

    echo "All market monitor scheduling removed."
}

case "${1:-}" in
    systemd) install_systemd ;;
    cron)    install_cron ;;
    remove)  remove_all ;;
    *)       usage ;;
esac
