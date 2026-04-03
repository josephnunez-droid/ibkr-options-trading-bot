"""Reporting engine: daily HTML reports, terminal dashboard, alerts."""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd
from jinja2 import Template
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from db.models import get_session, Trade, Position, DailyPnL, WheelTracker, AlertLog

logger = logging.getLogger(__name__)
console = Console()


class Reporter:
    """Generates reports, dashboards, and alerts."""

    def __init__(self, config: dict, data_feed=None, risk_engine=None):
        self.config = config
        self.report_cfg = config.get("reporting", {})
        self.notify_cfg = config.get("notifications", {})
        self.data_feed = data_feed
        self.risk_engine = risk_engine
        self.reports_dir = Path(self.report_cfg.get("reports_dir", "reports"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(self, account_summary: dict = None) -> str:
        """Generate daily HTML report and save to file."""
        session = get_session()

        today = date.today()
        open_positions = session.query(Position).filter_by(status="OPEN").all()
        today_trades = session.query(Trade).filter(
            Trade.timestamp >= datetime.combine(today, datetime.min.time())
        ).all()
        daily_pnl = session.query(DailyPnL).filter_by(date=today).first()
        active_wheels = session.query(WheelTracker).filter(
            WheelTracker.status != "COMPLETE"
        ).all()
        recent_alerts = session.query(AlertLog).filter(
            AlertLog.timestamp >= datetime.now() - timedelta(hours=24)
        ).order_by(AlertLog.timestamp.desc()).limit(20).all()

        session.close()

        greeks = self.risk_engine.get_portfolio_greeks() if self.risk_engine else {}

        report_data = {
            "date": today.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "account": account_summary or {},
            "positions": [
                {
                    "symbol": p.symbol,
                    "type": p.contract_type,
                    "strike": p.strike,
                    "expiry": p.expiry.strftime("%Y-%m-%d") if p.expiry else "N/A",
                    "qty": p.quantity,
                    "avg_price": f"${p.avg_price:.2f}",
                    "current": f"${p.current_price:.2f}",
                    "pnl": f"${p.unrealized_pnl:+.2f}",
                    "strategy": p.strategy,
                    "delta": f"{p.delta:.3f}",
                    "theta": f"{p.theta:.3f}",
                }
                for p in open_positions
            ],
            "trades": [
                {
                    "time": t.timestamp.strftime("%H:%M:%S"),
                    "symbol": t.symbol,
                    "action": f"{t.direction} {t.action}",
                    "type": t.contract_type,
                    "strike": t.strike,
                    "qty": t.quantity,
                    "price": f"${t.price:.2f}",
                    "strategy": t.strategy,
                }
                for t in today_trades
            ],
            "greeks": greeks,
            "daily_pnl": {
                "realized": f"${daily_pnl.realized_pnl:+.2f}" if daily_pnl else "$0.00",
                "unrealized": f"${daily_pnl.unrealized_pnl:+.2f}" if daily_pnl else "$0.00",
                "total": f"${daily_pnl.total_pnl:+.2f}" if daily_pnl else "$0.00",
            },
            "wheels": [
                {
                    "symbol": w.symbol,
                    "status": w.status,
                    "basis": f"${w.cost_basis:.2f}",
                    "premium": f"${w.total_premium_collected:.2f}",
                    "entries": f"{w.csp_entries}CSP/{w.cc_entries}CC",
                }
                for w in active_wheels
            ],
            "alerts": [
                {
                    "time": a.timestamp.strftime("%H:%M:%S"),
                    "level": a.level,
                    "message": a.message,
                }
                for a in recent_alerts
            ],
        }

        html = self._render_html(report_data)

        filename = f"daily_report_{today.strftime('%Y%m%d')}.html"
        filepath = self.reports_dir / filename
        filepath.write_text(html)

        logger.info(f"Daily report saved: {filepath}")
        return str(filepath)

    def _render_html(self, data: dict) -> str:
        """Render HTML report from template."""
        template = Template(HTML_REPORT_TEMPLATE)
        return template.render(**data)

    def display_dashboard(self, account_summary: dict = None):
        """Display colorized terminal dashboard using Rich."""
        console.clear()

        mode = self.config.get("connection", {}).get("trading_mode", "paper").upper()
        color = "green" if mode == "PAPER" else "red"
        console.print(Panel(
            f"[bold {color}]IBKR Options Trading Bot — {mode} MODE[/bold {color}]\n"
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            style=color,
        ))

        if account_summary:
            acct_table = Table(title="Account Summary", show_header=True)
            acct_table.add_column("Metric", style="cyan")
            acct_table.add_column("Value", style="green", justify="right")

            net_liq = account_summary.get("NetLiquidation", 0)
            acct_table.add_row("Net Liquidation", f"${net_liq:,.2f}")
            acct_table.add_row("Buying Power", f"${account_summary.get('BuyingPower', 0):,.2f}")
            acct_table.add_row("Excess Liquidity", f"${account_summary.get('ExcessLiquidity', 0):,.2f}")
            acct_table.add_row("Unrealized P&L", f"${account_summary.get('UnrealizedPnL', 0):+,.2f}")
            acct_table.add_row("Realized P&L", f"${account_summary.get('RealizedPnL', 0):+,.2f}")
            console.print(acct_table)

        if self.risk_engine:
            greeks = self.risk_engine.get_portfolio_greeks()
            greeks_table = Table(title="Portfolio Greeks", show_header=True)
            greeks_table.add_column("Greek", style="cyan")
            greeks_table.add_column("Value", style="yellow", justify="right")
            greeks_table.add_row("Delta", f"{greeks['delta']:+,.2f}")
            greeks_table.add_row("Gamma", f"{greeks['gamma']:+,.4f}")
            greeks_table.add_row("Theta", f"{greeks['theta']:+,.2f}")
            greeks_table.add_row("Vega", f"{greeks['vega']:+,.2f}")
            console.print(greeks_table)

        session = get_session()
        positions = session.query(Position).filter_by(status="OPEN").all()

        if positions:
            pos_table = Table(title="Open Positions", show_header=True)
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Type")
            pos_table.add_column("Strike", justify="right")
            pos_table.add_column("Expiry")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Avg Price", justify="right")
            pos_table.add_column("Current", justify="right")
            pos_table.add_column("P&L", justify="right")
            pos_table.add_column("Strategy")

            for p in positions:
                pnl_color = "green" if p.unrealized_pnl >= 0 else "red"
                pos_table.add_row(
                    p.symbol,
                    p.contract_type,
                    f"${p.strike:.2f}" if p.strike else "-",
                    p.expiry.strftime("%m/%d") if p.expiry else "-",
                    str(p.quantity),
                    f"${p.avg_price:.2f}",
                    f"${p.current_price:.2f}",
                    f"[{pnl_color}]${p.unrealized_pnl:+.2f}[/{pnl_color}]",
                    p.strategy,
                )

            console.print(pos_table)
        else:
            console.print("[dim]No open positions[/dim]")

        today_start = datetime.combine(date.today(), datetime.min.time())
        trades = session.query(Trade).filter(
            Trade.timestamp >= today_start
        ).order_by(Trade.timestamp.desc()).limit(10).all()

        if trades:
            trade_table = Table(title="Today's Trades", show_header=True)
            trade_table.add_column("Time")
            trade_table.add_column("Symbol", style="cyan")
            trade_table.add_column("Action")
            trade_table.add_column("Type")
            trade_table.add_column("Qty", justify="right")
            trade_table.add_column("Price", justify="right")
            trade_table.add_column("Strategy")

            for t in trades:
                trade_table.add_row(
                    t.timestamp.strftime("%H:%M:%S"),
                    t.symbol,
                    f"{t.direction} {t.action}",
                    t.contract_type,
                    str(t.quantity),
                    f"${t.price:.2f}",
                    t.strategy,
                )

            console.print(trade_table)

        recent_alerts = session.query(AlertLog).filter(
            AlertLog.timestamp >= datetime.now() - timedelta(hours=4)
        ).order_by(AlertLog.timestamp.desc()).limit(5).all()

        if recent_alerts:
            alert_table = Table(title="Recent Alerts", show_header=True)
            alert_table.add_column("Time")
            alert_table.add_column("Level")
            alert_table.add_column("Message")

            for a in recent_alerts:
                level_color = {
                    "INFO": "blue", "WARNING": "yellow", "CRITICAL": "red"
                }.get(a.level, "white")
                alert_table.add_row(
                    a.timestamp.strftime("%H:%M:%S"),
                    f"[{level_color}]{a.level}[/{level_color}]",
                    a.message,
                )

            console.print(alert_table)

        session.close()

    def send_alert(self, message: str, level: str = "INFO", category: str = "general"):
        """Log alert and optionally send notification."""
        try:
            session = get_session()
            alert = AlertLog(level=level, category=category, message=message)
            session.add(alert)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")

        color = {"INFO": "blue", "WARNING": "yellow", "CRITICAL": "red"}.get(level, "white")
        console.print(f"[{color}][{level}][/{color}] {message}")

        if self.notify_cfg.get("enabled", False):
            if category in self.notify_cfg.get("alert_on", []) or level == "CRITICAL":
                self._send_notification(message, level)

    def _send_notification(self, message: str, level: str):
        """Send notification via configured method."""
        method = self.notify_cfg.get("method", "telegram")
        if method == "telegram":
            self._send_telegram(message, level)
        elif method == "email":
            self._send_email(message, level)

        # Desktop notification (always attempted if enabled)
        if self.config.get("market_monitor", {}).get("desktop_notifications", False):
            self._send_desktop_notification(f"IBKR Bot [{level}]", message)

    def _send_telegram(self, message: str, level: str):
        """Send Telegram notification."""
        import requests

        token = self.notify_cfg.get("telegram", {}).get("bot_token", "")
        chat_id = self.notify_cfg.get("telegram", {}).get("chat_id", "")

        if not token or not chat_id:
            return

        try:
            text = f"*IBKR Bot [{level}]*\n{message}"
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }, timeout=10)
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")

    def _send_email(self, message: str, level: str):
        """Send email notification."""
        import smtplib
        from email.mime.text import MIMEText

        email_cfg = self.notify_cfg.get("email", {})
        if not email_cfg.get("smtp_server"):
            return

        try:
            msg = MIMEText(f"[{level}] {message}")
            msg["Subject"] = f"IBKR Bot Alert: {level}"
            msg["From"] = email_cfg["sender"]
            msg["To"] = email_cfg["recipient"]

            with smtplib.SMTP(email_cfg["smtp_server"], email_cfg["smtp_port"]) as server:
                server.starttls()
                server.login(email_cfg["sender"], email_cfg["password"])
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Email notification failed: {e}")

    @staticmethod
    def _send_desktop_notification(title: str, body: str):
        """Send OS desktop notification (Linux/macOS/Windows)."""
        import subprocess
        import platform

        try:
            system = platform.system()
            if system == "Linux":
                subprocess.run(
                    ["notify-send", "--urgency=normal", "--app-name=IBKR Bot", title, body],
                    timeout=5, check=False,
                )
            elif system == "Darwin":
                script = f'display notification "{body}" with title "{title}"'
                subprocess.run(["osascript", "-e", script], timeout=5, check=False)
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")


# --- HTML Report Template ---

HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>IBKR Trading Report - {{ date }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
        h1 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
        h2 { color: #00d4ff; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th { background: #16213e; color: #00d4ff; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #2a2a4a; }
        tr:hover { background: #16213e; }
        .positive { color: #00e676; }
        .negative { color: #ff5252; }
        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0; }
        .metric-card { background: #16213e; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00d4ff; }
        .metric-label { font-size: 12px; color: #888; margin-top: 5px; }
        .alert-warning { color: #ffab00; }
        .alert-critical { color: #ff5252; font-weight: bold; }
        .footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid #2a2a4a; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Daily Trading Report - {{ date }}</h1>
    <p>Generated: {{ generated_at }}</p>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">${{ account.get('NetLiquidation', 0) | default(0) | int }}</div>
            <div class="metric-label">Net Liquidation</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ daily_pnl.total }}</div>
            <div class="metric-label">Daily P&L</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ positions | length }}</div>
            <div class="metric-label">Open Positions</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ trades | length }}</div>
            <div class="metric-label">Today's Trades</div>
        </div>
    </div>

    {% if greeks %}
    <h2>Portfolio Greeks</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{{ '%.2f' | format(greeks.get('delta', 0)) }}</div>
            <div class="metric-label">Delta</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ '%.4f' | format(greeks.get('gamma', 0)) }}</div>
            <div class="metric-label">Gamma</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ '%.2f' | format(greeks.get('theta', 0)) }}</div>
            <div class="metric-label">Theta</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ '%.2f' | format(greeks.get('vega', 0)) }}</div>
            <div class="metric-label">Vega</div>
        </div>
    </div>
    {% endif %}

    {% if positions %}
    <h2>Open Positions</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Type</th><th>Strike</th><th>Expiry</th>
            <th>Qty</th><th>Avg Price</th><th>Current</th><th>P&L</th><th>Strategy</th>
            <th>Delta</th><th>Theta</th>
        </tr>
        {% for p in positions %}
        <tr>
            <td>{{ p.symbol }}</td><td>{{ p.type }}</td><td>{{ p.strike }}</td>
            <td>{{ p.expiry }}</td><td>{{ p.qty }}</td><td>{{ p.avg_price }}</td>
            <td>{{ p.current }}</td><td>{{ p.pnl }}</td><td>{{ p.strategy }}</td>
            <td>{{ p.delta }}</td><td>{{ p.theta }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if trades %}
    <h2>Today's Trades</h2>
    <table>
        <tr>
            <th>Time</th><th>Symbol</th><th>Action</th><th>Type</th>
            <th>Qty</th><th>Price</th><th>Strategy</th>
        </tr>
        {% for t in trades %}
        <tr>
            <td>{{ t.time }}</td><td>{{ t.symbol }}</td><td>{{ t.action }}</td>
            <td>{{ t.type }}</td><td>{{ t.qty }}</td><td>{{ t.price }}</td>
            <td>{{ t.strategy }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if wheels %}
    <h2>Wheel Tracker</h2>
    <table>
        <tr><th>Symbol</th><th>Status</th><th>Cost Basis</th><th>Premium</th><th>Entries</th></tr>
        {% for w in wheels %}
        <tr>
            <td>{{ w.symbol }}</td><td>{{ w.status }}</td><td>{{ w.basis }}</td>
            <td>{{ w.premium }}</td><td>{{ w.entries }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if alerts %}
    <h2>Recent Alerts</h2>
    <table>
        <tr><th>Time</th><th>Level</th><th>Message</th></tr>
        {% for a in alerts %}
        <tr>
            <td>{{ a.time }}</td>
            <td class="{{ 'alert-critical' if a.level == 'CRITICAL' else 'alert-warning' if a.level == 'WARNING' else '' }}">{{ a.level }}</td>
            <td>{{ a.message }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    <div class="footer">
        IBKR Automated Options Trading Bot | Report generated automatically
    </div>
</body>
</html>"""
