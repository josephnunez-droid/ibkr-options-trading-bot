# IBKR Automated Options Trading Bot

A production-ready automated options trading system for Interactive Brokers, built with Python and `ib_insync`.

## Features

- **6 Options Strategies**: Covered Calls, Cash-Secured Puts, Wheel, Iron Condors, Credit Spreads, Debit Spreads (earnings plays)
- **Dynamic Universe**: Mag 7 + Top ETFs + dynamic screener (top 20 by IV rank, volume, momentum)
- **Risk Engine**: Kelly-based sizing, portfolio exposure limits, daily loss auto-shutdown, margin monitoring
- **Hedge Manager**: Delta hedging, intraday drop stops, VIX spike pause, gap-down protection
- **Smart Execution**: Mid-price limit orders, aggressive retry, fill tracking in SQLite
- **Scheduling**: 5 daily sessions (pre-market, open, midday, power hour, after-hours)
- **Reporting**: Daily HTML reports, colorized terminal dashboard, Telegram/email alerts
- **Kill Switch**: Emergency cancel-all and close-all via `--kill`

## Prerequisites

1. **Interactive Brokers Account** (paper or live)
2. **TWS or IB Gateway** running on your machine
3. **Python 3.10+**
4. **TWS API Settings**:
   - Enable API: File > Global Configuration > API > Settings
   - Check "Enable ActiveX and Socket Clients"
   - Socket port: **7497** (paper) or **7496** (live)
   - Check "Allow connections from localhost only"
   - Uncheck "Read-Only API"

## Installation

```bash
cd "Python Script IBKR"
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:

```yaml
connection:
  trading_mode: "paper"    # Start here! Change to "live" only after testing
  host: "127.0.0.1"
  paper_port: 7497
  live_port: 7496
```

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `trading_mode` | "paper" or "live" | paper |
| `risk.max_daily_loss_pct` | Auto-shutdown threshold | 2% |
| `risk.max_portfolio_options_pct` | Max options exposure | 40% |
| `risk.max_single_name_pct` | Max per-stock exposure | 10% |
| `strategies.<name>.enabled` | Enable/disable strategy | true |

## Usage

### Start the Bot (Scheduled Mode)
```bash
python main.py
```
Runs 5 daily sessions Monday-Friday (ET):
- **9:00 AM** - Pre-market scan
- **9:35 AM** - Execute entries
- **12:00 PM** - Midday position check
- **3:30 PM** - Rolls and closes
- **4:30 PM** - Daily report

### One-Time Scan
```bash
python main.py --scan
```

### Terminal Dashboard
```bash
python main.py --dashboard
```

### Generate Report
```bash
python main.py --report
```

### Emergency Kill Switch
```bash
python main.py --kill
```
Cancels ALL open orders and closes ALL positions at market.

## Paper Trading Walkthrough

1. **Start TWS** and log in to your paper trading account
2. **Verify API** is enabled (port 7497)
3. **Run the bot**:
   ```bash
   python main.py --scan
   ```
4. **Review signals** - the bot will show opportunities without trading
5. **Start scheduled mode**:
   ```bash
   python main.py
   ```
6. **Monitor** via dashboard or daily reports in `reports/`

## Project Structure

```
├── main.py              # Entry point & scheduler
├── config.yaml          # All configurable settings
├── connection.py        # IBKR TWS connection manager
├── data_feeds.py        # Market data & options chains
├── screener.py          # Dynamic universe builder
├── risk_engine.py       # Position sizing & risk limits
├── executor.py          # Order routing & fill tracking
├── hedge_manager.py     # Protective hedging logic
├── reporter.py          # Reports, dashboard, alerts
├── strategies/
│   ├── base.py          # Abstract strategy base
│   ├── covered_call.py  # Covered call writer
│   ├── cash_secured_put.py  # Cash-secured put writer
│   ├── wheel.py         # Wheel strategy (CSP->CC cycle)
│   ├── iron_condor.py   # Iron condor on high-IV ETFs
│   ├── credit_spread.py # Bull put / bear call spreads
│   └── debit_spread.py  # Earnings debit spreads
├── db/
│   └── models.py        # SQLite schema (SQLAlchemy)
├── tests/               # Unit tests
├── reports/             # Generated HTML reports
├── logs/                # Rotating log files
└── requirements.txt
```

## Safety Features

- **Paper trading is the default** - live mode requires config change AND confirmation prompt
- **No market orders** except emergency hedges
- **All orders validated** against risk engine before placement
- **Daily loss limit** (2%) triggers automatic shutdown
- **Margin buffer** (30%) enforced before new entries
- **Earnings blackout** - no short premium trades within 5 days of earnings
- **Kill switch** - instant cancel-all and close-all

## Running Tests

```bash
python -m pytest tests/ -v
```

## Disclaimer

This software is for educational purposes. Trading options involves significant risk. Use paper trading mode for testing. The authors are not responsible for any financial losses.
