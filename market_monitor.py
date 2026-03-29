"""
Market Monitor: Screens top 100 NYSE stocks and analyzes technical indicators.

Runs multiple times per trading day (weekdays), calculates:
- Greeks (Delta, Gamma, Theta, Vega) for options
- IV Rank (52-week implied volatility ranking)
- Moving Averages: 20, 50, 200-day SMA
- RSI (14-period Relative Strength Index)
- MACD (12, 26, 9)
- Stochastic Oscillator (14, 3, 3)

Sends notifications when significant changes are detected.
"""

import logging
import json
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Top 100 NYSE-listed large-cap stocks (high liquidity, active options)
NYSE_TOP_100 = [
    # Mega-cap / Mag7 (NYSE-listed or cross-listed with high NYSE option volume)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # Consumer
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrials
    "CAT", "BA", "HON", "UPS", "GE", "RTX", "DE", "LMT", "MMM", "FDX",
    # Technology
    "CRM", "ORCL", "IBM", "INTC", "AMD", "AVGO", "QCOM", "TXN", "AMAT", "MU",
    "INTU", "NOW", "ADBE", "PANW", "SNPS",
    # Communication / Media
    "DIS", "NFLX", "T", "VZ", "CMCSA",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY",
    # Real Estate / Utilities
    "AMT", "PLD", "NEE", "DUK", "SO", "D",
    # ETFs (broad market + sector)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "GLD", "SOXX",
    # High-volume options names
    "COIN", "PLTR", "UBER", "SQ", "PYPL", "ABNB", "CRWD", "NET", "SNOW", "DKNG",
]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average of gains/losses."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Compute Stochastic %K and %D."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def _compute_iv_rank(symbol: str) -> float:
    """Calculate IV Rank based on 252-day historical volatility range."""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="1y")
        if hist.empty or len(hist) < 60:
            return 50.0

        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rolling_hv = returns.rolling(20).std() * np.sqrt(252) * 100
        rolling_hv = rolling_hv.dropna()

        if len(rolling_hv) < 20:
            return 50.0

        current = rolling_hv.iloc[-1]
        low = rolling_hv.min()
        high = rolling_hv.max()

        if high == low:
            return 50.0

        rank = float((current - low) / (high - low) * 100)
        return max(0.0, min(100.0, rank))
    except Exception:
        return 50.0


def _fetch_atm_greeks(symbol: str, price: float) -> dict:
    """Fetch Greeks for the nearest ATM option (call) from yfinance."""
    greeks = {"opt_delta": None, "opt_gamma": None, "opt_theta": None,
              "opt_vega": None, "opt_iv": None, "opt_strike": None,
              "opt_expiry": None}
    try:
        tk = yf.Ticker(symbol)
        expirations = tk.options
        if not expirations:
            return greeks

        # Pick the nearest expiry that is >= 20 DTE (skip weeklies)
        today = date.today()
        target_exp = None
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if 20 <= dte <= 60:
                target_exp = exp_str
                break
        if target_exp is None and expirations:
            target_exp = expirations[min(1, len(expirations) - 1)]

        chain = tk.option_chain(target_exp)
        calls = chain.calls
        if calls.empty:
            return greeks

        # Find the strike closest to current price (ATM)
        calls = calls.copy()
        calls["dist"] = abs(calls["strike"] - price)
        atm = calls.loc[calls["dist"].idxmin()]

        greeks["opt_strike"] = float(atm["strike"])
        greeks["opt_expiry"] = target_exp
        greeks["opt_iv"] = round(float(atm.get("impliedVolatility", 0)) * 100, 2)

        # yfinance option chains include Greeks when available
        for greek, col in [("opt_delta", "delta"), ("opt_gamma", "gamma"),
                           ("opt_theta", "theta"), ("opt_vega", "vega")]:
            val = atm.get(col)
            if val is not None and not pd.isna(val):
                greeks[greek] = round(float(val), 4)

        return greeks
    except Exception:
        return greeks


class MarketMonitor:
    """
    Monitors the top 100 NYSE stocks with full technical analysis.

    Runs on a weekday schedule, detects significant changes between scans,
    and sends notifications via the Reporter alert system.
    """

    def __init__(self, config: dict, reporter=None):
        self.config = config
        self.reporter = reporter
        self.monitor_cfg = config.get("market_monitor", {})
        self.symbols = self.monitor_cfg.get("symbols", NYSE_TOP_100[:100])
        self.top_n_report = self.monitor_cfg.get("top_n_report", 100)

        # Thresholds for "significant change" notifications
        self.rsi_overbought = self.monitor_cfg.get("rsi_overbought", 70)
        self.rsi_oversold = self.monitor_cfg.get("rsi_oversold", 30)
        self.iv_rank_high = self.monitor_cfg.get("iv_rank_high", 70)
        self.macd_crossover_notify = self.monitor_cfg.get("macd_crossover_notify", True)
        self.sma_crossover_notify = self.monitor_cfg.get("sma_crossover_notify", True)

        self.reports_dir = Path(config.get("reporting", {}).get("reports_dir", "reports"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._previous_scan: Dict[str, dict] = {}
        self._scan_count = 0

    def run_scan(self) -> pd.DataFrame:
        """
        Run a full technical analysis scan on all monitored symbols.

        Returns a DataFrame with one row per symbol containing all indicators.
        """
        scan_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"MARKET MONITOR SCAN #{self._scan_count + 1} at {scan_time.strftime('%H:%M:%S')}")
        logger.info(f"Scanning {len(self.symbols)} symbols...")
        logger.info("=" * 60)

        results = []
        for i, symbol in enumerate(self.symbols):
            try:
                row = self._analyze_symbol(symbol)
                if row is not None:
                    results.append(row)
                if (i + 1) % 25 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(self.symbols)} symbols analyzed")
            except Exception as e:
                logger.debug(f"  Failed to analyze {symbol}: {e}")

        if not results:
            logger.warning("Market monitor scan returned no results")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Rank by composite performance score
        df = self._rank_stocks(df)

        # Detect changes and send notifications
        alerts = self._detect_changes(df)
        self._send_alerts(alerts, scan_time)

        # Save current scan for next comparison
        self._previous_scan = {row["symbol"]: row for row in results}
        self._scan_count += 1

        # Save report
        self._save_scan_report(df, scan_time)

        # Send summary notification
        self._send_summary(df, scan_time)

        logger.info(f"Scan complete: {len(df)} symbols analyzed, {len(alerts)} alerts generated")
        return df

    def _analyze_symbol(self, symbol: str) -> Optional[dict]:
        """Compute all technical indicators for a single symbol."""
        tk = yf.Ticker(symbol)
        hist = tk.history(period="1y")

        if hist.empty or len(hist) < 200:
            # Need at least 200 bars for 200-day MA
            # Try with less if we have enough for other indicators
            if hist.empty or len(hist) < 30:
                return None

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        price = float(close.iloc[-1])

        # Moving Averages
        sma_20 = float(close.tail(20).mean()) if len(close) >= 20 else None
        sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else None
        sma_200 = float(close.tail(200).mean()) if len(close) >= 200 else None

        # RSI (14-period)
        rsi_series = _compute_rsi(close, 14)
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

        # MACD (12, 26, 9)
        if len(close) >= 35:
            macd_line, signal_line, macd_hist = _compute_macd(close)
            macd_val = float(macd_line.iloc[-1])
            macd_signal = float(signal_line.iloc[-1])
            macd_histogram = float(macd_hist.iloc[-1])
            # Check for crossover (MACD crossed signal in last 2 bars)
            macd_cross = None
            if len(macd_line) >= 2 and len(signal_line) >= 2:
                prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
                curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
                if prev_diff <= 0 < curr_diff:
                    macd_cross = "BULLISH"
                elif prev_diff >= 0 > curr_diff:
                    macd_cross = "BEARISH"
        else:
            macd_val = macd_signal = macd_histogram = None
            macd_cross = None

        # Stochastic Oscillator (14, 3, 3)
        if len(close) >= 17:
            stoch_k, stoch_d = _compute_stochastic(high, low, close)
            stoch_k_val = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else None
            stoch_d_val = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else None
        else:
            stoch_k_val = stoch_d_val = None

        # IV Rank
        iv_rank = _compute_iv_rank(symbol)

        # Price change metrics
        day_change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0
        week_change_pct = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
        month_change_pct = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0

        # SMA crossover detection
        sma_cross = None
        if sma_20 is not None and sma_50 is not None and len(close) >= 51:
            prev_sma_20 = float(close.iloc[-21:-1].mean())
            prev_sma_50 = float(close.iloc[-51:-1].mean())
            if prev_sma_20 <= prev_sma_50 and sma_20 > sma_50:
                sma_cross = "GOLDEN_CROSS"  # 20 crosses above 50
            elif prev_sma_20 >= prev_sma_50 and sma_20 < sma_50:
                sma_cross = "DEATH_CROSS"  # 20 crosses below 50

        # Volume analysis
        avg_volume = float(hist["Volume"].tail(20).mean()) if len(hist) >= 20 else 0
        today_volume = float(hist["Volume"].iloc[-1]) if not hist.empty else 0
        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0

        # ATM Options Greeks
        greeks = _fetch_atm_greeks(symbol, price)

        return {
            "symbol": symbol,
            "price": price,
            "day_change_pct": round(day_change_pct, 2),
            "week_change_pct": round(week_change_pct, 2),
            "month_change_pct": round(month_change_pct, 2),
            "sma_20": round(sma_20, 2) if sma_20 else None,
            "sma_50": round(sma_50, 2) if sma_50 else None,
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "price_vs_sma20": round((price / sma_20 - 1) * 100, 2) if sma_20 else None,
            "price_vs_sma50": round((price / sma_50 - 1) * 100, 2) if sma_50 else None,
            "price_vs_sma200": round((price / sma_200 - 1) * 100, 2) if sma_200 else None,
            "sma_cross": sma_cross,
            "rsi": round(rsi, 2) if rsi is not None else None,
            "macd": round(macd_val, 4) if macd_val is not None else None,
            "macd_signal": round(macd_signal, 4) if macd_signal is not None else None,
            "macd_histogram": round(macd_histogram, 4) if macd_histogram is not None else None,
            "macd_cross": macd_cross,
            "stoch_k": round(stoch_k_val, 2) if stoch_k_val is not None else None,
            "stoch_d": round(stoch_d_val, 2) if stoch_d_val is not None else None,
            "iv_rank": round(iv_rank, 2),
            "avg_volume": int(avg_volume),
            "volume_ratio": round(volume_ratio, 2),
            # ATM Options Greeks
            "opt_delta": greeks["opt_delta"],
            "opt_gamma": greeks["opt_gamma"],
            "opt_theta": greeks["opt_theta"],
            "opt_vega": greeks["opt_vega"],
            "opt_iv": greeks["opt_iv"],
            "opt_strike": greeks["opt_strike"],
            "opt_expiry": greeks["opt_expiry"],
        }

    def _rank_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank stocks by a composite performance score."""
        df = df.copy()

        # Composite: weighted blend of momentum + IV rank + volume
        scores = []
        for _, row in df.iterrows():
            momentum = (
                (row.get("day_change_pct") or 0) * 0.3 +
                (row.get("week_change_pct") or 0) * 0.3 +
                (row.get("month_change_pct") or 0) * 0.4
            )
            iv_score = row.get("iv_rank", 50)
            vol_score = min(row.get("volume_ratio", 1.0), 5.0) * 20  # cap at 100
            scores.append(abs(momentum) * 0.4 + iv_score * 0.3 + vol_score * 0.3)

        df["composite_score"] = scores
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def _detect_changes(self, df: pd.DataFrame) -> List[dict]:
        """Detect significant changes worth notifying about."""
        alerts = []

        for _, row in df.iterrows():
            symbol = row["symbol"]
            prev = self._previous_scan.get(symbol)

            # RSI extremes
            if row.get("rsi") is not None:
                if row["rsi"] >= self.rsi_overbought:
                    alerts.append({
                        "symbol": symbol,
                        "type": "RSI_OVERBOUGHT",
                        "message": f"{symbol} RSI={row['rsi']:.1f} (overbought >{self.rsi_overbought})",
                        "severity": "WARNING",
                    })
                elif row["rsi"] <= self.rsi_oversold:
                    alerts.append({
                        "symbol": symbol,
                        "type": "RSI_OVERSOLD",
                        "message": f"{symbol} RSI={row['rsi']:.1f} (oversold <{self.rsi_oversold})",
                        "severity": "WARNING",
                    })

            # IV Rank high
            if row.get("iv_rank", 0) >= self.iv_rank_high:
                alerts.append({
                    "symbol": symbol,
                    "type": "HIGH_IV_RANK",
                    "message": f"{symbol} IV Rank={row['iv_rank']:.1f}% (elevated, good for premium selling)",
                    "severity": "INFO",
                })

            # MACD crossover
            if self.macd_crossover_notify and row.get("macd_cross"):
                alerts.append({
                    "symbol": symbol,
                    "type": "MACD_CROSSOVER",
                    "message": f"{symbol} MACD {row['macd_cross']} crossover detected",
                    "severity": "INFO",
                })

            # SMA crossover (Golden/Death cross)
            if self.sma_crossover_notify and row.get("sma_cross"):
                cross_type = "Golden Cross (bullish)" if row["sma_cross"] == "GOLDEN_CROSS" else "Death Cross (bearish)"
                alerts.append({
                    "symbol": symbol,
                    "type": "SMA_CROSSOVER",
                    "message": f"{symbol} {cross_type}: SMA20 vs SMA50",
                    "severity": "WARNING",
                })

            # Stochastic extremes
            if row.get("stoch_k") is not None:
                if row["stoch_k"] >= 80 and row.get("stoch_d", 0) >= 80:
                    alerts.append({
                        "symbol": symbol,
                        "type": "STOCH_OVERBOUGHT",
                        "message": f"{symbol} Stochastic overbought: %K={row['stoch_k']:.1f}, %D={row['stoch_d']:.1f}",
                        "severity": "INFO",
                    })
                elif row["stoch_k"] <= 20 and row.get("stoch_d", 0) <= 20:
                    alerts.append({
                        "symbol": symbol,
                        "type": "STOCH_OVERSOLD",
                        "message": f"{symbol} Stochastic oversold: %K={row['stoch_k']:.1f}, %D={row['stoch_d']:.1f}",
                        "severity": "INFO",
                    })

            # Unusual volume
            if row.get("volume_ratio", 1.0) >= 2.0:
                alerts.append({
                    "symbol": symbol,
                    "type": "HIGH_VOLUME",
                    "message": f"{symbol} unusual volume: {row['volume_ratio']:.1f}x average",
                    "severity": "INFO",
                })

            # Large daily move
            if abs(row.get("day_change_pct", 0)) >= 5.0:
                direction = "UP" if row["day_change_pct"] > 0 else "DOWN"
                alerts.append({
                    "symbol": symbol,
                    "type": "LARGE_MOVE",
                    "message": f"{symbol} large move {direction} {row['day_change_pct']:+.2f}% today",
                    "severity": "WARNING",
                })

        return alerts

    def _send_alerts(self, alerts: List[dict], scan_time: datetime):
        """Send detected alerts via the reporter notification system."""
        if not alerts:
            return

        # Group alerts by severity
        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        warnings = [a for a in alerts if a["severity"] == "WARNING"]
        info = [a for a in alerts if a["severity"] == "INFO"]

        # Send critical/warning alerts individually
        for alert in critical + warnings:
            if self.reporter:
                self.reporter.send_alert(
                    f"[Monitor] {alert['message']}",
                    level=alert["severity"],
                    category="market_monitor",
                )
            else:
                logger.warning(f"[Monitor Alert] {alert['message']}")

        # Batch info alerts into a summary
        if info and len(info) > 5:
            summary = f"[Monitor] {len(info)} informational alerts detected. "
            summary += "Top items: " + "; ".join(a["message"] for a in info[:5])
            if self.reporter:
                self.reporter.send_alert(summary, level="INFO", category="market_monitor")
            else:
                logger.info(summary)
        else:
            for alert in info:
                if self.reporter:
                    self.reporter.send_alert(
                        f"[Monitor] {alert['message']}",
                        level="INFO",
                        category="market_monitor",
                    )

    def _send_summary(self, df: pd.DataFrame, scan_time: datetime):
        """Send a concise scan summary notification."""
        if df.empty:
            return

        top_5 = df.head(5)
        lines = [f"Market Monitor Scan #{self._scan_count} - {scan_time.strftime('%H:%M ET')}"]
        lines.append(f"Analyzed {len(df)} symbols\n")
        lines.append("Top 5 by composite score:")

        for _, row in top_5.iterrows():
            rsi_str = f"RSI={row['rsi']:.0f}" if row.get('rsi') is not None else "RSI=N/A"
            iv_str = f"IV={row['iv_rank']:.0f}%"
            chg_str = f"{row['day_change_pct']:+.1f}%"
            delta_str = f"Δ={row['opt_delta']:.2f}" if row.get('opt_delta') is not None else ""
            theta_str = f"Θ={row['opt_theta']:.3f}" if row.get('opt_theta') is not None else ""
            greeks_str = f" | {delta_str} {theta_str}" if delta_str else ""
            lines.append(
                f"  #{row['rank']} {row['symbol']:6s} ${row['price']:.2f} "
                f"({chg_str}) | {rsi_str} | {iv_str}{greeks_str}"
            )

        # Count notable conditions
        overbought = len(df[df["rsi"] >= self.rsi_overbought]) if "rsi" in df.columns else 0
        oversold = len(df[df["rsi"] <= self.rsi_oversold]) if "rsi" in df.columns else 0
        high_iv = len(df[df["iv_rank"] >= self.iv_rank_high])

        lines.append(f"\nSignals: {overbought} overbought, {oversold} oversold, {high_iv} high IV")

        summary = "\n".join(lines)

        if self.reporter:
            self.reporter.send_alert(summary, level="INFO", category="market_monitor")
        else:
            logger.info(summary)

    def _save_scan_report(self, df: pd.DataFrame, scan_time: datetime):
        """Save scan results to JSON and CSV for reference."""
        timestamp = scan_time.strftime("%Y%m%d_%H%M%S")

        # CSV report
        csv_path = self.reports_dir / f"monitor_scan_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Scan report saved: {csv_path}")

        # JSON summary (top 20 + notable stocks)
        summary = {
            "scan_time": scan_time.isoformat(),
            "scan_number": self._scan_count,
            "total_symbols": len(df),
            "top_20": df.head(20).to_dict("records"),
            "overbought_rsi": df[df["rsi"] >= self.rsi_overbought]["symbol"].tolist() if "rsi" in df.columns else [],
            "oversold_rsi": df[df["rsi"] <= self.rsi_oversold]["symbol"].tolist() if "rsi" in df.columns else [],
            "high_iv_rank": df[df["iv_rank"] >= self.iv_rank_high]["symbol"].tolist(),
            "macd_crossovers": df[df["macd_cross"].notna()][["symbol", "macd_cross"]].to_dict("records") if "macd_cross" in df.columns else [],
            "sma_crossovers": df[df["sma_cross"].notna()][["symbol", "sma_cross"]].to_dict("records") if "sma_cross" in df.columns else [],
        }

        json_path = self.reports_dir / f"monitor_summary_{timestamp}.json"
        json_path.write_text(json.dumps(summary, indent=2, default=str))

        return csv_path, json_path

    def build_email_html(self, df: pd.DataFrame, scan_time: datetime, scan_label: str = "Scan") -> str:
        """Build a professional HTML email report from scan results."""
        if df.empty:
            return "<p>Scan returned no results.</p>"

        top_10 = df.head(10)
        bottom_10 = df.tail(10).iloc[::-1] if len(df) >= 20 else pd.DataFrame()

        # Count signals
        overbought = df[df["rsi"] >= self.rsi_overbought] if "rsi" in df.columns else pd.DataFrame()
        oversold = df[df["rsi"] <= self.rsi_oversold] if "rsi" in df.columns else pd.DataFrame()
        high_iv = df[df["iv_rank"] >= self.iv_rank_high]
        macd_bull = df[df["macd_cross"] == "BULLISH"] if "macd_cross" in df.columns else pd.DataFrame()
        macd_bear = df[df["macd_cross"] == "BEARISH"] if "macd_cross" in df.columns else pd.DataFrame()
        golden = df[df["sma_cross"] == "GOLDEN_CROSS"] if "sma_cross" in df.columns else pd.DataFrame()
        death = df[df["sma_cross"] == "DEATH_CROSS"] if "sma_cross" in df.columns else pd.DataFrame()

        bullish_count = len(oversold) + len(macd_bull) + len(golden)
        bearish_count = len(overbought) + len(macd_bear) + len(death)
        if bullish_count > bearish_count:
            sentiment, sent_color = "BULLISH", "#00e676"
        elif bearish_count > bullish_count:
            sentiment, sent_color = "BEARISH", "#ff5252"
        else:
            sentiment, sent_color = "NEUTRAL", "#ffab00"

        def _fmt(val, fmt=".2f"):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "—"
            return f"{val:{fmt}}"

        def _color(val, positive_good=True):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "#888"
            if positive_good:
                return "#00e676" if val >= 0 else "#ff5252"
            return "#ff5252" if val >= 0 else "#00e676"

        def _rsi_color(val):
            if val is None:
                return "#888"
            if val >= 70:
                return "#ff5252"
            if val <= 30:
                return "#00e676"
            return "#e0e0e0"

        def _stock_row(row):
            chg_c = _color(row.get("day_change_pct"))
            rsi_c = _rsi_color(row.get("rsi"))
            macd_c = "#00e676" if row.get("macd_cross") == "BULLISH" else "#ff5252" if row.get("macd_cross") == "BEARISH" else "#e0e0e0"

            sma_status = ""
            if row.get("price_vs_sma200") is not None:
                if row["price_vs_sma200"] > 0:
                    sma_status = "Above 200"
                else:
                    sma_status = "Below 200"
            if row.get("sma_cross"):
                sma_status = row["sma_cross"].replace("_", " ").title()

            return f"""<tr>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a;color:#00d4ff;font-weight:bold">{row['symbol']}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">${row['price']:.2f}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a;color:{chg_c}" align="right">{row['day_change_pct']:+.2f}%</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a;color:{rsi_c}" align="right">{_fmt(row.get('rsi'), '.0f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a;color:{macd_c}" align="right">{_fmt(row.get('macd'), '.3f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('stoch_k'), '.0f')}/{_fmt(row.get('stoch_d'), '.0f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('iv_rank'), '.0f')}%</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a;font-size:12px">{sma_status}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('opt_delta'), '.3f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('opt_gamma'), '.4f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('opt_theta'), '.3f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('opt_vega'), '.3f')}</td>
                <td style="padding:6px 10px;border-bottom:1px solid #2a2a4a" align="right">{_fmt(row.get('volume_ratio'), '.1f')}x</td>
            </tr>"""

        table_header = """<tr>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:left">Symbol</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Price</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Chg%</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">RSI</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">MACD</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Stoch</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">IV Rank</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:left">SMA</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Δ</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Γ</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Θ</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">V</th>
            <th style="background:#16213e;color:#00d4ff;padding:8px 10px;text-align:right">Vol</th>
        </tr>"""

        top_rows = "\n".join(_stock_row(row) for _, row in top_10.iterrows())
        bottom_rows = "\n".join(_stock_row(row) for _, row in bottom_10.iterrows()) if not bottom_10.empty else ""

        # Alerts section
        alerts_html = ""
        alert_items = []
        if not overbought.empty:
            alert_items.append(f'<li style="color:#ff5252"><b>RSI Overbought (&gt;70):</b> {", ".join(overbought["symbol"].tolist())}</li>')
        if not oversold.empty:
            alert_items.append(f'<li style="color:#00e676"><b>RSI Oversold (&lt;30):</b> {", ".join(oversold["symbol"].tolist())}</li>')
        if not macd_bull.empty:
            alert_items.append(f'<li style="color:#00e676"><b>MACD Bullish Cross:</b> {", ".join(macd_bull["symbol"].tolist())}</li>')
        if not macd_bear.empty:
            alert_items.append(f'<li style="color:#ff5252"><b>MACD Bearish Cross:</b> {", ".join(macd_bear["symbol"].tolist())}</li>')
        if not golden.empty:
            alert_items.append(f'<li style="color:#00e676"><b>Golden Cross (SMA20&gt;50):</b> {", ".join(golden["symbol"].tolist())}</li>')
        if not death.empty:
            alert_items.append(f'<li style="color:#ff5252"><b>Death Cross (SMA20&lt;50):</b> {", ".join(death["symbol"].tolist())}</li>')
        if not high_iv.empty:
            alert_items.append(f'<li style="color:#ffab00"><b>High IV Rank (&gt;70%):</b> {", ".join(high_iv["symbol"].tolist())} — good for premium selling</li>')

        # High volume
        high_vol = df[df["volume_ratio"] >= 2.0] if "volume_ratio" in df.columns else pd.DataFrame()
        if not high_vol.empty:
            alert_items.append(f'<li style="color:#ffab00"><b>Unusual Volume (&gt;2x):</b> {", ".join(high_vol["symbol"].tolist())}</li>')

        if alert_items:
            alerts_html = f'<h2 style="color:#ffab00;margin-top:25px">Alerts &amp; Signals</h2><ul style="line-height:1.8">{"".join(alert_items)}</ul>'

        # Options opportunities
        options_html = ""
        if not high_iv.empty:
            opt_rows = "\n".join(_stock_row(row) for _, row in high_iv.head(10).iterrows())
            options_html = f"""
            <h2 style="color:#00d4ff;margin-top:25px">Options Opportunities (High IV Rank)</h2>
            <p style="color:#888">Stocks with elevated IV Rank — ideal for premium selling strategies (covered calls, cash-secured puts, iron condors).</p>
            <table style="border-collapse:collapse;width:100%;font-size:13px">{table_header}{opt_rows}</table>"""

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:20px;background:#1a1a2e;color:#e0e0e0">
    <div style="max-width:1100px;margin:0 auto">
        <h1 style="color:#00d4ff;border-bottom:2px solid #00d4ff;padding-bottom:10px">
            Market Monitor — {scan_label}
        </h1>
        <p style="color:#888">{scan_time.strftime('%A, %B %d, %Y at %I:%M %p ET')} | {len(df)} symbols scanned</p>

        <div style="display:flex;gap:15px;margin:20px 0;flex-wrap:wrap">
            <div style="background:#16213e;padding:15px 25px;border-radius:8px;text-align:center;flex:1;min-width:120px">
                <div style="font-size:28px;font-weight:bold;color:{sent_color}">{sentiment}</div>
                <div style="font-size:12px;color:#888">Market Sentiment</div>
            </div>
            <div style="background:#16213e;padding:15px 25px;border-radius:8px;text-align:center;flex:1;min-width:120px">
                <div style="font-size:28px;font-weight:bold;color:#00e676">{len(oversold)}</div>
                <div style="font-size:12px;color:#888">Oversold (RSI&lt;30)</div>
            </div>
            <div style="background:#16213e;padding:15px 25px;border-radius:8px;text-align:center;flex:1;min-width:120px">
                <div style="font-size:28px;font-weight:bold;color:#ff5252">{len(overbought)}</div>
                <div style="font-size:12px;color:#888">Overbought (RSI&gt;70)</div>
            </div>
            <div style="background:#16213e;padding:15px 25px;border-radius:8px;text-align:center;flex:1;min-width:120px">
                <div style="font-size:28px;font-weight:bold;color:#ffab00">{len(high_iv)}</div>
                <div style="font-size:12px;color:#888">High IV Rank</div>
            </div>
        </div>

        {alerts_html}

        <h2 style="color:#00d4ff;margin-top:25px">Top 10 by Composite Score</h2>
        <table style="border-collapse:collapse;width:100%;font-size:13px">{table_header}{top_rows}</table>

        {"<h2 style='color:#00d4ff;margin-top:25px'>Bottom 10</h2><table style='border-collapse:collapse;width:100%;font-size:13px'>" + table_header + bottom_rows + "</table>" if bottom_rows else ""}

        {options_html}

        <div style="margin-top:30px;padding-top:10px;border-top:1px solid #2a2a4a;color:#666;font-size:11px">
            IBKR Options Trading Bot — Market Monitor | Auto-generated report<br>
            Indicators: RSI(14), MACD(12,26,9), Stochastic(14,3,3), SMA(20,50,200) | Greeks: ATM Call Options
        </div>
    </div>
</body>
</html>"""
        return html
