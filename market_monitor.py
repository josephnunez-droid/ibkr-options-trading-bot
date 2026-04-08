"""
Market Monitor: Tracks top 100 NYSE stocks with full technical analysis.

Analyzes: Greeks, IV Rank, 20/50/200 day MA, RSI, MACD, Stochastics
Generates summaries and detects significant changes for notifications.
"""

import logging
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Top 100 NYSE-listed large-cap stocks to monitor
NYSE_TOP_100 = [
    # Mega-cap tech (NYSE-listed)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "TSM", "V", "MA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "ICE", "CME",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK",
    # Consumer
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX",
    "TGT", "LOW", "EL", "CL", "GIS",
    # Industrial
    "CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "UNP", "FDX", "EMR",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    # Tech / Semis
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "SNPS", "CDNS",
    # Communication / Media
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    # REITs / Utilities / Other
    "NEE", "DUK", "SO", "D", "SRE",
    "AMT", "PLD", "CCI", "SPG",
    "PLTR", "UBER", "SQ", "COIN", "CRWD", "PANW",
]


class MarketMonitor:
    """Monitors top 100 NYSE stocks with comprehensive technical analysis."""

    def __init__(self, config: dict, reports_dir: str = "reports"):
        self.config = config
        self.monitor_cfg = config.get("market_monitor", {})
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.reports_dir / "monitor_state.json"
        self._previous_state = self._load_state()
        self._symbols = NYSE_TOP_100[:100]

    # ---- Technical Indicator Calculations ----

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0

    @staticmethod
    def calc_macd(series: pd.Series) -> Dict[str, float]:
        """Calculate MACD (12, 26, 9)."""
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": float(macd_line.iloc[-1]),
            "signal": float(signal_line.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
        }

    @staticmethod
    def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calculate Stochastic Oscillator (%K and %D)."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        k = 100 * (close - lowest_low) / denom
        d = k.rolling(window=d_period).mean()
        return {
            "k": float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0,
            "d": float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0,
        }

    @staticmethod
    def calc_iv_rank(hist: pd.DataFrame) -> float:
        """Calculate IV Rank from historical data."""
        if hist.empty or len(hist) < 30:
            return 50.0
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(returns) < 20:
            return 50.0
        rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) < 2:
            return 50.0
        current = rolling_vol.iloc[-1]
        low = rolling_vol.min()
        high = rolling_vol.max()
        if high == low:
            return 50.0
        rank = (current - low) / (high - low) * 100
        return float(max(0, min(100, rank)))

    # ---- Data Fetching & Analysis ----

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Run full technical analysis on a single symbol."""
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1y")
            if hist.empty or len(hist) < 200:
                hist_shorter = tk.history(period="6mo")
                if hist_shorter.empty or len(hist_shorter) < 50:
                    return None
                hist = hist_shorter

            close = hist["Close"]
            high = hist["High"]
            low = hist["Low"]
            price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) > 1 else price
            daily_change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

            # Moving Averages
            ma_20 = float(close.tail(20).mean()) if len(close) >= 20 else None
            ma_50 = float(close.tail(50).mean()) if len(close) >= 50 else None
            ma_200 = float(close.tail(200).mean()) if len(close) >= 200 else None

            # RSI
            rsi = self.calc_rsi(close)

            # MACD
            macd = self.calc_macd(close)

            # Stochastics
            stoch = self.calc_stochastic(high, low, close)

            # IV Rank
            iv_rank = self.calc_iv_rank(hist)

            # Volume
            avg_volume = float(hist["Volume"].tail(20).mean())
            current_volume = float(hist["Volume"].iloc[-1])

            # Trend signals
            trend = "NEUTRAL"
            if ma_50 and ma_200:
                if ma_50 > ma_200 and price > ma_50:
                    trend = "BULLISH"
                elif ma_50 < ma_200 and price < ma_50:
                    trend = "BEARISH"
                elif price > ma_200:
                    trend = "MODERATELY BULLISH"
                else:
                    trend = "MODERATELY BEARISH"

            # Options Greeks (ATM approximation using IV)
            returns = np.log(close / close.shift(1)).dropna()
            hv_20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.2
            # Simplified ATM greeks estimate
            greeks = {
                "est_delta_call": 0.50,
                "est_delta_put": -0.50,
                "est_gamma": round(1 / (price * hv_20 * np.sqrt(30 / 365)), 6) if hv_20 > 0 else 0,
                "est_theta_daily": round(-price * hv_20 / (2 * np.sqrt(365)), 2),
                "est_vega": round(price * np.sqrt(30 / 365) * 0.01, 2),
                "hv_20": round(hv_20 * 100, 2),
            }

            # Signal flags
            signals = []
            if rsi > 70:
                signals.append("RSI_OVERBOUGHT")
            elif rsi < 30:
                signals.append("RSI_OVERSOLD")
            if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
                signals.append("MACD_BULLISH")
            elif macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
                signals.append("MACD_BEARISH")
            if stoch["k"] > 80:
                signals.append("STOCH_OVERBOUGHT")
            elif stoch["k"] < 20:
                signals.append("STOCH_OVERSOLD")
            if ma_20 and ma_50 and ma_20 > ma_50:
                signals.append("GOLDEN_CROSS_20_50")
            if iv_rank > 70:
                signals.append("HIGH_IV_RANK")
            elif iv_rank < 20:
                signals.append("LOW_IV_RANK")

            return {
                "symbol": symbol,
                "price": round(price, 2),
                "daily_change_pct": round(daily_change_pct, 2),
                "ma_20": round(ma_20, 2) if ma_20 else None,
                "ma_50": round(ma_50, 2) if ma_50 else None,
                "ma_200": round(ma_200, 2) if ma_200 else None,
                "rsi": round(rsi, 2),
                "macd": {k: round(v, 4) for k, v in macd.items()},
                "stochastic": {k: round(v, 2) for k, v in stoch.items()},
                "iv_rank": round(iv_rank, 2),
                "greeks": greeks,
                "trend": trend,
                "signals": signals,
                "avg_volume": round(avg_volume),
                "current_volume": round(current_volume),
                "volume_ratio": round(current_volume / avg_volume, 2) if avg_volume > 0 else 0,
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return None

    def run_full_scan(self) -> List[Dict]:
        """Analyze all 100 symbols and return sorted results."""
        logger.info(f"Starting full market scan of {len(self._symbols)} NYSE stocks...")
        results = []

        for i, symbol in enumerate(self._symbols):
            try:
                result = self.analyze_symbol(symbol)
                if result:
                    results.append(result)
                if (i + 1) % 20 == 0:
                    logger.info(f"  Scanned {i + 1}/{len(self._symbols)} symbols...")
            except Exception as e:
                logger.debug(f"Skipped {symbol}: {e}")

        # Sort by daily performance (best first)
        results.sort(key=lambda x: x["daily_change_pct"], reverse=True)

        logger.info(f"Scan complete: {len(results)} stocks analyzed")
        return results

    def detect_changes(self, current: List[Dict]) -> List[Dict]:
        """Compare current scan to previous state and flag significant changes."""
        changes = []
        prev_map = {s["symbol"]: s for s in self._previous_state}

        for stock in current:
            sym = stock["symbol"]
            prev = prev_map.get(sym)

            if not prev:
                if stock["signals"]:
                    changes.append({
                        "symbol": sym,
                        "type": "NEW_SIGNALS",
                        "detail": f"New signals: {', '.join(stock['signals'])}",
                        "price": stock["price"],
                    })
                continue

            # Detect new signal transitions
            old_signals = set(prev.get("signals", []))
            new_signals = set(stock["signals"])
            added = new_signals - old_signals
            if added:
                changes.append({
                    "symbol": sym,
                    "type": "SIGNAL_CHANGE",
                    "detail": f"New: {', '.join(added)}",
                    "price": stock["price"],
                })

            # Detect large price moves (>3% since last scan)
            prev_price = prev.get("price", 0)
            if prev_price > 0:
                move_pct = abs(stock["price"] - prev_price) / prev_price * 100
                if move_pct > 3:
                    direction = "UP" if stock["price"] > prev_price else "DOWN"
                    changes.append({
                        "symbol": sym,
                        "type": "LARGE_MOVE",
                        "detail": f"{direction} {move_pct:.1f}% (${prev_price:.2f} -> ${stock['price']:.2f})",
                        "price": stock["price"],
                    })

            # Detect IV rank regime change
            prev_iv = prev.get("iv_rank", 50)
            curr_iv = stock["iv_rank"]
            if (prev_iv < 50 and curr_iv >= 70) or (prev_iv >= 50 and curr_iv < 20):
                changes.append({
                    "symbol": sym,
                    "type": "IV_REGIME_CHANGE",
                    "detail": f"IV Rank: {prev_iv:.0f} -> {curr_iv:.0f}",
                    "price": stock["price"],
                })

            # Detect trend change
            if prev.get("trend") != stock["trend"]:
                changes.append({
                    "symbol": sym,
                    "type": "TREND_CHANGE",
                    "detail": f"Trend: {prev.get('trend', 'N/A')} -> {stock['trend']}",
                    "price": stock["price"],
                })

        return changes

    def generate_summary(self, results: List[Dict], changes: List[Dict]) -> str:
        """Generate a formatted text summary for notification."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        lines = [
            f"=== MARKET MONITOR SUMMARY ===",
            f"Scan Time: {now}",
            f"Stocks Analyzed: {len(results)}",
            "",
        ]

        # Top 10 performers
        lines.append("--- TOP 10 PERFORMERS ---")
        for stock in results[:10]:
            ma_str = ""
            if stock["ma_20"]:
                ma_str = f" | MA20:{stock['ma_20']}"
            if stock["ma_50"]:
                ma_str += f" MA50:{stock['ma_50']}"
            if stock["ma_200"]:
                ma_str += f" MA200:{stock['ma_200']}"
            lines.append(
                f"  {stock['symbol']:6s} ${stock['price']:>8.2f} "
                f"({stock['daily_change_pct']:+.2f}%) "
                f"RSI:{stock['rsi']:.0f} IV:{stock['iv_rank']:.0f} "
                f"Stoch:{stock['stochastic']['k']:.0f}/{stock['stochastic']['d']:.0f}"
                f"{ma_str}"
            )
        lines.append("")

        # Bottom 10 performers
        lines.append("--- BOTTOM 10 PERFORMERS ---")
        for stock in results[-10:]:
            lines.append(
                f"  {stock['symbol']:6s} ${stock['price']:>8.2f} "
                f"({stock['daily_change_pct']:+.2f}%) "
                f"RSI:{stock['rsi']:.0f} IV:{stock['iv_rank']:.0f}"
            )
        lines.append("")

        # High IV Rank stocks (options opportunity)
        high_iv = [s for s in results if s["iv_rank"] > 60]
        if high_iv:
            lines.append(f"--- HIGH IV RANK (>60) — {len(high_iv)} STOCKS ---")
            high_iv.sort(key=lambda x: x["iv_rank"], reverse=True)
            for stock in high_iv[:15]:
                lines.append(
                    f"  {stock['symbol']:6s} IV:{stock['iv_rank']:.0f} "
                    f"HV20:{stock['greeks']['hv_20']:.1f}% "
                    f"Theta~${stock['greeks']['est_theta_daily']:.2f}/day "
                    f"Vega~${stock['greeks']['est_vega']:.2f}"
                )
            lines.append("")

        # Overbought/Oversold
        overbought = [s for s in results if s["rsi"] > 70]
        oversold = [s for s in results if s["rsi"] < 30]
        if overbought:
            lines.append(f"--- RSI OVERBOUGHT (>70) — {len(overbought)} STOCKS ---")
            for s in overbought[:10]:
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} Stoch:{s['stochastic']['k']:.0f}")
            lines.append("")
        if oversold:
            lines.append(f"--- RSI OVERSOLD (<30) — {len(oversold)} STOCKS ---")
            for s in oversold[:10]:
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} Stoch:{s['stochastic']['k']:.0f}")
            lines.append("")

        # MACD crossover signals
        bullish_macd = [s for s in results if "MACD_BULLISH" in s["signals"]]
        bearish_macd = [s for s in results if "MACD_BEARISH" in s["signals"]]
        if bullish_macd:
            lines.append(f"--- MACD BULLISH CROSSOVER — {len(bullish_macd)} STOCKS ---")
            for s in bullish_macd[:10]:
                lines.append(
                    f"  {s['symbol']:6s} MACD:{s['macd']['macd']:.4f} "
                    f"Signal:{s['macd']['signal']:.4f} Hist:{s['macd']['histogram']:.4f}"
                )
            lines.append("")

        # Golden crosses
        golden = [s for s in results if "GOLDEN_CROSS_20_50" in s["signals"]]
        if golden:
            lines.append(f"--- MA20 > MA50 CROSSOVER — {len(golden)} STOCKS ---")
            for s in golden[:10]:
                lines.append(
                    f"  {s['symbol']:6s} MA20:{s['ma_20']} MA50:{s['ma_50']} "
                    f"MA200:{s['ma_200'] or 'N/A'} Trend:{s['trend']}"
                )
            lines.append("")

        # Changes / Alerts
        if changes:
            lines.append(f"--- CHANGES SINCE LAST SCAN ({len(changes)}) ---")
            for c in changes[:20]:
                lines.append(f"  [{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}")
            lines.append("")

        # Full Greeks summary for top 20
        lines.append("--- OPTIONS GREEKS ESTIMATES (Top 20 by IV) ---")
        by_iv = sorted(results, key=lambda x: x["iv_rank"], reverse=True)
        for s in by_iv[:20]:
            g = s["greeks"]
            lines.append(
                f"  {s['symbol']:6s} IV:{s['iv_rank']:5.1f} "
                f"HV20:{g['hv_20']:5.1f}% "
                f"Gamma:{g['est_gamma']:.6f} "
                f"Theta:{g['est_theta_daily']:>7.2f} "
                f"Vega:{g['est_vega']:>6.2f}"
            )
        lines.append("")

        lines.append("=== END MARKET MONITOR ===")
        return "\n".join(lines)

    def generate_html_report(self, results: List[Dict], changes: List[Dict]) -> str:
        """Generate a full HTML report for email."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

        # Build HTML rows for top performers
        top_rows = ""
        for s in results[:25]:
            change_color = "color:#00e676" if s["daily_change_pct"] >= 0 else "color:#ff5252"
            top_rows += f"""<tr>
                <td>{s['symbol']}</td>
                <td>${s['price']:.2f}</td>
                <td style="{change_color}">{s['daily_change_pct']:+.2f}%</td>
                <td>{s['rsi']:.1f}</td>
                <td>{s['iv_rank']:.1f}</td>
                <td>{s['ma_20'] or '-'}</td>
                <td>{s['ma_50'] or '-'}</td>
                <td>{s['ma_200'] or '-'}</td>
                <td>{s['macd']['histogram']:+.4f}</td>
                <td>{s['stochastic']['k']:.0f}/{s['stochastic']['d']:.0f}</td>
                <td>{s['greeks']['hv_20']:.1f}%</td>
                <td>{s['greeks']['est_theta_daily']:.2f}</td>
                <td>{s['trend']}</td>
                <td>{', '.join(s['signals']) or '-'}</td>
            </tr>"""

        change_rows = ""
        for c in changes[:30]:
            change_rows += f"""<tr>
                <td>{c['symbol']}</td>
                <td>{c['type']}</td>
                <td>${c['price']:.2f}</td>
                <td>{c['detail']}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Market Monitor Report - {now}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #00d4ff; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
        th {{ background: #16213e; color: #00d4ff; padding: 8px; text-align: left; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #2a2a4a; }}
        tr:hover {{ background: #16213e; }}
        .stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 15px 0; }}
        .stat-card {{ background: #16213e; padding: 15px 20px; border-radius: 8px; text-align: center; min-width: 140px; }}
        .stat-value {{ font-size: 22px; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ font-size: 11px; color: #888; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>Market Monitor Report</h1>
    <p>Generated: {now} | Stocks Analyzed: {len(results)}</p>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(results)}</div>
            <div class="stat-label">Stocks Scanned</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(changes)}</div>
            <div class="stat-label">Changes Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len([s for s in results if s['rsi'] > 70])}</div>
            <div class="stat-label">RSI Overbought</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len([s for s in results if s['rsi'] < 30])}</div>
            <div class="stat-label">RSI Oversold</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len([s for s in results if s['iv_rank'] > 60])}</div>
            <div class="stat-label">High IV Rank</div>
        </div>
    </div>

    <h2>Top 25 Performers — Full Technical Analysis</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Price</th><th>Change</th><th>RSI</th><th>IV Rank</th>
            <th>MA20</th><th>MA50</th><th>MA200</th><th>MACD Hist</th>
            <th>Stoch K/D</th><th>HV20</th><th>Theta</th><th>Trend</th><th>Signals</th>
        </tr>
        {top_rows}
    </table>

    {"<h2>Significant Changes</h2><table><tr><th>Symbol</th><th>Type</th><th>Price</th><th>Detail</th></tr>" + change_rows + "</table>" if changes else ""}

    <p style="color:#666; font-size:11px; margin-top:30px;">
        IBKR Options Trading Bot — Market Monitor | Auto-generated report
    </p>
</body>
</html>"""
        return html

    # ---- State Management ----

    def _load_state(self) -> List[Dict]:
        """Load previous scan state."""
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"No previous state: {e}")
        return []

    def _save_state(self, results: List[Dict]):
        """Persist current scan state for change detection."""
        try:
            # Save only essential fields to keep file small
            slim = []
            for r in results:
                slim.append({
                    "symbol": r["symbol"],
                    "price": r["price"],
                    "rsi": r["rsi"],
                    "iv_rank": r["iv_rank"],
                    "trend": r["trend"],
                    "signals": r["signals"],
                    "macd": r["macd"],
                    "stochastic": r["stochastic"],
                })
            with open(self._state_file, "w") as f:
                json.dump(slim, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    # ---- Main Entry Points ----

    def run_scan_and_report(self) -> Tuple[str, str, List[Dict]]:
        """
        Run full scan, detect changes, generate reports.

        Returns:
            (text_summary, html_report, changes)
        """
        results = self.run_full_scan()
        self._last_results = results  # Cache for callers that need DataFrame
        changes = self.detect_changes(results)

        text_summary = self.generate_summary(results, changes)
        html_report = self.generate_html_report(results, changes)

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        html_path = self.reports_dir / f"market_monitor_{timestamp}.html"
        html_path.write_text(html_report)
        logger.info(f"HTML report saved: {html_path}")

        # Update state for next comparison
        self._save_state(results)

        return text_summary, html_report, changes


def run_scan(session_label: str = "Market Scan", include_options: bool = True) -> tuple:
    """
    Module-level convenience function for monitor_runner.py and other callers.

    Args:
        session_label: Human-readable label for this scan session.
        include_options: Whether to include options greeks estimates.

    Returns:
        (df, text_summary, html_report) where df is a pandas DataFrame of results.
    """
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(__file__).parent / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    monitor = MarketMonitor(config)
    text_summary, html_report, changes = monitor.run_scan_and_report()

    # Prepend session label to the text summary
    header = f"[{session_label}] — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}\n\n"
    text_summary = header + text_summary

    # Convert results to DataFrame for callers that need tabular data
    results = monitor.run_full_scan() if not hasattr(monitor, '_last_results') else monitor._last_results
    df = pd.DataFrame(results) if results else pd.DataFrame()

    return df, text_summary, html_report


def run_standalone():
    """Run the market monitor as a standalone script."""
    df, text_summary, html_report = run_scan(session_label="Standalone Scan")
    print(text_summary)

    if not df.empty:
        advancing = len(df[df["daily_change_pct"] > 0])
        declining = len(df[df["daily_change_pct"] < 0])
        print(f"\n[!] {advancing} advancing, {declining} declining")


if __name__ == "__main__":
    run_standalone()
