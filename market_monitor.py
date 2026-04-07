"""
Market Monitor: Active market surveillance for top 100 best-performing
NYSE stocks and options.

Dynamically screens a broad universe of 300+ NYSE-listed stocks to
identify the top 100 best performers, then runs full technical analysis:
  - Options Greeks (Delta, Gamma, Theta, Vega) from live options chains
  - IV Rank (implied volatility percentile over 1 year)
  - 20, 50, 200 day Moving Averages with crossover detection
  - RSI (14-period Relative Strength Index)
  - MACD (12, 26, 9) with histogram and signal crossovers
  - Stochastic Oscillator (%K, %D) with overbought/oversold zones

Runs multiple times daily on weekdays via scheduled_monitor.py and
sends Telegram/email notifications when actionable changes are detected.
"""

import logging
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Broad universe of NYSE-listed stocks to screen from (~300 large/mid caps)
# The monitor dynamically ranks these by performance and selects the top 100
NYSE_SCREENING_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "TSM", "V", "MA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "ICE", "CME", "AIG", "MET", "PRU", "ALL", "TRV",
    "AFL", "FITB", "HBAN", "KEY", "RF", "CFG", "ZION", "NTRS", "STT",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "CI", "HUM", "ELV",
    "ZTS", "REGN", "VRTX", "IDXX", "IQV", "MTD", "A", "BAX", "BDX",
    "BSX", "EW", "HOLX", "DXCM", "ALGN",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "GIS", "K",
    "KHC", "HSY", "SJM", "MKC", "CHD", "CLX", "KMB", "STZ", "TAP",
    # Consumer Discretionary
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "EL", "TJX", "ROST",
    "DHI", "LEN", "PHM", "NVR", "GM", "F", "CMG", "YUM", "DPZ",
    "BKNG", "MAR", "HLT", "RCL", "CCL", "WYNN", "LVS", "MGM",
    # Industrial
    "CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "UNP", "FDX", "EMR", "ITW", "ETN", "ROK", "SWK", "GD", "NOC",
    "TXT", "HII", "WM", "RSG", "GWW", "FAST", "URI", "PWR", "CARR",
    "OTIS", "CSX", "NSC", "DAL", "UAL", "AAL", "LUV", "JBHT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "DVN", "FANG", "HAL", "BKR", "HES", "MRO", "APA", "CTRA",
    # Tech / Semis
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "SNPS", "CDNS",
    "MRVL", "ON", "NXPI", "SWKS", "MCHP", "KLAC", "ADI", "MPWR",
    # Software / Cloud
    "PLTR", "UBER", "SQ", "COIN", "CRWD", "PANW", "SNOW", "DDOG",
    "ZS", "NET", "MDB", "TEAM", "WDAY", "NOW", "ADBE", "INTU",
    "FTNT", "CYBR", "OKTA", "HUBS",
    # Communication / Media
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD",
    "PARA", "FOX", "LYV", "RBLX", "SNAP", "PINS",
    # REITs
    "AMT", "PLD", "CCI", "SPG", "O", "VICI", "DLR", "EQIX",
    "PSA", "EXR", "AVB", "EQR", "ARE", "MAA", "UDR", "WELL",
    # Utilities
    "NEE", "DUK", "SO", "D", "SRE", "AEP", "XEL", "WEC", "ES",
    "EXC", "ED", "DTE", "AEE", "CMS", "PEG",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "DOW",
    "NUE", "STLD", "CF", "MOS", "ALB", "FMC",
    # Other notable large-caps
    "ACN", "IBM", "CSCO", "ADP", "PAYX", "CTAS", "VRSK",
    "CPRT", "ODFL", "CHRW", "XPO", "RHI", "PAYC",
]


class MarketMonitor:
    """Monitors top 100 best-performing NYSE stocks with comprehensive
    technical and options analysis."""

    def __init__(self, config: dict, reports_dir: str = "reports"):
        self.config = config
        self.monitor_cfg = config.get("market_monitor", {})
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.reports_dir / "monitor_state.json"
        self._ranked_file = self.reports_dir / "top100_ranked.json"
        self._previous_state = self._load_state()
        self._top_count = self.monitor_cfg.get("symbols_count", 100)
        self._screening_universe = NYSE_SCREENING_UNIVERSE
        self._symbols = self._load_ranked_symbols()
        self._max_workers = self.monitor_cfg.get("max_workers", 8)

    # ---- Dynamic Top 100 Screening ----

    def _load_ranked_symbols(self) -> List[str]:
        """Load previously ranked top 100 or fall back to default."""
        try:
            if self._ranked_file.exists():
                with open(self._ranked_file) as f:
                    data = json.load(f)
                    # Refresh ranking if older than 6 hours
                    saved_at = datetime.fromisoformat(data.get("ranked_at", "2000-01-01"))
                    if (datetime.now() - saved_at).total_seconds() < 21600:
                        return data.get("symbols", self._screening_universe[:100])
        except Exception:
            pass
        return self._screening_universe[:100]

    def screen_top_performers(self) -> List[str]:
        """Screen the broad universe and rank by recent performance to find
        the top 100 best-performing NYSE stocks.

        Ranking criteria (weighted composite score):
          - 1-month return (40% weight) — recent momentum
          - 5-day return (30% weight) — short-term strength
          - Average daily volume (20% weight) — liquidity
          - Options availability (10% bonus) — tradeable options
        """
        logger.info(f"Screening {len(self._screening_universe)} stocks for top {self._top_count} performers...")

        scored = []

        def _quick_score(symbol: str) -> Optional[Dict]:
            """Fetch minimal data needed to rank a stock."""
            try:
                tk = yf.Ticker(symbol)
                hist = tk.history(period="1mo")
                if hist.empty or len(hist) < 5:
                    return None
                close = hist["Close"]
                price = float(close.iloc[-1])
                if price < 5:  # Skip penny stocks
                    return None

                ret_1mo = (close.iloc[-1] / close.iloc[0] - 1) * 100
                ret_5d = (close.iloc[-1] / close.iloc[-min(5, len(close))] - 1) * 100
                avg_vol = float(hist["Volume"].mean())

                # Check if options are available
                has_options = False
                try:
                    exps = tk.options
                    has_options = len(exps) > 0
                except Exception:
                    pass

                # Composite score
                score = (
                    float(ret_1mo) * 0.40 +
                    float(ret_5d) * 0.30 +
                    (np.log10(max(avg_vol, 1)) * 0.20) +
                    (5.0 if has_options else 0)
                )

                return {
                    "symbol": symbol,
                    "price": round(price, 2),
                    "ret_1mo": round(float(ret_1mo), 2),
                    "ret_5d": round(float(ret_5d), 2),
                    "avg_volume": round(avg_vol),
                    "has_options": has_options,
                    "score": round(score, 4),
                }
            except Exception as e:
                logger.debug(f"Screening skip {symbol}: {e}")
                return None

        # Parallel screening for speed
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_quick_score, sym): sym for sym in self._screening_universe}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    scored.append(result)
                if (i + 1) % 50 == 0:
                    logger.info(f"  Screened {i + 1}/{len(self._screening_universe)}...")

        # Rank by composite score, take top N
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_symbols = [s["symbol"] for s in scored[:self._top_count]]

        # Persist ranked list
        try:
            with open(self._ranked_file, "w") as f:
                json.dump({
                    "ranked_at": datetime.now().isoformat(),
                    "symbols": top_symbols,
                    "full_scores": scored[:self._top_count],
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rankings: {e}")

        self._symbols = top_symbols
        logger.info(f"Top {len(top_symbols)} performers identified")
        return top_symbols

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

    # ---- Options Chain Greeks ----

    @staticmethod
    def fetch_options_greeks(tk, price: float) -> Dict:
        """Fetch real options chain Greeks from yfinance for the nearest
        monthly expiration ATM strike."""
        greeks = {
            "source": "estimated",
            "call_delta": 0.50, "put_delta": -0.50,
            "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "call_iv": 0.0, "put_iv": 0.0,
            "call_bid_ask": None, "put_bid_ask": None,
            "expiry": None, "strike": None,
            "call_volume": 0, "put_volume": 0,
            "call_oi": 0, "put_oi": 0,
            "hv_20": 0.0,
        }
        try:
            expirations = tk.options
            if not expirations:
                return greeks

            # Pick expiration 20-45 days out (nearest monthly)
            today = date.today()
            target_exp = None
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 20 <= dte <= 60:
                    target_exp = exp_str
                    break
            if not target_exp:
                # Fall back to first available expiration
                target_exp = expirations[0]

            chain = tk.option_chain(target_exp)
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return greeks

            # Find ATM strike (closest to current price)
            strikes = calls["strike"].values
            atm_idx = int(np.abs(strikes - price).argmin())
            atm_strike = float(strikes[atm_idx])

            atm_call = calls.iloc[atm_idx]
            atm_put_idx = int(np.abs(puts["strike"].values - atm_strike).argmin())
            atm_put = puts.iloc[atm_put_idx]

            greeks["source"] = "live_chain"
            greeks["expiry"] = target_exp
            greeks["strike"] = atm_strike

            # Extract Greeks if available in chain data
            for field, key in [("impliedVolatility", "call_iv")]:
                if field in atm_call.index:
                    greeks[key] = round(float(atm_call[field]) * 100, 2)
            for field, key in [("impliedVolatility", "put_iv")]:
                if field in atm_put.index:
                    greeks[key] = round(float(atm_put[field]) * 100, 2)

            # Volume and open interest
            greeks["call_volume"] = int(atm_call.get("volume", 0) or 0)
            greeks["put_volume"] = int(atm_put.get("volume", 0) or 0)
            greeks["call_oi"] = int(atm_call.get("openInterest", 0) or 0)
            greeks["put_oi"] = int(atm_put.get("openInterest", 0) or 0)

            # Bid-ask spread
            greeks["call_bid_ask"] = f"{atm_call.get('bid', 0):.2f}/{atm_call.get('ask', 0):.2f}"
            greeks["put_bid_ask"] = f"{atm_put.get('bid', 0):.2f}/{atm_put.get('ask', 0):.2f}"

            # Compute Greeks from IV using Black-Scholes approximations
            iv = float(atm_call.get("impliedVolatility", 0.25) or 0.25)
            exp_date = datetime.strptime(target_exp, "%Y-%m-%d").date()
            dte = max((exp_date - today).days, 1)
            t = dte / 365.0
            sqrt_t = np.sqrt(t)

            # Delta approximation (N(d1) for ATM ~ 0.5 + adjustment)
            d1 = (np.log(price / atm_strike) + 0.5 * iv**2 * t) / (iv * sqrt_t) if iv > 0 else 0
            from scipy.stats import norm
            greeks["call_delta"] = round(float(norm.cdf(d1)), 4)
            greeks["put_delta"] = round(float(norm.cdf(d1) - 1), 4)

            # Gamma
            greeks["gamma"] = round(float(norm.pdf(d1) / (price * iv * sqrt_t)), 6) if iv > 0 else 0

            # Theta (per day)
            greeks["theta"] = round(float(
                -(price * norm.pdf(d1) * iv) / (2 * sqrt_t * 365)
            ), 2) if iv > 0 else 0

            # Vega (per 1% IV move)
            greeks["vega"] = round(float(price * sqrt_t * norm.pdf(d1) * 0.01), 2)

            # Total option chain summary (all strikes)
            total_call_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
            total_put_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
            greeks["total_call_volume"] = total_call_vol
            greeks["total_put_volume"] = total_put_vol
            greeks["put_call_ratio"] = round(total_put_vol / max(total_call_vol, 1), 2)

        except Exception as e:
            logger.debug(f"Options chain fetch failed: {e}")

        return greeks

    # ---- Data Fetching & Analysis ----

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Run full technical + options analysis on a single symbol."""
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

            # MA crossover detection
            ma_crossovers = []
            if len(close) >= 50:
                prev_ma20 = float(close.iloc[-21:-1].mean())
                prev_ma50 = float(close.iloc[-51:-1].mean())
                if ma_20 and ma_50:
                    if prev_ma20 <= prev_ma50 and ma_20 > ma_50:
                        ma_crossovers.append("MA20_CROSSED_ABOVE_MA50")
                    elif prev_ma20 >= prev_ma50 and ma_20 < ma_50:
                        ma_crossovers.append("MA20_CROSSED_BELOW_MA50")
            if len(close) >= 200 and ma_50:
                prev_ma50_2 = float(close.iloc[-51:-1].mean())
                prev_ma200 = float(close.iloc[-201:-1].mean())
                if ma_200:
                    if prev_ma50_2 <= prev_ma200 and ma_50 > ma_200:
                        ma_crossovers.append("GOLDEN_CROSS_50_200")
                    elif prev_ma50_2 >= prev_ma200 and ma_50 < ma_200:
                        ma_crossovers.append("DEATH_CROSS_50_200")

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

            # Historical Volatility
            returns = np.log(close / close.shift(1)).dropna()
            hv_20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.2

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

            # Fetch real options chain Greeks
            fetch_options = self.monitor_cfg.get("fetch_options_greeks", True)
            if fetch_options:
                greeks = self.fetch_options_greeks(tk, price)
            else:
                greeks = {
                    "source": "estimated",
                    "call_delta": 0.50, "put_delta": -0.50,
                    "gamma": round(1 / (price * hv_20 * np.sqrt(30 / 365)), 6) if hv_20 > 0 else 0,
                    "theta": round(-price * hv_20 / (2 * np.sqrt(365)), 2),
                    "vega": round(price * np.sqrt(30 / 365) * 0.01, 2),
                    "call_iv": 0.0, "put_iv": 0.0,
                }
            greeks["hv_20"] = round(hv_20 * 100, 2)

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
            signals.extend(ma_crossovers)

            # Options-specific signals
            if greeks.get("put_call_ratio", 0) > 1.5:
                signals.append("HIGH_PUT_CALL_RATIO")
            elif greeks.get("put_call_ratio", 0) < 0.5 and greeks.get("put_call_ratio", 0) > 0:
                signals.append("LOW_PUT_CALL_RATIO")
            if greeks.get("call_iv", 0) > 80:
                signals.append("EXTREME_CALL_IV")
            if greeks.get("put_iv", 0) > 80:
                signals.append("EXTREME_PUT_IV")

            # Performance ranking score
            perf_score = daily_change_pct * 0.3 + (rsi - 50) * 0.1

            return {
                "symbol": symbol,
                "price": round(price, 2),
                "daily_change_pct": round(daily_change_pct, 2),
                "ma_20": round(ma_20, 2) if ma_20 else None,
                "ma_50": round(ma_50, 2) if ma_50 else None,
                "ma_200": round(ma_200, 2) if ma_200 else None,
                "ma_crossovers": ma_crossovers,
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
                "perf_score": round(perf_score, 2),
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return None

    def run_full_scan(self, rescreen: bool = False) -> List[Dict]:
        """Analyze all top 100 symbols with parallel execution.

        Args:
            rescreen: If True, re-run the dynamic screener to find fresh
                      top 100 performers before analysis (used for morning scan).
        """
        if rescreen:
            self.screen_top_performers()

        logger.info(f"Starting full analysis of {len(self._symbols)} NYSE stocks...")
        results = []
        completed = 0

        def _analyze_wrapper(symbol: str) -> Optional[Dict]:
            return self.analyze_symbol(symbol)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_analyze_wrapper, sym): sym for sym in self._symbols}
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    results.append(result)
                if completed % 20 == 0:
                    logger.info(f"  Analyzed {completed}/{len(self._symbols)} symbols...")

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

    def generate_summary(self, results: List[Dict], changes: List[Dict],
                         session_type: str = "") -> str:
        """Generate a formatted text summary for notification."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        lines = [
            f"{'=' * 55}",
            f"  MARKET MONITOR — {session_type or 'SCAN SUMMARY'}",
            f"  {now} | {len(results)} Stocks Analyzed",
            f"{'=' * 55}",
            "",
        ]

        # Market breadth snapshot
        bullish = len([s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")])
        bearish = len([s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")])
        avg_rsi = np.mean([s["rsi"] for s in results]) if results else 50
        avg_iv = np.mean([s["iv_rank"] for s in results]) if results else 50
        lines.append("--- MARKET BREADTH ---")
        lines.append(f"  Bullish: {bullish} | Bearish: {bearish} | Neutral: {len(results) - bullish - bearish}")
        lines.append(f"  Avg RSI: {avg_rsi:.1f} | Avg IV Rank: {avg_iv:.1f}")
        lines.append("")

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
            lines.append(f"  {'Symbol':6s} {'IV':>5s} {'HV20':>6s} {'Theta':>7s} {'Vega':>6s} {'P/C':>5s} {'CallIV':>7s} {'PutIV':>6s}")
            high_iv.sort(key=lambda x: x["iv_rank"], reverse=True)
            for stock in high_iv[:15]:
                g = stock["greeks"]
                lines.append(
                    f"  {stock['symbol']:6s} {stock['iv_rank']:5.0f} "
                    f"{g['hv_20']:5.1f}% "
                    f"${g.get('theta', 0):6.2f} "
                    f"${g.get('vega', 0):5.2f} "
                    f"{g.get('put_call_ratio', 0):5.2f} "
                    f"{g.get('call_iv', 0):6.1f}% "
                    f"{g.get('put_iv', 0):5.1f}%"
                )
            lines.append("")

        # Options opportunities — premium selling candidates
        premium_sell = [s for s in results
                        if s["iv_rank"] > 50 and s.get("greeks", {}).get("source") == "live_chain"]
        if premium_sell:
            lines.append(f"--- PREMIUM SELLING OPPORTUNITIES ({len(premium_sell)}) ---")
            premium_sell.sort(key=lambda x: x["iv_rank"], reverse=True)
            for s in premium_sell[:10]:
                g = s["greeks"]
                lines.append(
                    f"  {s['symbol']:6s} IV:{s['iv_rank']:.0f} "
                    f"Strike:${g.get('strike', 0):.0f} "
                    f"Exp:{g.get('expiry', 'N/A')} "
                    f"CallBid/Ask:{g.get('call_bid_ask', '-')} "
                    f"PutBid/Ask:{g.get('put_bid_ask', '-')} "
                    f"Trend:{s['trend']}"
                )
            lines.append("")

        # Overbought/Oversold
        overbought = [s for s in results if s["rsi"] > 70]
        oversold = [s for s in results if s["rsi"] < 30]
        if overbought:
            lines.append(f"--- RSI OVERBOUGHT (>70) — {len(overbought)} STOCKS ---")
            for s in overbought[:10]:
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} Stoch:{s['stochastic']['k']:.0f} Trend:{s['trend']}")
            lines.append("")
        if oversold:
            lines.append(f"--- RSI OVERSOLD (<30) — {len(oversold)} STOCKS ---")
            for s in oversold[:10]:
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} Stoch:{s['stochastic']['k']:.0f} Trend:{s['trend']}")
            lines.append("")

        # MACD crossover signals
        bullish_macd = [s for s in results if "MACD_BULLISH" in s["signals"]]
        if bullish_macd:
            lines.append(f"--- MACD BULLISH CROSSOVER — {len(bullish_macd)} STOCKS ---")
            for s in bullish_macd[:10]:
                lines.append(
                    f"  {s['symbol']:6s} MACD:{s['macd']['macd']:.4f} "
                    f"Signal:{s['macd']['signal']:.4f} Hist:{s['macd']['histogram']:.4f}"
                )
            lines.append("")

        # MA Crossovers (golden/death crosses)
        ma_cross_stocks = [s for s in results if s.get("ma_crossovers")]
        if ma_cross_stocks:
            lines.append(f"--- MOVING AVERAGE CROSSOVERS ({len(ma_cross_stocks)}) ---")
            for s in ma_cross_stocks:
                for cross in s["ma_crossovers"]:
                    lines.append(
                        f"  {s['symbol']:6s} {cross} "
                        f"MA20:{s['ma_20']} MA50:{s['ma_50']} MA200:{s['ma_200'] or 'N/A'}"
                    )
            lines.append("")

        # Golden cross 20/50
        golden = [s for s in results if "GOLDEN_CROSS_20_50" in s["signals"]]
        if golden:
            lines.append(f"--- MA20 > MA50 — {len(golden)} STOCKS ---")
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

        # Full Greeks summary for top 20 by IV
        lines.append("--- OPTIONS GREEKS (Top 20 by IV Rank) ---")
        lines.append(f"  {'Sym':6s} {'IV':>5s} {'HV20':>6s} {'Delta':>6s} {'Gamma':>8s} {'Theta':>7s} {'Vega':>6s} {'P/C':>5s} {'Src':>5s}")
        by_iv = sorted(results, key=lambda x: x["iv_rank"], reverse=True)
        for s in by_iv[:20]:
            g = s["greeks"]
            lines.append(
                f"  {s['symbol']:6s} {s['iv_rank']:5.1f} "
                f"{g['hv_20']:5.1f}% "
                f"{g.get('call_delta', 0.50):6.3f} "
                f"{g.get('gamma', 0):8.6f} "
                f"{g.get('theta', 0):>7.2f} "
                f"{g.get('vega', 0):>6.2f} "
                f"{g.get('put_call_ratio', 0):>5.2f} "
                f"{'LIVE' if g.get('source') == 'live_chain' else 'EST':>5s}"
            )
        lines.append("")

        # High put/call ratio (unusual options activity)
        high_pc = [s for s in results if s.get("greeks", {}).get("put_call_ratio", 0) > 1.5]
        if high_pc:
            lines.append(f"--- UNUSUAL PUT/CALL RATIO (>1.5) — {len(high_pc)} STOCKS ---")
            high_pc.sort(key=lambda x: x["greeks"].get("put_call_ratio", 0), reverse=True)
            for s in high_pc[:10]:
                g = s["greeks"]
                lines.append(
                    f"  {s['symbol']:6s} P/C:{g['put_call_ratio']:.2f} "
                    f"CallVol:{g.get('total_call_volume', 0):,} "
                    f"PutVol:{g.get('total_put_volume', 0):,}"
                )
            lines.append("")

        lines.append(f"{'=' * 55}")
        lines.append(f"  END MARKET MONITOR")
        lines.append(f"{'=' * 55}")
        return "\n".join(lines)

    def generate_html_report(self, results: List[Dict], changes: List[Dict]) -> str:
        """Generate a full HTML report for email with options chain data."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

        # Market breadth stats
        bullish = len([s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")])
        bearish = len([s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")])
        avg_rsi = np.mean([s["rsi"] for s in results]) if results else 50
        avg_iv = np.mean([s["iv_rank"] for s in results]) if results else 50
        live_greeks_count = len([s for s in results if s.get("greeks", {}).get("source") == "live_chain"])

        # Build HTML rows for top performers
        top_rows = ""
        for s in results[:25]:
            g = s["greeks"]
            change_color = "color:#00e676" if s["daily_change_pct"] >= 0 else "color:#ff5252"
            src_badge = '<span style="color:#00e676">LIVE</span>' if g.get("source") == "live_chain" else '<span style="color:#888">EST</span>'
            top_rows += f"""<tr>
                <td><b>{s['symbol']}</b></td>
                <td>${s['price']:.2f}</td>
                <td style="{change_color}">{s['daily_change_pct']:+.2f}%</td>
                <td>{s['rsi']:.1f}</td>
                <td>{s['iv_rank']:.1f}</td>
                <td>{s['ma_20'] or '-'}</td>
                <td>{s['ma_50'] or '-'}</td>
                <td>{s['ma_200'] or '-'}</td>
                <td>{s['macd']['histogram']:+.4f}</td>
                <td>{s['stochastic']['k']:.0f}/{s['stochastic']['d']:.0f}</td>
                <td>{g.get('call_delta', 0.5):.3f}</td>
                <td>{g.get('gamma', 0):.5f}</td>
                <td>{g.get('theta', 0):.2f}</td>
                <td>{g.get('vega', 0):.2f}</td>
                <td>{g.get('call_iv', 0):.1f}%</td>
                <td>{g.get('put_call_ratio', 0):.2f}</td>
                <td>{s['trend']}</td>
                <td>{', '.join(s['signals'][:3]) or '-'}</td>
            </tr>"""

        # Options opportunities table
        options_rows = ""
        high_iv_stocks = sorted([s for s in results if s["iv_rank"] > 50],
                                key=lambda x: x["iv_rank"], reverse=True)
        for s in high_iv_stocks[:20]:
            g = s["greeks"]
            options_rows += f"""<tr>
                <td><b>{s['symbol']}</b></td>
                <td>${s['price']:.2f}</td>
                <td>{s['iv_rank']:.1f}</td>
                <td>{g['hv_20']:.1f}%</td>
                <td>{g.get('call_iv', 0):.1f}%</td>
                <td>{g.get('put_iv', 0):.1f}%</td>
                <td>{g.get('call_bid_ask', '-')}</td>
                <td>{g.get('put_bid_ask', '-')}</td>
                <td>{g.get('strike', '-')}</td>
                <td>{g.get('expiry', '-')}</td>
                <td>{g.get('theta', 0):.2f}</td>
                <td>{g.get('put_call_ratio', 0):.2f}</td>
                <td>{s['trend']}</td>
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
        h3 {{ color: #aaa; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 12px; }}
        th {{ background: #16213e; color: #00d4ff; padding: 8px; text-align: left; position: sticky; top: 0; }}
        td {{ padding: 5px 8px; border-bottom: 1px solid #2a2a4a; }}
        tr:hover {{ background: #16213e; }}
        .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 15px 0; }}
        .stat-card {{ background: #16213e; padding: 12px 18px; border-radius: 8px; text-align: center; min-width: 120px; }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ font-size: 10px; color: #888; margin-top: 4px; }}
        .green {{ color: #00e676; }}
        .red {{ color: #ff5252; }}
    </style>
</head>
<body>
    <h1>Market Monitor Report</h1>
    <p>Generated: {now} | Top {len(results)} Best-Performing NYSE Stocks</p>

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
            <div class="stat-value">{bullish}</div>
            <div class="stat-label">Bullish Trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{bearish}</div>
            <div class="stat-label">Bearish Trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_rsi:.0f}</div>
            <div class="stat-label">Avg RSI</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_iv:.0f}</div>
            <div class="stat-label">Avg IV Rank</div>
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
            <div class="stat-value">{live_greeks_count}</div>
            <div class="stat-label">Live Options Data</div>
        </div>
    </div>

    <h2>Top 25 Performers — Full Technical + Options Analysis</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Price</th><th>Chg%</th><th>RSI</th><th>IV Rank</th>
            <th>MA20</th><th>MA50</th><th>MA200</th><th>MACD Hist</th>
            <th>Stoch</th><th>Delta</th><th>Gamma</th><th>Theta</th><th>Vega</th>
            <th>Call IV</th><th>P/C</th><th>Trend</th><th>Signals</th>
        </tr>
        {top_rows}
    </table>

    <h2>Options Opportunities — Top 20 by IV Rank</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Price</th><th>IV Rank</th><th>HV20</th>
            <th>Call IV</th><th>Put IV</th><th>Call Bid/Ask</th><th>Put Bid/Ask</th>
            <th>ATM Strike</th><th>Expiry</th><th>Theta/Day</th><th>P/C Ratio</th><th>Trend</th>
        </tr>
        {options_rows}
    </table>

    {"<h2>Significant Changes</h2><table><tr><th>Symbol</th><th>Type</th><th>Price</th><th>Detail</th></tr>" + change_rows + "</table>" if changes else ""}

    <p style="color:#666; font-size:11px; margin-top:30px;">
        IBKR Options Trading Bot — Market Monitor | Dynamically ranked top performers | Auto-generated
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

    def run_scan_and_report(self, session_type: str = "",
                            rescreen: bool = False) -> Tuple[str, str, List[Dict]]:
        """
        Run full scan, detect changes, generate reports.

        Args:
            session_type: Label for this scan session (e.g. "Morning Briefing")
            rescreen: Re-run dynamic top 100 screening before analysis

        Returns:
            (text_summary, html_report, changes)
        """
        results = self.run_full_scan(rescreen=rescreen)
        changes = self.detect_changes(results)

        text_summary = self.generate_summary(results, changes, session_type)
        html_report = self.generate_html_report(results, changes)

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        html_path = self.reports_dir / f"market_monitor_{timestamp}.html"
        html_path.write_text(html_report)
        logger.info(f"HTML report saved: {html_path}")

        # Update state for next comparison
        self._save_state(results)

        return text_summary, html_report, changes


def run_standalone():
    """Run the market monitor as a standalone script."""
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

    print(text_summary)

    if changes:
        print(f"\n[!] {len(changes)} significant changes detected.")


if __name__ == "__main__":
    run_standalone()
