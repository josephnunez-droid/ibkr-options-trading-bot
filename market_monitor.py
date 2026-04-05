"""
Market Monitor: Actively tracks top 100 NYSE stocks with full technical
and options analysis.  Dynamically discovers the best-performing stocks
from a broad universe of ~300 large-cap NYSE names and ranks them by a
composite performance score.

Analyzes: Options Greeks (real chain data), IV Rank, 20/50/200 day MA,
RSI, MACD, Stochastics, unusual volume, MA crossovers.

Generates summaries, morning briefings, and detects significant changes
for proactive notifications.
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

# ---------------------------------------------------------------------------
# Broad NYSE universe (~300 large-cap stocks) for dynamic discovery
# ---------------------------------------------------------------------------

NYSE_BROAD_UNIVERSE = [
    # Mega-cap tech (NYSE-listed)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "TSM", "V", "MA", "AVGO", "ORCL", "CRM", "ACN", "ADBE",
    "CSCO", "IBM", "NOW", "INTU", "SHOP", "SNOW", "UBER", "SQ",
    # Semiconductors
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "SNPS", "CDNS",
    "MRVL", "ON", "NXPI", "KLAC", "ADI", "MPWR", "MCHP", "SWKS",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "ICE", "CME", "MCO", "SPGI", "MMC", "AIG",
    "MET", "PRU", "ALL", "TRV", "AMP", "BK", "STT", "FITB", "RF",
    "HBAN", "KEY", "CFG", "ZION", "FRC",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "BDX", "ZTS", "CI",
    "HUM", "ELV", "CVS", "MCK", "CAH", "ABC", "BAX", "BSX", "EW",
    "REGN", "VRTX", "IDXX", "IQV", "A", "DXCM", "HOLX", "MTD",
    # Consumer Discretionary
    "HD", "LOW", "NKE", "SBUX", "TGT", "TJX", "ROST", "DHI", "LEN",
    "PHM", "NVR", "GM", "F", "TSCO", "AZO", "ORLY", "BBY", "DG",
    "DLTR", "YUM", "DPZ", "CMG", "DARDEN", "MAR", "HLT", "WYNN",
    "LVS", "MGM", "RCL", "CCL", "NCLH", "BKNG",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "MCD", "EL", "CL", "GIS",
    "K", "HSY", "SJM", "MKC", "CHD", "KMB", "CLX", "CPB", "CAG",
    "MDLZ", "MNST", "KDP", "STZ", "BF-B", "TAP", "SAM", "PM", "MO",
    "ADM", "BG", "TSN", "HRL",
    # Industrials
    "CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "UNP", "FDX", "EMR", "ETN", "ITW", "PH", "ROK", "CMI", "PCAR",
    "WM", "RSG", "GD", "NOC", "TXT", "HWM", "TDG", "IR", "DOV",
    "SWK", "GWW", "FAST", "CSX", "NSC", "DAL", "UAL", "AAL", "LUV",
    "CARR", "OTIS", "JCI", "A", "AME",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "PXD", "DVN", "FANG", "HES", "HAL", "BKR", "CTRA", "MRO", "APA",
    "KMI", "WMB", "OKE", "ET", "TRGP",
    # Communication / Media
    "DIS", "NFLX", "CMCSA", "T", "VZ", "CHTR", "TMUS",
    "WBD", "PARA", "FOX", "FOXA", "LUMN",
    # Utilities
    "NEE", "DUK", "SO", "D", "SRE", "AEP", "XEL", "WEC", "ES",
    "EXC", "ED", "DTE", "PPL", "FE", "AES", "CMS", "EVRG", "PNW",
    "ATO", "NI",
    # REITs
    "AMT", "PLD", "CCI", "SPG", "O", "WELL", "PSA", "DLR", "EQIX",
    "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "INVH", "SUI",
    "KIM", "REG", "FRT", "BXP",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "STLD",
    "VMC", "MLM", "BLL", "PKG", "IP", "CF", "MOS", "ALB", "EMN",
    "PPG", "CE",
    # Cybersecurity / Growth
    "PLTR", "COIN", "CRWD", "PANW", "FTNT", "ZS", "NET",
    "DDOG", "MDB", "OKTA", "ABNB",
]

# Fallback static list (original top 100)
NYSE_TOP_100_STATIC = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "TSM", "V", "MA", "AVGO", "ORCL", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "ICE", "CME",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK",
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX",
    "TGT", "LOW", "EL", "CL", "GIS",
    "CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "UNP", "FDX", "EMR",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "SNPS", "CDNS",
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    "NEE", "DUK", "SO", "D", "SRE",
    "AMT", "PLD", "CCI", "SPG",
    "PLTR", "UBER", "SQ", "COIN", "CRWD", "PANW",
]


class MarketMonitor:
    """Monitors top 100 NYSE stocks with comprehensive technical and options analysis."""

    def __init__(self, config: dict, reports_dir: str = "reports"):
        self.config = config
        self.monitor_cfg = config.get("market_monitor", {})
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.reports_dir / "monitor_state.json"
        self._discovery_cache_file = self.reports_dir / "discovery_cache.json"
        self._previous_state = self._load_state()
        self._use_dynamic = self.monitor_cfg.get("dynamic_discovery", True)
        self._symbols: List[str] = []  # populated lazily

    # ---- Dynamic Stock Discovery ----

    def discover_top_performers(self) -> List[str]:
        """Dynamically discover top 100 best-performing NYSE stocks.

        Ranks by a composite score of:
          - Daily change (25%)
          - 5-day change (25%)
          - 20-day momentum (25%)
          - Volume surge vs 20-day avg (25%)

        Falls back to the static list on failure.
        """
        logger.info(f"Discovering top performers from {len(NYSE_BROAD_UNIVERSE)} NYSE stocks...")

        # De-duplicate the broad universe
        universe = list(dict.fromkeys(NYSE_BROAD_UNIVERSE))

        scores: List[Tuple[str, float]] = []
        batch_size = 50

        for batch_start in range(0, len(universe), batch_size):
            batch = universe[batch_start:batch_start + batch_size]
            tickers_str = " ".join(batch)
            try:
                data = yf.download(
                    tickers_str, period="1mo", group_by="ticker",
                    threads=True, progress=False,
                )
            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                continue

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        df = data
                    else:
                        df = data[symbol] if symbol in data.columns.get_level_values(0) else None
                    if df is None or df.empty or len(df) < 5:
                        continue

                    close = df["Close"].dropna()
                    volume = df["Volume"].dropna()
                    if len(close) < 5:
                        continue

                    price = float(close.iloc[-1])
                    if price < 5:  # skip penny stocks
                        continue

                    # Daily change
                    daily_chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) >= 2 else 0
                    # 5-day change
                    d5_chg = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100 if len(close) >= 5 else 0
                    # 20-day momentum
                    d20_chg = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100 if len(close) >= 10 else 0
                    # Volume surge
                    avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
                    cur_vol = float(volume.iloc[-1])
                    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

                    # Composite score (higher = better performer)
                    score = (
                        0.25 * daily_chg
                        + 0.25 * d5_chg
                        + 0.25 * d20_chg
                        + 0.25 * min(vol_ratio * 10, 30)  # cap volume boost
                    )
                    scores.append((symbol, score))
                except Exception:
                    continue

            if (batch_start + batch_size) % 100 == 0:
                logger.info(f"  Screened {min(batch_start + batch_size, len(universe))}/{len(universe)} stocks...")

        if len(scores) < 50:
            logger.warning(f"Discovery returned only {len(scores)} stocks, using static fallback")
            return NYSE_TOP_100_STATIC[:100]

        # Sort by composite score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top_100 = [sym for sym, _ in scores[:100]]

        logger.info(f"Top 100 discovered. Best: {top_100[0]} | Worst of top: {top_100[-1]}")

        # Cache discovery results
        try:
            cache = {
                "timestamp": datetime.now().isoformat(),
                "symbols": top_100,
                "scores": {sym: round(sc, 4) for sym, sc in scores[:100]},
            }
            with open(self._discovery_cache_file, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.debug(f"Failed to cache discovery: {e}")

        return top_100

    def _get_symbols(self) -> List[str]:
        """Get the list of symbols to monitor, with caching."""
        if self._symbols:
            return self._symbols

        if self._use_dynamic:
            # Check if we have a recent cache (< 4 hours old)
            try:
                if self._discovery_cache_file.exists():
                    with open(self._discovery_cache_file) as f:
                        cache = json.load(f)
                    cached_time = datetime.fromisoformat(cache["timestamp"])
                    if (datetime.now() - cached_time).total_seconds() < 14400:  # 4 hours
                        self._symbols = cache["symbols"][:100]
                        logger.info(f"Using cached discovery ({len(self._symbols)} symbols, "
                                    f"cached {cached_time.strftime('%H:%M')})")
                        return self._symbols
            except Exception:
                pass
            self._symbols = self.discover_top_performers()
        else:
            self._symbols = NYSE_TOP_100_STATIC[:100]

        return self._symbols

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

    # ---- Options Chain Analysis ----

    def analyze_options_chain(self, symbol: str, price: float) -> Optional[Dict]:
        """Fetch real options chain Greeks from yfinance.

        Looks for the nearest monthly expiration in the 25-45 DTE range
        and returns ATM call/put Greeks, volume data, and put/call ratios.
        """
        try:
            tk = yf.Ticker(symbol)
            expirations = tk.options
            if not expirations:
                return None

            dte_min = self.monitor_cfg.get("options_dte_min", 25)
            dte_max = self.monitor_cfg.get("options_dte_max", 45)
            today = date.today()

            # Find best expiration in DTE range
            best_exp = None
            best_dte = None
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte_min <= dte <= dte_max:
                    if best_dte is None or dte < best_dte:
                        best_exp = exp_str
                        best_dte = dte

            # Fallback: pick nearest expiration > 14 DTE
            if best_exp is None:
                for exp_str in expirations:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - today).days
                    if dte >= 14:
                        best_exp = exp_str
                        best_dte = dte
                        break

            if best_exp is None:
                return None

            chain = tk.option_chain(best_exp)
            calls = chain.calls
            puts = chain.puts

            if calls.empty and puts.empty:
                return None

            # Find ATM strikes (closest to current price)
            atm_call = None
            atm_put = None

            if not calls.empty:
                calls_sorted = calls.copy()
                calls_sorted["dist"] = abs(calls_sorted["strike"] - price)
                atm_call = calls_sorted.loc[calls_sorted["dist"].idxmin()]

            if not puts.empty:
                puts_sorted = puts.copy()
                puts_sorted["dist"] = abs(puts_sorted["strike"] - price)
                atm_put = puts_sorted.loc[puts_sorted["dist"].idxmin()]

            # Aggregate volume
            total_call_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
            total_put_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
            total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

            pc_volume_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
            pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            # Build result
            result = {
                "expiration": best_exp,
                "dte": best_dte,
                "total_call_volume": total_call_vol,
                "total_put_volume": total_put_vol,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "put_call_volume_ratio": round(pc_volume_ratio, 2),
                "put_call_oi_ratio": round(pc_oi_ratio, 2),
            }

            # ATM call Greeks
            if atm_call is not None:
                result["atm_call"] = {
                    "strike": float(atm_call.get("strike", 0)),
                    "last_price": float(atm_call.get("lastPrice", 0)),
                    "bid": float(atm_call.get("bid", 0)),
                    "ask": float(atm_call.get("ask", 0)),
                    "volume": int(atm_call.get("volume", 0)) if pd.notna(atm_call.get("volume")) else 0,
                    "open_interest": int(atm_call.get("openInterest", 0)) if pd.notna(atm_call.get("openInterest")) else 0,
                    "implied_vol": round(float(atm_call.get("impliedVolatility", 0)) * 100, 2),
                    "delta": round(float(atm_call.get("delta", 0.5)), 4) if pd.notna(atm_call.get("delta")) else None,
                    "gamma": round(float(atm_call.get("gamma", 0)), 6) if pd.notna(atm_call.get("gamma")) else None,
                    "theta": round(float(atm_call.get("theta", 0)), 4) if pd.notna(atm_call.get("theta")) else None,
                    "vega": round(float(atm_call.get("vega", 0)), 4) if pd.notna(atm_call.get("vega")) else None,
                }

            # ATM put Greeks
            if atm_put is not None:
                result["atm_put"] = {
                    "strike": float(atm_put.get("strike", 0)),
                    "last_price": float(atm_put.get("lastPrice", 0)),
                    "bid": float(atm_put.get("bid", 0)),
                    "ask": float(atm_put.get("ask", 0)),
                    "volume": int(atm_put.get("volume", 0)) if pd.notna(atm_put.get("volume")) else 0,
                    "open_interest": int(atm_put.get("openInterest", 0)) if pd.notna(atm_put.get("openInterest")) else 0,
                    "implied_vol": round(float(atm_put.get("impliedVolatility", 0)) * 100, 2),
                    "delta": round(float(atm_put.get("delta", -0.5)), 4) if pd.notna(atm_put.get("delta")) else None,
                    "gamma": round(float(atm_put.get("gamma", 0)), 6) if pd.notna(atm_put.get("gamma")) else None,
                    "theta": round(float(atm_put.get("theta", 0)), 4) if pd.notna(atm_put.get("theta")) else None,
                    "vega": round(float(atm_put.get("vega", 0)), 4) if pd.notna(atm_put.get("vega")) else None,
                }

            # Find most active option (highest volume)
            all_opts = pd.concat([calls, puts], ignore_index=True)
            if not all_opts.empty and "volume" in all_opts.columns:
                all_opts_valid = all_opts.dropna(subset=["volume"])
                if not all_opts_valid.empty:
                    most_active = all_opts_valid.loc[all_opts_valid["volume"].idxmax()]
                    result["most_active"] = {
                        "type": "CALL" if most_active.get("strike", 0) in calls["strike"].values else "PUT",
                        "strike": float(most_active.get("strike", 0)),
                        "volume": int(most_active.get("volume", 0)),
                        "open_interest": int(most_active.get("openInterest", 0)) if pd.notna(most_active.get("openInterest")) else 0,
                    }

            return result

        except Exception as e:
            logger.debug(f"Options chain failed for {symbol}: {e}")
            return None

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

            # Estimated Greeks (from historical volatility)
            returns = np.log(close / close.shift(1)).dropna()
            hv_20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.2
            greeks = {
                "est_delta_call": 0.50,
                "est_delta_put": -0.50,
                "est_gamma": round(1 / (price * hv_20 * np.sqrt(30 / 365)), 6) if hv_20 > 0 else 0,
                "est_theta_daily": round(-price * hv_20 / (2 * np.sqrt(365)), 2),
                "est_vega": round(price * np.sqrt(30 / 365) * 0.01, 2),
                "hv_20": round(hv_20 * 100, 2),
            }

            # Real options chain data (if enabled)
            options_data = None
            if self.monitor_cfg.get("options_analysis", True):
                options_data = self.analyze_options_chain(symbol, price)

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
            if ma_50 and ma_200:
                if ma_50 > ma_200:
                    signals.append("MA50_ABOVE_MA200")
                else:
                    signals.append("MA50_BELOW_MA200")
            if iv_rank > 70:
                signals.append("HIGH_IV_RANK")
            elif iv_rank < 20:
                signals.append("LOW_IV_RANK")

            # Volume surge detection
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            threshold = self.monitor_cfg.get("change_thresholds", {}).get("volume_surge_ratio", 2.0)
            if vol_ratio > threshold:
                signals.append("VOLUME_SURGE")

            # MA proximity (within 1% of a key MA)
            if ma_200 and abs(price - ma_200) / ma_200 < 0.01:
                signals.append("NEAR_MA200")
            if ma_50 and abs(price - ma_50) / ma_50 < 0.01:
                signals.append("NEAR_MA50")

            # Options-specific signals
            if options_data:
                pcr = options_data.get("put_call_volume_ratio", 0)
                if pcr > 1.5:
                    signals.append("HIGH_PUT_CALL_RATIO")
                elif pcr < 0.5 and pcr > 0:
                    signals.append("LOW_PUT_CALL_RATIO")

            result = {
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
                "volume_ratio": round(vol_ratio, 2),
                "analyzed_at": datetime.now().isoformat(),
            }

            if options_data:
                result["options"] = options_data

            return result

        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return None

    def run_full_scan(self) -> List[Dict]:
        """Analyze all 100 symbols and return sorted results."""
        symbols = self._get_symbols()
        logger.info(f"Starting full market scan of {len(symbols)} NYSE stocks...")
        results = []

        for i, symbol in enumerate(symbols):
            try:
                result = self.analyze_symbol(symbol)
                if result:
                    results.append(result)
                if (i + 1) % 20 == 0:
                    logger.info(f"  Scanned {i + 1}/{len(symbols)} symbols...")
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

            # Detect MA golden/death cross (50/200)
            old_above = "MA50_ABOVE_MA200" in old_signals
            new_above = "MA50_ABOVE_MA200" in new_signals
            if not old_above and new_above:
                changes.append({
                    "symbol": sym,
                    "type": "GOLDEN_CROSS_50_200",
                    "detail": f"MA50 crossed ABOVE MA200 — bullish",
                    "price": stock["price"],
                })
            elif old_above and not new_above and "MA50_BELOW_MA200" in new_signals:
                changes.append({
                    "symbol": sym,
                    "type": "DEATH_CROSS_50_200",
                    "detail": f"MA50 crossed BELOW MA200 — bearish",
                    "price": stock["price"],
                })

            # Detect volume surge appearance
            if "VOLUME_SURGE" in new_signals and "VOLUME_SURGE" not in old_signals:
                changes.append({
                    "symbol": sym,
                    "type": "UNUSUAL_VOLUME",
                    "detail": f"Volume {stock['volume_ratio']:.1f}x average",
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
            f"Discovery Mode: {'Dynamic' if self._use_dynamic else 'Static'}",
            "",
        ]

        # Market breadth
        bullish = len([s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")])
        bearish = len([s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")])
        neutral = len(results) - bullish - bearish
        lines.append(f"--- MARKET BREADTH ---")
        lines.append(f"  Bullish: {bullish} ({bullish/len(results)*100:.0f}%)  |  "
                      f"Neutral: {neutral}  |  Bearish: {bearish} ({bearish/len(results)*100:.0f}%)")
        avg_rsi = np.mean([s["rsi"] for s in results]) if results else 50
        lines.append(f"  Average RSI: {avg_rsi:.1f}")
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

        # Volume surges
        surges = [s for s in results if "VOLUME_SURGE" in s["signals"]]
        if surges:
            surges.sort(key=lambda x: x["volume_ratio"], reverse=True)
            lines.append(f"--- UNUSUAL VOLUME — {len(surges)} STOCKS ---")
            for s in surges[:10]:
                lines.append(
                    f"  {s['symbol']:6s} Vol:{s['current_volume']:>12,} "
                    f"({s['volume_ratio']:.1f}x avg) ${s['price']:.2f}"
                )
            lines.append("")

        # Options flow summary (top 10 by options volume)
        opts_stocks = [s for s in results if "options" in s and s["options"]]
        if opts_stocks:
            opts_stocks.sort(
                key=lambda x: x["options"].get("total_call_volume", 0) + x["options"].get("total_put_volume", 0),
                reverse=True,
            )
            lines.append(f"--- OPTIONS FLOW (Top 10 by Volume) ---")
            for s in opts_stocks[:10]:
                o = s["options"]
                lines.append(
                    f"  {s['symbol']:6s} Calls:{o.get('total_call_volume',0):>8,} "
                    f"Puts:{o.get('total_put_volume',0):>8,} "
                    f"P/C:{o.get('put_call_volume_ratio',0):.2f} "
                    f"Exp:{o.get('expiration','?')} "
                    f"DTE:{o.get('dte','?')}"
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
            opts_str = ""
            if "options" in s and s["options"]:
                atm_c = s["options"].get("atm_call", {})
                if atm_c.get("implied_vol"):
                    opts_str = f" | ATM IV:{atm_c['implied_vol']:.1f}%"
                    if atm_c.get("delta") is not None:
                        opts_str += f" D:{atm_c['delta']:.3f}"
                    if atm_c.get("gamma") is not None:
                        opts_str += f" G:{atm_c['gamma']:.5f}"
                    if atm_c.get("theta") is not None:
                        opts_str += f" T:{atm_c['theta']:.3f}"
                    if atm_c.get("vega") is not None:
                        opts_str += f" V:{atm_c['vega']:.3f}"
            lines.append(
                f"  {s['symbol']:6s} IV:{s['iv_rank']:5.1f} "
                f"HV20:{g['hv_20']:5.1f}% "
                f"Gamma:{g['est_gamma']:.6f} "
                f"Theta:{g['est_theta_daily']:>7.2f} "
                f"Vega:{g['est_vega']:>6.2f}"
                f"{opts_str}"
            )
        lines.append("")

        lines.append("=== END MARKET MONITOR ===")
        return "\n".join(lines)

    def generate_morning_briefing(self) -> str:
        """Generate a comprehensive pre-market morning briefing.

        This is the flagship daily summary — always sent at 07:00 ET.
        Includes a full market overview, sector analysis, key levels,
        options opportunities, and actionable signals.
        """
        results = self.run_full_scan()
        changes = self.detect_changes(results)

        now = datetime.now().strftime("%A, %B %d, %Y — %H:%M ET")
        lines = [
            "=" * 60,
            f"  MORNING MARKET BRIEFING",
            f"  {now}",
            "=" * 60,
            "",
        ]

        if not results:
            lines.append("  No data available. Market data may not be loaded yet.")
            self._save_state(results)
            return "\n".join(lines)

        # ---- 1. MARKET OVERVIEW ----
        lines.append(">>> MARKET OVERVIEW")
        total = len(results)
        gainers = [s for s in results if s["daily_change_pct"] > 0]
        losers = [s for s in results if s["daily_change_pct"] < 0]
        unchanged = total - len(gainers) - len(losers)
        bullish = len([s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")])
        bearish = len([s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")])

        avg_change = np.mean([s["daily_change_pct"] for s in results])
        avg_rsi = np.mean([s["rsi"] for s in results])
        avg_iv = np.mean([s["iv_rank"] for s in results])

        lines.append(f"  Stocks Tracked: {total}")
        lines.append(f"  Gainers: {len(gainers)} | Losers: {len(losers)} | Flat: {unchanged}")
        lines.append(f"  Avg Daily Change: {avg_change:+.2f}%")
        lines.append(f"  Trend: {bullish} bullish, {bearish} bearish")
        lines.append(f"  Avg RSI: {avg_rsi:.1f} | Avg IV Rank: {avg_iv:.1f}")
        lines.append("")

        # ---- 2. TOP & BOTTOM MOVERS ----
        lines.append(">>> TOP 10 GAINERS")
        for s in results[:10]:
            lines.append(
                f"  {s['symbol']:6s} ${s['price']:>8.2f} ({s['daily_change_pct']:+.2f}%) "
                f"RSI:{s['rsi']:.0f} MACD:{s['macd']['histogram']:+.4f} "
                f"Trend:{s['trend']}"
            )
        lines.append("")

        lines.append(">>> TOP 10 DECLINERS")
        for s in results[-10:]:
            lines.append(
                f"  {s['symbol']:6s} ${s['price']:>8.2f} ({s['daily_change_pct']:+.2f}%) "
                f"RSI:{s['rsi']:.0f} Trend:{s['trend']}"
            )
        lines.append("")

        # ---- 3. TECHNICAL SIGNALS ----
        lines.append(">>> TECHNICAL SIGNALS")
        overbought = [s for s in results if s["rsi"] > 70]
        oversold = [s for s in results if s["rsi"] < 30]
        macd_bull = [s for s in results if "MACD_BULLISH" in s["signals"]]
        macd_bear = [s for s in results if "MACD_BEARISH" in s["signals"]]
        stoch_ob = [s for s in results if "STOCH_OVERBOUGHT" in s["signals"]]
        stoch_os = [s for s in results if "STOCH_OVERSOLD" in s["signals"]]

        lines.append(f"  RSI Overbought (>70): {len(overbought)} stocks")
        if overbought:
            lines.append(f"    {', '.join(s['symbol'] for s in overbought[:15])}")
        lines.append(f"  RSI Oversold (<30):   {len(oversold)} stocks")
        if oversold:
            lines.append(f"    {', '.join(s['symbol'] for s in oversold[:15])}")
        lines.append(f"  MACD Bullish:         {len(macd_bull)} stocks")
        lines.append(f"  MACD Bearish:         {len(macd_bear)} stocks")
        lines.append(f"  Stoch Overbought:     {len(stoch_ob)} stocks")
        lines.append(f"  Stoch Oversold:       {len(stoch_os)} stocks")
        lines.append("")

        # ---- 4. KEY LEVELS (stocks near MA support/resistance) ----
        near_ma = [s for s in results if "NEAR_MA200" in s["signals"] or "NEAR_MA50" in s["signals"]]
        if near_ma:
            lines.append(">>> KEY LEVELS (Stocks Near Moving Average Support/Resistance)")
            for s in near_ma[:15]:
                ma_levels = []
                if "NEAR_MA200" in s["signals"] and s["ma_200"]:
                    ma_levels.append(f"MA200=${s['ma_200']:.2f}")
                if "NEAR_MA50" in s["signals"] and s["ma_50"]:
                    ma_levels.append(f"MA50=${s['ma_50']:.2f}")
                lines.append(
                    f"  {s['symbol']:6s} ${s['price']:.2f} near {', '.join(ma_levels)} "
                    f"({s['trend']})"
                )
            lines.append("")

        # ---- 5. OPTIONS OPPORTUNITIES ----
        high_iv = sorted([s for s in results if s["iv_rank"] > 60],
                         key=lambda x: x["iv_rank"], reverse=True)
        if high_iv:
            lines.append(f">>> OPTIONS OPPORTUNITIES — HIGH IV RANK ({len(high_iv)} stocks)")
            lines.append("  (Elevated IV = premium selling opportunities)")
            for s in high_iv[:15]:
                opts_info = ""
                if "options" in s and s["options"]:
                    o = s["options"]
                    atm_c = o.get("atm_call", {})
                    opts_info = (
                        f" | ATM Call ${atm_c.get('last_price', 0):.2f}"
                        f" IV:{atm_c.get('implied_vol', 0):.0f}%"
                    )
                lines.append(
                    f"  {s['symbol']:6s} IV Rank:{s['iv_rank']:5.1f} "
                    f"HV20:{s['greeks']['hv_20']:.1f}% "
                    f"RSI:{s['rsi']:.0f} {s['trend']}"
                    f"{opts_info}"
                )
            lines.append("")

        # ---- 6. OPTIONS FLOW ----
        opts_stocks = [s for s in results if "options" in s and s["options"]]
        if opts_stocks:
            opts_stocks.sort(
                key=lambda x: x["options"].get("total_call_volume", 0) + x["options"].get("total_put_volume", 0),
                reverse=True,
            )
            lines.append(">>> OPTIONS FLOW — MOST ACTIVE")
            for s in opts_stocks[:10]:
                o = s["options"]
                total_vol = o.get("total_call_volume", 0) + o.get("total_put_volume", 0)
                lines.append(
                    f"  {s['symbol']:6s} Total Vol:{total_vol:>10,} "
                    f"P/C Ratio:{o.get('put_call_volume_ratio', 0):.2f} "
                    f"Exp:{o.get('expiration', '?')} DTE:{o.get('dte', '?')}"
                )
            lines.append("")

            # Highest P/C ratios (bearish options sentiment)
            high_pcr = [s for s in opts_stocks if s["options"].get("put_call_volume_ratio", 0) > 1.2]
            if high_pcr:
                high_pcr.sort(key=lambda x: x["options"]["put_call_volume_ratio"], reverse=True)
                lines.append(">>> BEARISH OPTIONS SENTIMENT (P/C > 1.2)")
                for s in high_pcr[:8]:
                    lines.append(
                        f"  {s['symbol']:6s} P/C:{s['options']['put_call_volume_ratio']:.2f} "
                        f"${s['price']:.2f} RSI:{s['rsi']:.0f}"
                    )
                lines.append("")

        # ---- 7. VOLUME SURGES ----
        surges = sorted([s for s in results if "VOLUME_SURGE" in s["signals"]],
                        key=lambda x: x["volume_ratio"], reverse=True)
        if surges:
            lines.append(f">>> UNUSUAL VOLUME — {len(surges)} STOCKS")
            for s in surges[:10]:
                lines.append(
                    f"  {s['symbol']:6s} {s['volume_ratio']:.1f}x avg volume "
                    f"${s['price']:.2f} ({s['daily_change_pct']:+.2f}%)"
                )
            lines.append("")

        # ---- 8. CHANGES FROM YESTERDAY ----
        if changes:
            lines.append(f">>> CHANGES SINCE LAST SCAN ({len(changes)})")
            for c in changes[:25]:
                lines.append(f"  [{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("  End of Morning Briefing")
        lines.append("=" * 60)

        # Update state for next comparison
        self._save_state(results)

        return "\n".join(lines)

    def generate_html_report(self, results: List[Dict], changes: List[Dict]) -> str:
        """Generate a full HTML report for email."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

        # Build HTML rows for top performers
        top_rows = ""
        for s in results[:25]:
            change_color = "color:#00e676" if s["daily_change_pct"] >= 0 else "color:#ff5252"
            opts_iv = ""
            if "options" in s and s["options"]:
                atm_c = s["options"].get("atm_call", {})
                if atm_c.get("implied_vol"):
                    opts_iv = f"{atm_c['implied_vol']:.0f}%"
                    if atm_c.get("delta") is not None:
                        opts_iv += f" (D:{atm_c['delta']:.2f})"
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
                <td>{opts_iv or '-'}</td>
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

        # Options flow rows
        opts_rows = ""
        opts_stocks = [s for s in results if "options" in s and s["options"]]
        opts_stocks.sort(
            key=lambda x: x["options"].get("total_call_volume", 0) + x["options"].get("total_put_volume", 0),
            reverse=True,
        )
        for s in opts_stocks[:15]:
            o = s["options"]
            atm_c = o.get("atm_call", {})
            opts_rows += f"""<tr>
                <td>{s['symbol']}</td>
                <td>{o.get('total_call_volume', 0):,}</td>
                <td>{o.get('total_put_volume', 0):,}</td>
                <td>{o.get('put_call_volume_ratio', 0):.2f}</td>
                <td>{atm_c.get('implied_vol', '-')}</td>
                <td>{atm_c.get('delta', '-')}</td>
                <td>{atm_c.get('gamma', '-')}</td>
                <td>{atm_c.get('theta', '-')}</td>
                <td>{atm_c.get('vega', '-')}</td>
                <td>{o.get('expiration', '-')}</td>
            </tr>"""

        # Market breadth stats
        bullish = len([s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")])
        bearish = len([s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")])

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
    <p>Generated: {now} | Stocks Analyzed: {len(results)} | Discovery: {'Dynamic' if self._use_dynamic else 'Static'}</p>

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
        <div class="stat-card">
            <div class="stat-value">{len([s for s in results if 'VOLUME_SURGE' in s['signals']])}</div>
            <div class="stat-label">Volume Surges</div>
        </div>
    </div>

    <h2>Top 25 Performers — Full Technical Analysis</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Price</th><th>Change</th><th>RSI</th><th>IV Rank</th>
            <th>MA20</th><th>MA50</th><th>MA200</th><th>MACD Hist</th>
            <th>Stoch K/D</th><th>HV20</th><th>Theta</th><th>Options IV</th><th>Trend</th><th>Signals</th>
        </tr>
        {top_rows}
    </table>

    {"<h2>Options Flow — Top 15 Most Active</h2><table><tr><th>Symbol</th><th>Call Vol</th><th>Put Vol</th><th>P/C Ratio</th><th>ATM IV</th><th>Delta</th><th>Gamma</th><th>Theta</th><th>Vega</th><th>Expiration</th></tr>" + opts_rows + "</table>" if opts_rows else ""}

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
                    "volume_ratio": r.get("volume_ratio", 0),
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
