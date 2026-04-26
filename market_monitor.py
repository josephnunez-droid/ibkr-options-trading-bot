"""
Market Monitor: Tracks top 100 NYSE stocks with full technical analysis.

Dynamically discovers the top 100 best-performing NYSE-listed stocks,
then analyzes: Greeks (from live options chains), IV Rank, 20/50/200
day MA, RSI, MACD, Stochastics.

Generates summaries, detects significant changes, and identifies the
best options opportunities for notifications.
"""

import logging
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Broad pool of NYSE-listed stocks to screen from (~250 large/mid-cap).
# The monitor dynamically ranks these by performance and picks the top 100.
NYSE_SCREENING_POOL = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "TSM", "V", "MA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "ICE", "CME", "MCO", "SPGI", "MMC", "AON",
    "CB", "MET", "PRU", "AIG", "ALL", "TRV",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "ELV", "CI", "HCA",
    "ZTS", "BSX", "BDX", "REGN", "VRTX", "DXCM", "A", "IQV",
    # Consumer
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX",
    "TGT", "LOW", "EL", "CL", "GIS", "YUM", "CMG", "ORLY", "AZO",
    "ROST", "TJX", "DG", "DLTR", "KR", "SYY", "KDP",
    # Industrial
    "CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "UNP", "FDX", "EMR", "ITW", "PH", "ROK", "ETN", "GD", "NOC",
    "WM", "RSG", "FAST", "CTAS", "CARR", "OTIS", "IR",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "DVN", "HAL", "BKR", "FANG", "HES", "WMB", "KMI", "OKE",
    # Tech / Semis
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "SNPS", "CDNS",
    "KLAC", "MRVL", "ON", "NXPI", "MPWR", "SWKS",
    # Communication / Media
    "DIS", "NFLX", "CMCSA", "T", "VZ", "CHTR", "TMUS",
    # Software / Cloud
    "PLTR", "UBER", "SQ", "COIN", "CRWD", "PANW", "FTNT", "ZS",
    "DDOG", "SNOW", "NET", "ABNB", "DASH", "TTD", "HUBS",
    # REITs
    "AMT", "PLD", "CCI", "SPG", "O", "EQIX", "PSA", "DLR", "WELL",
    # Utilities
    "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL", "ED", "WEC",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "CF", "DOW",
    # Autos / Transport
    "GM", "F", "TM", "DAL", "UAL", "LUV", "CSX", "NSC",
    # Miscellaneous large-cap
    "PYPL", "INTU", "NOW", "ACN", "ADP", "FIS", "FISV", "GPN",
    "MKTX", "MSCI", "CPRT", "VRSK", "RBLX", "DKNG", "SOFI",
    "RIVN", "MARA", "SMCI", "ARM", "GRAB",
]

# Fallback static list if dynamic discovery fails
NYSE_TOP_100_FALLBACK = [
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
    """Monitors top 100 NYSE stocks with comprehensive technical analysis."""

    def __init__(self, config: dict, reports_dir: str = "reports"):
        self.config = config
        self.monitor_cfg = config.get("market_monitor", {})
        self._max_workers = self.monitor_cfg.get("max_workers", 8)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.reports_dir / "monitor_state.json"
        self._rankings_file = self.reports_dir / "top100_rankings.json"
        self._previous_state = self._load_state()
        self._symbols = self._load_or_discover_top_100()

    # ---- Dynamic Top 100 Discovery ----

    def _load_or_discover_top_100(self) -> List[str]:
        """Load cached rankings or discover top 100 performers."""
        try:
            if self._rankings_file.exists():
                with open(self._rankings_file) as f:
                    data = json.load(f)
                cached_date = data.get("date", "")
                today = date.today().isoformat()
                # Use cached rankings if from today
                if cached_date == today and data.get("symbols"):
                    logger.info(f"Using cached top 100 rankings from {cached_date}")
                    return data["symbols"][:100]
        except Exception as e:
            logger.debug(f"No cached rankings: {e}")

        return self.discover_top_100()

    def discover_top_100(self) -> List[str]:
        """
        Dynamically discover the top 100 best-performing NYSE stocks.

        Ranks stocks from the screening pool by a composite score:
        - 5-day return (40% weight) — recent momentum
        - 20-day return (30% weight) — short-term trend
        - Relative volume (30% weight) — institutional interest
        """
        logger.info(f"Discovering top 100 performers from {len(NYSE_SCREENING_POOL)} candidates...")

        scores = []

        def _score_candidate(symbol):
            try:
                tk = yf.Ticker(symbol)
                hist = tk.history(period="1mo")
                if hist.empty or len(hist) < 10:
                    return None

                close = hist["Close"]
                volume = hist["Volume"]
                price = float(close.iloc[-1])

                # Skip penny stocks
                if price < 5.0:
                    return None

                # 5-day return
                ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
                # 20-day return
                ret_20d = (close.iloc[-1] / close.iloc[0] - 1) * 100

                # Relative volume (today vs 20-day avg)
                avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
                curr_vol = float(volume.iloc[-1])
                rel_vol = curr_vol / avg_vol if avg_vol > 0 else 1.0

                return {
                    "symbol": symbol,
                    "price": price,
                    "ret_5d": float(ret_5d),
                    "ret_20d": float(ret_20d),
                    "rel_volume": float(rel_vol),
                    "avg_volume": avg_vol,
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(_score_candidate, sym): sym for sym in NYSE_SCREENING_POOL}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scores.append(result)

        if len(scores) < 50:
            logger.warning(f"Only {len(scores)} candidates scored, using fallback list")
            return NYSE_TOP_100_FALLBACK[:100]

        df = pd.DataFrame(scores)

        # Normalize each factor to 0-100 scale
        for col in ["ret_5d", "ret_20d", "rel_volume"]:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max != col_min:
                df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min) * 100
            else:
                df[f"{col}_norm"] = 50

        # Composite performance score
        df["perf_score"] = (
            df["ret_5d_norm"] * 0.40 +
            df["ret_20d_norm"] * 0.30 +
            df["rel_volume_norm"] * 0.30
        )

        df = df.sort_values("perf_score", ascending=False)
        top_100 = df.head(100)["symbol"].tolist()

        # Cache rankings for today
        try:
            cache_data = {
                "date": date.today().isoformat(),
                "symbols": top_100,
                "full_rankings": df[["symbol", "price", "ret_5d", "ret_20d",
                                      "rel_volume", "perf_score"]].head(100).to_dict("records"),
            }
            with open(self._rankings_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached top 100 rankings for {date.today()}")
        except Exception as e:
            logger.warning(f"Failed to cache rankings: {e}")

        logger.info(f"Top 10 performers: {', '.join(top_100[:10])}")
        return top_100

    # ---- Options Chain Analysis ----

    def analyze_options_chain(self, symbol: str, price: float) -> Dict:
        """
        Fetch real options chain data from yfinance and compute proper
        Black-Scholes Greeks for ATM options, plus put/call ratios and
        unusual activity detection.
        """
        RISK_FREE_RATE = 0.043

        result = {
            "has_options": False,
            "atm_call": {},
            "atm_put": {},
            "put_call_ratio": None,
            "total_call_oi": 0,
            "total_put_oi": 0,
            "total_call_volume": 0,
            "total_put_volume": 0,
            "nearest_expiry": None,
            "dte": None,
            "options_unusual_activity": False,
        }

        try:
            tk = yf.Ticker(symbol)
            expirations = tk.options
            if not expirations:
                return result

            today = date.today()
            target_exp = None
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 25 <= dte <= 50:
                    target_exp = exp_str
                    break

            if not target_exp:
                for exp_str in expirations:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - today).days
                    if dte >= 7:
                        target_exp = exp_str
                        break

            if not target_exp:
                return result

            exp_date = datetime.strptime(target_exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            T = dte / 365.0

            chain = tk.option_chain(target_exp)
            calls = chain.calls
            puts = chain.puts

            result["has_options"] = True
            result["nearest_expiry"] = target_exp
            result["dte"] = dte

            result["total_call_oi"] = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            result["total_put_oi"] = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
            result["total_call_volume"] = int(calls["volume"].fillna(0).sum()) if "volume" in calls.columns else 0
            result["total_put_volume"] = int(puts["volume"].fillna(0).sum()) if "volume" in puts.columns else 0

            if result["total_call_oi"] > 0:
                result["put_call_ratio"] = round(result["total_put_oi"] / result["total_call_oi"], 2)

            total_vol = result["total_call_volume"] + result["total_put_volume"]
            total_oi = result["total_call_oi"] + result["total_put_oi"]
            if total_oi > 0 and total_vol > 2 * total_oi:
                result["options_unusual_activity"] = True

            # ATM call with real Black-Scholes Greeks
            if not calls.empty:
                calls = calls.copy()
                calls["dist"] = abs(calls["strike"] - price)
                atm_call = calls.loc[calls["dist"].idxmin()]
                strike = float(atm_call["strike"])
                iv = float(atm_call.get("impliedVolatility", 0) or 0)
                greeks = self.black_scholes_greeks(price, strike, T, RISK_FREE_RATE, iv, "call") if iv > 0 else {}
                result["atm_call"] = {
                    "strike": strike,
                    "bid": float(atm_call.get("bid", 0)),
                    "ask": float(atm_call.get("ask", 0)),
                    "last": float(atm_call.get("lastPrice", 0)),
                    "volume": int(atm_call.get("volume", 0) or 0),
                    "oi": int(atm_call.get("openInterest", 0) or 0),
                    "iv": iv,
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                }

            # ATM put with real Black-Scholes Greeks
            if not puts.empty:
                puts = puts.copy()
                puts["dist"] = abs(puts["strike"] - price)
                atm_put = puts.loc[puts["dist"].idxmin()]
                strike = float(atm_put["strike"])
                iv = float(atm_put.get("impliedVolatility", 0) or 0)
                greeks = self.black_scholes_greeks(price, strike, T, RISK_FREE_RATE, iv, "put") if iv > 0 else {}
                result["atm_put"] = {
                    "strike": strike,
                    "bid": float(atm_put.get("bid", 0)),
                    "ask": float(atm_put.get("ask", 0)),
                    "last": float(atm_put.get("lastPrice", 0)),
                    "volume": int(atm_put.get("volume", 0) or 0),
                    "oi": int(atm_put.get("openInterest", 0) or 0),
                    "iv": iv,
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                }

        except Exception as e:
            logger.debug(f"Options chain analysis failed for {symbol}: {e}")

        return result

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
    def black_scholes_greeks(S: float, K: float, T: float, r: float,
                             sigma: float, option_type: str = "call") -> Dict[str, float]:
        """Compute Black-Scholes Greeks for a single option.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate (annualized, e.g. 0.05 for 5%)
            sigma: Implied volatility (annualized, e.g. 0.30 for 30%)
            option_type: "call" or "put"
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        gamma = float(norm.pdf(d1) / (S * sigma * sqrt_T))
        vega = float(S * norm.pdf(d1) * sqrt_T / 100)

        if option_type == "call":
            delta = float(norm.cdf(d1))
            theta = float(
                (-S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            )
        else:
            delta = float(norm.cdf(d1) - 1)
            theta = float(
                (-S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            )

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
        }

    @staticmethod
    def calc_bollinger_bands(series: pd.Series, period: int = 20,
                             num_std: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(series) < period:
            price = float(series.iloc[-1])
            return {"upper": price, "middle": price, "lower": price,
                    "bandwidth": 0.0, "pct_b": 0.5}
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        middle = float(sma.iloc[-1])
        upper_val = float(upper.iloc[-1])
        lower_val = float(lower.iloc[-1])
        price = float(series.iloc[-1])
        bandwidth = (upper_val - lower_val) / middle * 100 if middle else 0
        pct_b = (price - lower_val) / (upper_val - lower_val) if (upper_val - lower_val) > 0 else 0.5
        return {
            "upper": round(upper_val, 2),
            "middle": round(middle, 2),
            "lower": round(lower_val, 2),
            "bandwidth": round(bandwidth, 2),
            "pct_b": round(pct_b, 2),
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

            # Bollinger Bands
            bbands = self.calc_bollinger_bands(close)

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

            # Historical volatility
            returns = np.log(close / close.shift(1)).dropna()
            hv_20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.2

            # Base Greeks (estimated, always available)
            greeks = {
                "hv_20": round(hv_20 * 100, 2),
            }

            # Real options chain data with Black-Scholes Greeks
            options_data = self.analyze_options_chain(symbol, price)
            if options_data["has_options"]:
                atm_call = options_data.get("atm_call", {})
                atm_put = options_data.get("atm_put", {})
                if atm_call.get("iv"):
                    greeks["atm_call_iv"] = round(atm_call["iv"] * 100, 2)
                    greeks["atm_call_bid"] = atm_call.get("bid", 0)
                    greeks["atm_call_ask"] = atm_call.get("ask", 0)
                    greeks["call_delta"] = atm_call.get("delta", 0)
                    greeks["call_gamma"] = atm_call.get("gamma", 0)
                    greeks["call_theta"] = atm_call.get("theta", 0)
                    greeks["call_vega"] = atm_call.get("vega", 0)
                if atm_put.get("iv"):
                    greeks["atm_put_iv"] = round(atm_put["iv"] * 100, 2)
                    greeks["atm_put_bid"] = atm_put.get("bid", 0)
                    greeks["atm_put_ask"] = atm_put.get("ask", 0)
                    greeks["put_delta"] = atm_put.get("delta", 0)
                    greeks["put_gamma"] = atm_put.get("gamma", 0)
                    greeks["put_theta"] = atm_put.get("theta", 0)
                    greeks["put_vega"] = atm_put.get("vega", 0)
            else:
                # Fallback estimated Greeks when no options chain available
                greeks["call_delta"] = 0.50
                greeks["put_delta"] = -0.50
                greeks["call_gamma"] = round(1 / (price * hv_20 * np.sqrt(30 / 365)), 6) if hv_20 > 0 else 0
                greeks["call_theta"] = round(-price * hv_20 / (2 * np.sqrt(365)), 4)
                greeks["call_vega"] = round(price * np.sqrt(30 / 365) * 0.01, 4)
                greeks["estimated"] = True
                options_data = {}

            # Earnings proximity
            earnings_days = None
            try:
                cal = tk.calendar
                if cal is not None and not cal.empty:
                    for col in cal.columns:
                        val = cal.iloc[0][col]
                        if hasattr(val, "date"):
                            days_to = (val.date() - date.today()).days
                            if days_to >= 0:
                                earnings_days = days_to
                                break
                        elif hasattr(val, "days"):
                            pass
            except Exception:
                pass

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
            if avg_volume > 0 and current_volume / avg_volume >= 2.0:
                signals.append("VOLUME_SPIKE")
            if bbands["pct_b"] > 1.0:
                signals.append("ABOVE_UPPER_BB")
            elif bbands["pct_b"] < 0.0:
                signals.append("BELOW_LOWER_BB")
            if earnings_days is not None and earnings_days <= 7:
                signals.append(f"EARNINGS_IN_{earnings_days}D")

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
                "bollinger": bbands,
                "iv_rank": round(iv_rank, 2),
                "greeks": greeks,
                "earnings_days": earnings_days,
                "trend": trend,
                "signals": signals,
                "avg_volume": round(avg_volume),
                "current_volume": round(current_volume),
                "volume_ratio": round(current_volume / avg_volume, 2) if avg_volume > 0 else 0,
                "options": options_data if options_data else {},
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return None

    def run_full_scan(self) -> List[Dict]:
        """Analyze all 100 symbols in parallel and return sorted results."""
        logger.info(f"Starting full market scan of {len(self._symbols)} NYSE stocks...")
        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(self.analyze_symbol, sym): sym for sym in self._symbols}
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    results.append(result)
                if completed % 25 == 0:
                    logger.info(f"  Scanned {completed}/{len(self._symbols)} symbols...")

        # Sort by daily performance (best first)
        results.sort(key=lambda x: x["daily_change_pct"], reverse=True)

        logger.info(f"Scan complete: {len(results)} stocks analyzed")
        return results

    def refresh_top_100(self):
        """Force re-discovery of top 100 performers (called once daily)."""
        self._symbols = self.discover_top_100()

    def detect_changes(self, current: List[Dict]) -> List[Dict]:
        """Compare current scan to previous state and flag significant changes.

        Uses thresholds from config.yaml market_monitor.change_thresholds.
        """
        changes = []
        prev_map = {s["symbol"]: s for s in self._previous_state}
        thresholds = self.monitor_cfg.get("change_thresholds", {})
        price_move_pct = thresholds.get("price_move_pct", 2.0)
        iv_high = thresholds.get("iv_rank_high", 65)
        iv_low = thresholds.get("iv_rank_low", 25)
        vol_spike = thresholds.get("volume_spike_ratio", 2.0)

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
                        "priority": "MEDIUM",
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
                    "priority": "HIGH" if any(
                        s in added for s in ("RSI_OVERBOUGHT", "RSI_OVERSOLD",
                                             "MACD_BULLISH", "MACD_BEARISH")
                    ) else "MEDIUM",
                })

            # Detect large price moves (configurable threshold)
            prev_price = prev.get("price", 0)
            if prev_price > 0:
                move_pct = abs(stock["price"] - prev_price) / prev_price * 100
                if move_pct > price_move_pct:
                    direction = "UP" if stock["price"] > prev_price else "DOWN"
                    changes.append({
                        "symbol": sym,
                        "type": "LARGE_MOVE",
                        "detail": f"{direction} {move_pct:.1f}% (${prev_price:.2f} -> ${stock['price']:.2f})",
                        "price": stock["price"],
                        "priority": "HIGH" if move_pct > 5 else "MEDIUM",
                    })

            # Detect IV rank regime change
            prev_iv = prev.get("iv_rank", 50)
            curr_iv = stock["iv_rank"]
            if (prev_iv < iv_high and curr_iv >= iv_high):
                changes.append({
                    "symbol": sym,
                    "type": "IV_SPIKE",
                    "detail": f"IV Rank surged: {prev_iv:.0f} -> {curr_iv:.0f} (sell premium opportunity)",
                    "price": stock["price"],
                    "priority": "HIGH",
                })
            elif (prev_iv >= iv_low and curr_iv < iv_low):
                changes.append({
                    "symbol": sym,
                    "type": "IV_CRUSH",
                    "detail": f"IV Rank dropped: {prev_iv:.0f} -> {curr_iv:.0f}",
                    "price": stock["price"],
                    "priority": "MEDIUM",
                })

            # Detect trend change
            if prev.get("trend") != stock["trend"]:
                changes.append({
                    "symbol": sym,
                    "type": "TREND_CHANGE",
                    "detail": f"Trend: {prev.get('trend', 'N/A')} -> {stock['trend']}",
                    "price": stock["price"],
                    "priority": "HIGH",
                })

            # Detect MACD crossover (new this scan)
            prev_hist = prev.get("macd", {}).get("histogram", 0)
            curr_hist = stock["macd"]["histogram"]
            if prev_hist <= 0 < curr_hist:
                changes.append({
                    "symbol": sym,
                    "type": "MACD_BULLISH_CROSS",
                    "detail": f"MACD crossed bullish (hist: {prev_hist:.4f} -> {curr_hist:.4f})",
                    "price": stock["price"],
                    "priority": "HIGH",
                })
            elif prev_hist >= 0 > curr_hist:
                changes.append({
                    "symbol": sym,
                    "type": "MACD_BEARISH_CROSS",
                    "detail": f"MACD crossed bearish (hist: {prev_hist:.4f} -> {curr_hist:.4f})",
                    "price": stock["price"],
                    "priority": "HIGH",
                })

            # Detect stochastic crossover
            prev_k = prev.get("stochastic", {}).get("k", 50)
            curr_k = stock["stochastic"]["k"]
            if prev_k <= 20 < curr_k:
                changes.append({
                    "symbol": sym,
                    "type": "STOCH_BULLISH_CROSS",
                    "detail": f"Stochastic crossed up from oversold (%K: {prev_k:.0f} -> {curr_k:.0f})",
                    "price": stock["price"],
                    "priority": "MEDIUM",
                })
            elif prev_k >= 80 > curr_k:
                changes.append({
                    "symbol": sym,
                    "type": "STOCH_BEARISH_CROSS",
                    "detail": f"Stochastic crossed down from overbought (%K: {prev_k:.0f} -> {curr_k:.0f})",
                    "price": stock["price"],
                    "priority": "MEDIUM",
                })

            # Detect volume spikes
            if stock.get("volume_ratio", 0) >= vol_spike:
                prev_vol_ratio = prev.get("volume_ratio", 1)
                if prev_vol_ratio < vol_spike:
                    changes.append({
                        "symbol": sym,
                        "type": "VOLUME_SPIKE",
                        "detail": f"Volume {stock['volume_ratio']:.1f}x average ({stock['current_volume']:,.0f} vs avg {stock['avg_volume']:,.0f})",
                        "price": stock["price"],
                        "priority": "MEDIUM",
                    })

        # Sort changes: HIGH priority first
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        changes.sort(key=lambda c: priority_order.get(c.get("priority", "LOW"), 2))

        return changes

    def generate_summary(self, results: List[Dict], changes: List[Dict]) -> str:
        """Generate a formatted text summary for notification."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

        # Quick-glance market pulse
        advancing = len([s for s in results if s["daily_change_pct"] > 0])
        declining = len([s for s in results if s["daily_change_pct"] < 0])
        avg_change = sum(s["daily_change_pct"] for s in results) / len(results) if results else 0
        high_iv_count = len([s for s in results if s["iv_rank"] > 60])
        overbought_count = len([s for s in results if s["rsi"] > 70])
        oversold_count = len([s for s in results if s["rsi"] < 30])
        high_prio = len([c for c in changes if c.get("priority") == "HIGH"])

        lines = [
            f"=== MARKET MONITOR SUMMARY ===",
            f"Scan Time: {now}",
            f"Stocks Analyzed: {len(results)} | Changes: {len(changes)} ({high_prio} high priority)",
            "",
            f"QUICK PULSE: {'RISK-ON' if avg_change > 0.3 else 'RISK-OFF' if avg_change < -0.3 else 'MIXED'}  |  "
            f"Adv:{advancing} Dec:{declining}  |  Avg:{avg_change:+.2f}%  |  "
            f"High IV:{high_iv_count}  |  OB:{overbought_count} OS:{oversold_count}",
            "",
        ]

        # High-priority changes first
        if high_prio:
            lines.append(f"*** HIGH PRIORITY ALERTS ({high_prio}) ***")
            for c in changes:
                if c.get("priority") == "HIGH":
                    lines.append(f"  [{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}")
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

        # Changes / Alerts (already showed HIGH priority above, show the rest)
        medium_changes = [c for c in changes if c.get("priority") != "HIGH"]
        if medium_changes:
            lines.append(f"--- OTHER CHANGES SINCE LAST SCAN ({len(medium_changes)}) ---")
            for c in medium_changes[:20]:
                lines.append(f"  [{c['type']}] {c['symbol']} @ ${c['price']:.2f}: {c['detail']}")
            lines.append("")

        # Full Greeks summary for top 20 (Black-Scholes computed)
        lines.append("--- OPTIONS GREEKS — Black-Scholes (Top 20 by IV) ---")
        by_iv = sorted(results, key=lambda x: x["iv_rank"], reverse=True)
        for s in by_iv[:20]:
            g = s["greeks"]
            iv_str = ""
            if g.get("atm_call_iv"):
                iv_str = f" CallIV:{g['atm_call_iv']:.1f}%"
            if g.get("atm_put_iv"):
                iv_str += f" PutIV:{g['atm_put_iv']:.1f}%"
            delta_str = f"D:{g.get('call_delta', 0):+.3f}/{g.get('put_delta', 0):+.3f}"
            gamma_str = f"G:{g.get('call_gamma', 0):.5f}"
            theta_str = f"T:{g.get('call_theta', 0):.3f}"
            vega_str = f"V:{g.get('call_vega', 0):.3f}"
            est_tag = " [est]" if g.get("estimated") else ""
            lines.append(
                f"  {s['symbol']:6s} IV:{s['iv_rank']:5.1f} "
                f"HV20:{g['hv_20']:5.1f}% "
                f"{delta_str} {gamma_str} {theta_str} {vega_str}{iv_str}{est_tag}"
            )
        lines.append("")

        # Bollinger Bands extremes
        bb_above = [s for s in results if s.get("bollinger", {}).get("pct_b", 0.5) > 1.0]
        bb_below = [s for s in results if s.get("bollinger", {}).get("pct_b", 0.5) < 0.0]
        if bb_above or bb_below:
            lines.append(f"--- BOLLINGER BAND EXTREMES ---")
            if bb_above:
                lines.append(f"  Above Upper Band ({len(bb_above)}):")
                for s in bb_above[:8]:
                    bb = s["bollinger"]
                    lines.append(
                        f"    {s['symbol']:6s} ${s['price']:.2f} "
                        f"BB:{bb['lower']:.2f}/{bb['middle']:.2f}/{bb['upper']:.2f} "
                        f"%B:{bb['pct_b']:.2f} BW:{bb['bandwidth']:.1f}%"
                    )
            if bb_below:
                lines.append(f"  Below Lower Band ({len(bb_below)}):")
                for s in bb_below[:8]:
                    bb = s["bollinger"]
                    lines.append(
                        f"    {s['symbol']:6s} ${s['price']:.2f} "
                        f"BB:{bb['lower']:.2f}/{bb['middle']:.2f}/{bb['upper']:.2f} "
                        f"%B:{bb['pct_b']:.2f} BW:{bb['bandwidth']:.1f}%"
                    )
            lines.append("")

        # Upcoming earnings
        earnings_soon = [s for s in results if s.get("earnings_days") is not None and s["earnings_days"] <= 14]
        if earnings_soon:
            earnings_soon.sort(key=lambda x: x["earnings_days"])
            lines.append(f"--- UPCOMING EARNINGS ({len(earnings_soon)} stocks within 14 days) ---")
            for s in earnings_soon:
                days = s["earnings_days"]
                urgency = "***" if days <= 3 else "**" if days <= 7 else "*"
                lines.append(
                    f"  {urgency} {s['symbol']:6s} in {days}d "
                    f"| IV:{s['iv_rank']:.0f} RSI:{s['rsi']:.0f} "
                    f"${s['price']:.2f} ({s['daily_change_pct']:+.2f}%)"
                )
            lines.append("")

        # Options opportunities section
        opts_with_data = [s for s in results if s.get("options", {}).get("has_options")]
        if opts_with_data:
            # Best premium-selling opportunities (high IV + high OI)
            premium_opps = sorted(opts_with_data, key=lambda x: x["iv_rank"], reverse=True)[:10]
            lines.append(f"--- TOP OPTIONS OPPORTUNITIES (Premium Selling) ---")
            for s in premium_opps:
                opts = s["options"]
                atm_c = opts.get("atm_call", {})
                atm_p = opts.get("atm_put", {})
                pcr = opts.get("put_call_ratio")
                pcr_str = f"P/C:{pcr:.2f}" if pcr else "P/C:N/A"
                unusual = " [UNUSUAL ACTIVITY]" if opts.get("options_unusual_activity") else ""
                lines.append(
                    f"  {s['symbol']:6s} IV:{s['iv_rank']:5.1f} "
                    f"CallBid:${atm_c.get('bid', 0):.2f} PutBid:${atm_p.get('bid', 0):.2f} "
                    f"{pcr_str} OI:{opts.get('total_call_oi', 0) + opts.get('total_put_oi', 0):,}"
                    f"{unusual}"
                )
            lines.append("")

            # Unusual options activity
            unusual = [s for s in opts_with_data if s.get("options", {}).get("options_unusual_activity")]
            if unusual:
                lines.append(f"--- UNUSUAL OPTIONS ACTIVITY ({len(unusual)} STOCKS) ---")
                for s in unusual[:10]:
                    opts = s["options"]
                    lines.append(
                        f"  {s['symbol']:6s} ${s['price']:.2f} "
                        f"CallVol:{opts.get('total_call_volume', 0):,} "
                        f"PutVol:{opts.get('total_put_volume', 0):,} "
                        f"CallOI:{opts.get('total_call_oi', 0):,} "
                        f"PutOI:{opts.get('total_put_oi', 0):,}"
                    )
                lines.append("")

        # Volume spikes
        vol_spikes = [s for s in results if s.get("volume_ratio", 0) >= 2.0]
        if vol_spikes:
            vol_spikes.sort(key=lambda x: x["volume_ratio"], reverse=True)
            lines.append(f"--- VOLUME SPIKES (>2x avg) — {len(vol_spikes)} STOCKS ---")
            for s in vol_spikes[:10]:
                lines.append(
                    f"  {s['symbol']:6s} ${s['price']:.2f} "
                    f"Vol:{s['current_volume']:,.0f} ({s['volume_ratio']:.1f}x avg) "
                    f"RSI:{s['rsi']:.0f} Change:{s['daily_change_pct']:+.2f}%"
                )
            lines.append("")

        # Market breadth summary
        bullish_count = len([s for s in results if s["trend"] == "BULLISH"])
        bearish_count = len([s for s in results if s["trend"] == "BEARISH"])
        unchanged = len(results) - advancing - declining

        lines.append("--- MARKET BREADTH ---")
        lines.append(f"  Advancing: {advancing} | Declining: {declining} | Unchanged: {unchanged}")
        lines.append(f"  Avg Change: {avg_change:+.2f}% | Bullish Trend: {bullish_count} | Bearish Trend: {bearish_count}")
        lines.append("")

        lines.append("=== END MARKET MONITOR ===")
        return "\n".join(lines)

    def generate_morning_briefing(self, results: List[Dict], changes: List[Dict]) -> str:
        """
        Generate a comprehensive morning briefing summary.
        This is the first scan of the day — provides a full market overview.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        weekday = datetime.now().strftime("%A")
        lines = [
            f"{'=' * 60}",
            f"  DAILY MORNING BRIEFING — {weekday}, {now}",
            f"{'=' * 60}",
            "",
        ]

        if not results:
            lines.append("No data available — markets may be closed.")
            return "\n".join(lines)

        # VIX (estimate from market volatility)
        avg_iv = sum(s["iv_rank"] for s in results) / len(results) if results else 50
        lines.append(f"MARKET VOLATILITY: Avg IV Rank {avg_iv:.1f}")
        lines.append("")

        # Market breadth
        advancing = len([s for s in results if s["daily_change_pct"] > 0])
        declining = len([s for s in results if s["daily_change_pct"] < 0])
        avg_change = sum(s["daily_change_pct"] for s in results) / len(results)

        sentiment = "BULLISH" if avg_change > 0.5 else "BEARISH" if avg_change < -0.5 else "NEUTRAL"
        lines.append(f"MARKET SENTIMENT: {sentiment}")
        lines.append(f"  Breadth: {advancing} advancing / {declining} declining")
        lines.append(f"  Average Move: {avg_change:+.2f}%")
        lines.append("")

        # Sector breakdown
        sectors = {
            "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL", "CRM", "AMD", "INTC", "QCOM"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP"],
            "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT"],
            "Consumer": ["PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY"],
            "Industrial": ["CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT"],
        }
        sym_map = {s["symbol"]: s for s in results}

        lines.append("SECTOR PERFORMANCE:")
        for sector, syms in sectors.items():
            sector_stocks = [sym_map[s] for s in syms if s in sym_map]
            if sector_stocks:
                avg_ret = sum(s["daily_change_pct"] for s in sector_stocks) / len(sector_stocks)
                arrow = "^" if avg_ret > 0 else "v" if avg_ret < 0 else "-"
                lines.append(f"  {sector:12s} {arrow} {avg_ret:+.2f}% ({len(sector_stocks)} stocks)")
        lines.append("")

        # Earnings calendar for the week
        earnings_soon = [s for s in results if s.get("earnings_days") is not None and s["earnings_days"] <= 14]
        if earnings_soon:
            earnings_soon.sort(key=lambda x: x["earnings_days"])
            lines.append(f"EARNINGS CALENDAR — {len(earnings_soon)} stocks within 14 days:")
            for s in earnings_soon:
                days = s["earnings_days"]
                urgency = "TODAY" if days == 0 else f"in {days}d"
                warn = " *** AVOID SHORT PREMIUM ***" if days <= 5 else ""
                lines.append(
                    f"  {s['symbol']:6s} {urgency:>8s} "
                    f"| IV:{s['iv_rank']:.0f} RSI:{s['rsi']:.0f} "
                    f"${s['price']:.2f}{warn}"
                )
            lines.append("")

        # Key levels to watch (stocks near MA crossovers)
        near_crossover = []
        for s in results:
            if s["ma_20"] and s["ma_50"]:
                gap = abs(s["ma_20"] - s["ma_50"]) / s["price"] * 100
                if gap < 1.0:
                    cross_type = "Golden" if s["ma_20"] > s["ma_50"] else "Death"
                    near_crossover.append((s, cross_type, gap))

        if near_crossover:
            lines.append(f"KEY LEVELS TO WATCH — Near MA Crossovers ({len(near_crossover)}):")
            for s, ct, gap in sorted(near_crossover, key=lambda x: x[2])[:10]:
                lines.append(
                    f"  {s['symbol']:6s} ${s['price']:.2f} — {ct} cross "
                    f"(MA20:{s['ma_20']:.2f} MA50:{s['ma_50']:.2f}, gap {gap:.2f}%)"
                )
            lines.append("")

        # Extreme RSI readings
        extreme_rsi = [s for s in results if s["rsi"] > 75 or s["rsi"] < 25]
        if extreme_rsi:
            lines.append(f"EXTREME RSI READINGS ({len(extreme_rsi)}):")
            for s in sorted(extreme_rsi, key=lambda x: x["rsi"]):
                label = "OVERSOLD" if s["rsi"] < 30 else "OVERBOUGHT"
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} [{label}] ${s['price']:.2f}")
            lines.append("")

        # Bollinger Band extremes
        bb_above = [s for s in results if s.get("bollinger", {}).get("pct_b", 0.5) > 1.0]
        bb_below = [s for s in results if s.get("bollinger", {}).get("pct_b", 0.5) < 0.0]
        if bb_above or bb_below:
            lines.append("BOLLINGER BAND EXTREMES:")
            if bb_above:
                for s in bb_above[:5]:
                    bb = s["bollinger"]
                    lines.append(
                        f"  {s['symbol']:6s} ABOVE upper band "
                        f"(%B:{bb['pct_b']:.2f}) ${s['price']:.2f} "
                        f"BW:{bb['bandwidth']:.1f}%"
                    )
            if bb_below:
                for s in bb_below[:5]:
                    bb = s["bollinger"]
                    lines.append(
                        f"  {s['symbol']:6s} BELOW lower band "
                        f"(%B:{bb['pct_b']:.2f}) ${s['price']:.2f} "
                        f"BW:{bb['bandwidth']:.1f}%"
                    )
            lines.append("")

        # Today's top options plays with real Greeks
        opts_with_data = [s for s in results if s.get("options", {}).get("has_options")]
        if opts_with_data:
            high_iv_opts = [s for s in opts_with_data if s["iv_rank"] > 60]
            if high_iv_opts:
                lines.append(f"TODAY'S OPTIONS FOCUS — High IV Rank (>60): {len(high_iv_opts)} stocks")
                for s in sorted(high_iv_opts, key=lambda x: x["iv_rank"], reverse=True)[:10]:
                    opts = s["options"]
                    atm_c = opts.get("atm_call", {})
                    atm_p = opts.get("atm_put", {})
                    g = s["greeks"]
                    delta_str = (f"D:{g.get('call_delta', 0):+.2f}/{g.get('put_delta', 0):+.2f}"
                                 if g.get("call_delta") else "")
                    lines.append(
                        f"  {s['symbol']:6s} IV:{s['iv_rank']:.0f} "
                        f"Call ${atm_c.get('bid', 0):.2f}/{atm_c.get('ask', 0):.2f} "
                        f"Put ${atm_p.get('bid', 0):.2f}/{atm_p.get('ask', 0):.2f} "
                        f"{delta_str} Trend:{s['trend']}"
                    )
                lines.append("")

        # Append the full standard summary
        lines.append("")
        lines.append(self.generate_summary(results, changes))

        return "\n".join(lines)

    def generate_html_report(self, results: List[Dict], changes: List[Dict]) -> str:
        """Generate a full HTML report for email."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")

        # Build HTML rows for top performers
        top_rows = ""
        for s in results[:25]:
            change_color = "color:#00e676" if s["daily_change_pct"] >= 0 else "color:#ff5252"
            opts = s.get("options", {})
            pcr = opts.get("put_call_ratio")
            pcr_str = f"{pcr:.2f}" if pcr else "-"
            g = s["greeks"]
            call_iv = g.get("atm_call_iv", "-")
            if isinstance(call_iv, float):
                call_iv = f"{call_iv:.1f}%"
            call_delta = g.get("call_delta", 0)
            call_theta = g.get("call_theta", 0)
            bb = s.get("bollinger", {})
            pct_b = bb.get("pct_b", "-")
            if isinstance(pct_b, float):
                pct_b = f"{pct_b:.2f}"
            earn = s.get("earnings_days")
            earn_str = f"{earn}d" if earn is not None else "-"
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
                <td>{pct_b}</td>
                <td>{g['hv_20']:.1f}%</td>
                <td>{call_iv}</td>
                <td>{call_delta:+.3f}</td>
                <td>{call_theta:.3f}</td>
                <td>{pcr_str}</td>
                <td>{earn_str}</td>
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
            <th>Stoch K/D</th><th>BB %B</th><th>HV20</th><th>ATM IV</th>
            <th>Delta</th><th>Theta</th><th>P/C Ratio</th><th>Earnings</th>
            <th>Trend</th><th>Signals</th>
        </tr>
        {top_rows}
    </table>

    {"<h2>Significant Changes</h2><table><tr><th>Symbol</th><th>Type</th><th>Price</th><th>Detail</th></tr>" + change_rows + "</table>" if changes else ""}

    {self._html_options_section(results)}

    <p style="color:#666; font-size:11px; margin-top:30px;">
        IBKR Options Trading Bot — Market Monitor | Auto-generated report
    </p>
</body>
</html>"""
        return html

    def _html_options_section(self, results: List[Dict]) -> str:
        """Generate HTML section for options opportunities."""
        opts_stocks = [s for s in results if s.get("options", {}).get("has_options")]
        if not opts_stocks:
            return ""

        high_iv = sorted(
            [s for s in opts_stocks if s["iv_rank"] > 50],
            key=lambda x: x["iv_rank"], reverse=True
        )[:15]

        if not high_iv:
            return ""

        rows = ""
        for s in high_iv:
            opts = s["options"]
            atm_c = opts.get("atm_call", {})
            atm_p = opts.get("atm_put", {})
            g = s["greeks"]
            pcr = opts.get("put_call_ratio")
            unusual = '<span style="color:#ffab00">UNUSUAL</span>' if opts.get("options_unusual_activity") else ""
            earn = s.get("earnings_days")
            earn_warn = f'<span style="color:#ff5252">{earn}d</span>' if earn is not None and earn <= 7 else (f"{earn}d" if earn is not None else "-")
            rows += f"""<tr>
                <td>{s['symbol']}</td>
                <td>${s['price']:.2f}</td>
                <td>{s['iv_rank']:.1f}</td>
                <td>{g['hv_20']:.1f}%</td>
                <td>${atm_c.get('bid', 0):.2f}/{atm_c.get('ask', 0):.2f}</td>
                <td>{g.get('call_delta', 0):+.3f}</td>
                <td>{g.get('call_theta', 0):.3f}</td>
                <td>{g.get('call_gamma', 0):.5f}</td>
                <td>{g.get('call_vega', 0):.3f}</td>
                <td>${atm_p.get('bid', 0):.2f}/{atm_p.get('ask', 0):.2f}</td>
                <td>{g.get('put_delta', 0):+.3f}</td>
                <td>{f"{pcr:.2f}" if pcr else "-"}</td>
                <td>{opts.get('total_call_oi', 0) + opts.get('total_put_oi', 0):,}</td>
                <td>{earn_warn}</td>
                <td>{unusual}</td>
            </tr>"""

        return f"""
    <h2>Options Opportunities — Premium Selling Candidates (Black-Scholes Greeks)</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Price</th><th>IV Rank</th><th>HV20</th>
            <th>ATM Call Bid/Ask</th><th>Call Delta</th><th>Call Theta</th>
            <th>Call Gamma</th><th>Call Vega</th>
            <th>ATM Put Bid/Ask</th><th>Put Delta</th>
            <th>P/C Ratio</th><th>Total OI</th><th>Earnings</th><th>Flag</th>
        </tr>
        {rows}
    </table>"""

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
                    "bollinger": r.get("bollinger", {}),
                    "volume_ratio": r.get("volume_ratio", 1.0),
                    "current_volume": r.get("current_volume", 0),
                    "avg_volume": r.get("avg_volume", 0),
                    "earnings_days": r.get("earnings_days"),
                })
            with open(self._state_file, "w") as f:
                json.dump(slim, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    # ---- Main Entry Points ----

    def run_scan_and_report(self, is_morning_briefing: bool = False) -> Tuple[str, str, List[Dict]]:
        """
        Run full scan, detect changes, generate reports.

        Args:
            is_morning_briefing: If True, generates comprehensive morning
                briefing instead of standard summary.

        Returns:
            (text_summary, html_report, changes)
        """
        results = self.run_full_scan()
        self._last_results = results  # Cache for callers that need DataFrame
        changes = self.detect_changes(results)

        if is_morning_briefing:
            text_summary = self.generate_morning_briefing(results, changes)
        else:
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


def run_scan(session_label: str = "Market Scan", include_options: bool = True,
             is_morning_briefing: bool = False) -> tuple:
    """
    Module-level convenience function for monitor_runner.py and other callers.

    Args:
        session_label: Human-readable label for this scan session.
        include_options: Whether to include options greeks estimates.
        is_morning_briefing: If True, generates comprehensive morning briefing.

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
    text_summary, html_report, changes = monitor.run_scan_and_report(
        is_morning_briefing=is_morning_briefing
    )

    # Prepend session label to the text summary
    header = f"[{session_label}] — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}\n\n"
    text_summary = header + text_summary

    # Convert results to DataFrame for callers that need tabular data
    results = monitor._last_results if hasattr(monitor, '_last_results') else []
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
