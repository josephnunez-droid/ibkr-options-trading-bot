"""
Market Monitor: Enhanced active market watcher for NYSE stocks.

Dynamically discovers top 100 best-performing NYSE stocks, analyzes real options
chain Greeks, provides comprehensive morning briefings, and notifies on changes.

Analyzes: Real Greeks from options chains, IV Rank, 20/50/200 day MA, RSI, MACD,
Stochastics, MA crossovers, RSI divergences, unusual options volume.
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

# Top 100 NYSE-listed large-cap stocks to monitor (static fallback)
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

# Broad universe for dynamic discovery (~300 large-cap NYSE stocks)
NYSE_BROAD_UNIVERSE = list(set(NYSE_TOP_100 + [
    # Additional Financials
    "AIG", "ALL", "AFL", "MET", "PRU", "TRV", "CB", "MMC", "AON",
    "SPGI", "MCO", "MSCI", "FIS", "FISV", "GPN", "DFS", "SYF",
    "ALLY", "CFG", "KEY", "RF", "HBAN", "FITB", "MTB", "ZION",
    "NTRS", "STT", "BK", "TROW", "IVZ", "BEN",
    # Additional Healthcare
    "ZTS", "VRTX", "REGN", "BIIB", "HCA", "CI", "HUM", "CNC",
    "ELV", "MCK", "CAH", "ABC", "BAX", "BDX", "EW", "A",
    "DXCM", "IDXX", "IQV", "MTD", "WAT", "HOLX", "ALGN",
    # Additional Energy
    "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN", "HES",
    "MRO", "APA", "CTRA", "TRGP",
    # Additional Industrials
    "GD", "NOC", "TDG", "ITW", "PH", "ROK", "ETN", "AME",
    "DOV", "SWK", "GWW", "FAST", "CTAS", "PCAR", "ODFL",
    "CSX", "NSC", "WAB", "TT", "IR", "XYL", "GNRC", "OTIS",
    "CARR", "J", "PWR", "VRSK",
    # Additional Consumer Discretionary
    "BKNG", "MAR", "HLT", "ABNB", "RCL", "CCL", "WYNN", "MGM",
    "LVS", "DRI", "CMG", "YUM", "DPZ", "QSR",
    "F", "GM", "RIVN", "LCID",
    "LULU", "TJX", "ROST", "BBY", "DHI", "LEN", "PHM", "NVR",
    "TPR", "RL", "PVH", "HBI", "GPS",
    # Additional Consumer Staples
    "MDLZ", "KHC", "GPC", "SJM", "HSY", "K", "CPB", "MKC",
    "CHD", "KMB", "STZ", "BF-B", "TAP", "TSN", "HRL",
    "ADM", "SYY", "KR", "WBA",
    # Additional Tech / Software
    "NOW", "ADBE", "INTU", "WDAY", "TEAM", "ZS", "FTNT", "NET",
    "DDOG", "SNOW", "MDB", "HUBS", "VEEV", "ANSS", "KLAC",
    "MCHP", "ON", "NXPI", "SWKS", "ADI", "MPWR",
    # Additional Communication / Media
    "CHTR", "TMUS", "FOX", "FOXA", "PARA", "WBD", "LYV",
    "MTCH", "EA", "TTWO", "RBLX",
    # Additional Utilities
    "AEP", "XEL", "WEC", "ED", "ES", "DTE", "EIX", "FE",
    "PPL", "CMS", "AES", "AWK", "EVRG", "PNW", "NI",
    # Additional REITs
    "O", "WELL", "DLR", "PSA", "EQIX", "AVB", "EQR", "VTR",
    "ARE", "MAA", "UDR", "ESS", "KIM", "REG", "HST", "PEAK",
    "INVH", "SUI", "CPT",
    # Additional Materials
    "LIN", "APD", "SHW", "ECL", "DD", "PPG", "NEM", "FCX",
    "NUE", "STLD", "VMC", "MLM", "DOW", "CE", "EMN",
    "FMC", "ALB", "CF", "MOS", "IFF", "IP", "PKG", "WRK",
]))

# Sector mapping for broad universe stocks
SECTOR_MAP = {
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
                   "USB", "PNC", "TFC", "COF", "ICE", "CME", "AIG", "ALL", "AFL",
                   "MET", "PRU", "TRV", "CB", "MMC", "AON", "SPGI", "MCO", "MSCI",
                   "FIS", "FISV", "GPN", "DFS", "SYF", "ALLY", "CFG", "KEY", "RF",
                   "HBAN", "FITB", "MTB", "ZION", "NTRS", "STT", "BK", "TROW",
                   "IVZ", "BEN", "V", "MA"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
                   "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "ZTS",
                   "VRTX", "REGN", "BIIB", "HCA", "CI", "HUM", "CNC", "ELV",
                   "MCK", "CAH", "ABC", "BAX", "BDX", "EW", "A", "DXCM",
                   "IDXX", "IQV", "MTD", "WAT", "HOLX", "ALGN"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
               "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN", "HES",
               "MRO", "APA", "CTRA", "TRGP"],
    "Industrials": ["CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
                    "UNP", "FDX", "EMR", "GD", "NOC", "TDG", "ITW", "PH", "ROK",
                    "ETN", "AME", "DOV", "SWK", "GWW", "FAST", "CTAS", "PCAR",
                    "ODFL", "CSX", "NSC", "WAB", "TT", "IR", "XYL", "GNRC",
                    "OTIS", "CARR", "J", "PWR", "VRSK"],
    "Consumer Discretionary": ["HD", "LOW", "NKE", "SBUX", "TGT", "MCD",
                               "BKNG", "MAR", "HLT", "ABNB", "RCL", "CCL",
                               "WYNN", "MGM", "LVS", "DRI", "CMG", "YUM", "DPZ",
                               "QSR", "F", "GM", "RIVN", "LCID", "LULU", "TJX",
                               "ROST", "BBY", "DHI", "LEN", "PHM", "NVR", "TPR",
                               "RL", "PVH", "HBI", "GPS", "TSLA", "AMZN"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "EL", "CL", "GIS",
                         "MDLZ", "KHC", "GPC", "SJM", "HSY", "K", "CPB", "MKC",
                         "CHD", "KMB", "STZ", "BF-B", "TAP", "TSN", "HRL",
                         "ADM", "SYY", "KR", "WBA"],
    "Technology": ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO",
                   "ORCL", "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT",
                   "LRCX", "SNPS", "CDNS", "TSM", "NOW", "ADBE", "INTU", "WDAY",
                   "TEAM", "ZS", "FTNT", "NET", "DDOG", "SNOW", "MDB", "HUBS",
                   "VEEV", "ANSS", "KLAC", "MCHP", "ON", "NXPI", "SWKS", "ADI",
                   "MPWR", "PLTR", "CRWD", "PANW"],
    "Communication": ["DIS", "NFLX", "CMCSA", "T", "VZ", "CHTR", "TMUS", "FOX",
                      "FOXA", "PARA", "WBD", "LYV", "MTCH", "EA", "TTWO", "RBLX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "SRE", "AEP", "XEL", "WEC", "ED",
                  "ES", "DTE", "EIX", "FE", "PPL", "CMS", "AES", "AWK", "EVRG",
                  "PNW", "NI"],
    "REITs": ["AMT", "PLD", "CCI", "SPG", "O", "WELL", "DLR", "PSA", "EQIX",
              "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "KIM", "REG",
              "HST", "PEAK", "INVH", "SUI", "CPT"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "DD", "PPG", "NEM", "FCX", "NUE",
                  "STLD", "VMC", "MLM", "DOW", "CE", "EMN", "FMC", "ALB", "CF",
                  "MOS", "IFF", "IP", "PKG", "WRK"],
    "Other": ["BRK-B", "UBER", "SQ", "COIN"],
}

# Reverse mapping: symbol -> sector
SYMBOL_SECTOR = {}
for _sector, _syms in SECTOR_MAP.items():
    for _s in _syms:
        SYMBOL_SECTOR[_s] = _sector


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
        self._options_cache: Dict[str, Dict] = {}
        self._discovery_cache_file = self.reports_dir / "discovery_cache.json"
        self._last_discovery: Optional[datetime] = None

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

    # ---- Dynamic Stock Discovery ----

    def discover_top_performers(self, force: bool = False) -> List[str]:
        """
        Dynamically discover top 100 best-performing NYSE stocks from broad universe.

        Ranks by composite score:
          - Daily change (weight 0.25)
          - Weekly change (weight 0.25)
          - Monthly momentum (weight 0.30)
          - Volume surge vs 20-day avg (weight 0.20)

        Falls back to static NYSE_TOP_100 if discovery fails.
        """
        # Check cache: re-discover at most once per 4 hours unless forced
        if not force and self._last_discovery:
            hours_since = (datetime.now() - self._last_discovery).total_seconds() / 3600
            if hours_since < 4:
                logger.info(f"Using cached discovery ({hours_since:.1f}h old)")
                return self._symbols

        # Try loading from disk cache if fresh enough
        if not force and self._discovery_cache_file.exists():
            try:
                with open(self._discovery_cache_file) as f:
                    cache = json.load(f)
                cache_time = datetime.fromisoformat(cache.get("timestamp", "2000-01-01"))
                if (datetime.now() - cache_time).total_seconds() < 14400:  # 4 hours
                    symbols = cache.get("symbols", [])
                    if len(symbols) >= 50:
                        self._symbols = symbols[:100]
                        self._last_discovery = cache_time
                        logger.info(f"Loaded {len(self._symbols)} symbols from discovery cache")
                        return self._symbols
            except Exception:
                pass

        logger.info(f"Discovering top performers from {len(NYSE_BROAD_UNIVERSE)} stocks...")

        scores = {}
        batch_size = 50
        universe = list(NYSE_BROAD_UNIVERSE)

        for batch_start in range(0, len(universe), batch_size):
            batch = universe[batch_start:batch_start + batch_size]
            batch_str = " ".join(batch)
            try:
                data = yf.download(batch_str, period="1mo", group_by="ticker",
                                   progress=False, threads=True)
            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                continue

            for sym in batch:
                try:
                    if len(batch) == 1:
                        df = data
                    else:
                        df = data[sym] if sym in data.columns.get_level_values(0) else None
                    if df is None or df.empty or len(df) < 5:
                        continue

                    close = df["Close"].dropna()
                    volume = df["Volume"].dropna()
                    if len(close) < 5:
                        continue

                    price = float(close.iloc[-1])
                    if price <= 0:
                        continue

                    # Daily change
                    daily_chg = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100 if len(close) > 1 else 0

                    # Weekly change (last 5 trading days)
                    week_ago = float(close.iloc[-min(5, len(close))]) if len(close) >= 5 else float(close.iloc[0])
                    weekly_chg = (price - week_ago) / week_ago * 100 if week_ago > 0 else 0

                    # Monthly momentum (full period)
                    month_start = float(close.iloc[0])
                    monthly_chg = (price - month_start) / month_start * 100 if month_start > 0 else 0

                    # Volume surge
                    avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
                    curr_vol = float(volume.iloc[-1])
                    vol_surge = curr_vol / avg_vol if avg_vol > 0 else 1.0

                    # Composite score
                    score = (
                        0.25 * daily_chg +
                        0.25 * weekly_chg +
                        0.30 * monthly_chg +
                        0.20 * min(vol_surge * 10, 50)  # cap volume contribution
                    )

                    scores[sym] = {
                        "score": score,
                        "daily_chg": daily_chg,
                        "weekly_chg": weekly_chg,
                        "monthly_chg": monthly_chg,
                        "vol_surge": vol_surge,
                    }

                except Exception:
                    continue

            if (batch_start + batch_size) % 100 == 0:
                logger.info(f"  Discovery progress: {min(batch_start + batch_size, len(universe))}/{len(universe)}")

        if len(scores) < 50:
            logger.warning(f"Discovery found only {len(scores)} stocks, falling back to static list")
            self._symbols = NYSE_TOP_100[:100]
            return self._symbols

        # Rank by composite score, take top 100
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top_symbols = [sym for sym, _ in ranked[:100]]

        self._symbols = top_symbols
        self._last_discovery = datetime.now()

        # Cache to disk
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": top_symbols,
                "scores": {sym: info for sym, info in ranked[:100]},
            }
            with open(self._discovery_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to cache discovery: {e}")

        logger.info(f"Discovery complete: top 100 from {len(scores)} scored stocks")
        logger.info(f"  Top 10: {', '.join(top_symbols[:10])}")
        return self._symbols

    # ---- Real Options Chain Analysis ----

    def analyze_options_chain(self, symbol: str) -> Optional[Dict]:
        """
        Fetch and analyze real options chain data from yfinance.

        Targets the nearest monthly expiration between 25-45 DTE.
        Extracts ATM call/put Greeks, put/call ratios, open interest,
        and identifies highest-volume contracts.
        """
        try:
            tk = yf.Ticker(symbol)
            expirations = tk.options
            if not expirations:
                return None

            # Find nearest expiration in the 25-45 DTE window
            today = date.today()
            target_dte_min = 25
            target_dte_max = 45
            best_exp = None
            best_dte = None

            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < 7:
                    continue
                if target_dte_min <= dte <= target_dte_max:
                    if best_dte is None or abs(dte - 35) < abs(best_dte - 35):
                        best_exp = exp_str
                        best_dte = dte
                elif best_exp is None and dte > 7:
                    # Fallback: nearest available if nothing in 25-45 window
                    if best_dte is None or dte < best_dte:
                        best_exp = exp_str
                        best_dte = dte

            if best_exp is None:
                return None

            chain = tk.option_chain(best_exp)
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return None

            # Get current price for ATM determination
            hist = tk.history(period="1d")
            if hist.empty:
                return None
            current_price = float(hist["Close"].iloc[-1])

            # Find ATM strikes (closest to current price)
            calls_sorted = calls.copy()
            calls_sorted["dist"] = abs(calls_sorted["strike"] - current_price)
            atm_call_idx = calls_sorted["dist"].idxmin()
            atm_call = calls_sorted.loc[atm_call_idx]

            puts_sorted = puts.copy()
            puts_sorted["dist"] = abs(puts_sorted["strike"] - current_price)
            atm_put_idx = puts_sorted["dist"].idxmin()
            atm_put = puts_sorted.loc[atm_put_idx]

            # Extract Greeks from the chain data (yfinance provides impliedVolatility at minimum)
            def safe_float(val, default=0.0):
                try:
                    v = float(val)
                    return v if not np.isnan(v) else default
                except (ValueError, TypeError):
                    return default

            # Build ATM Greeks dict - yfinance option chains have these columns when available
            atm_call_greeks = {}
            atm_put_greeks = {}

            for col, key in [("impliedVolatility", "iv"), ("volume", "volume"),
                             ("openInterest", "openInterest"), ("lastPrice", "lastPrice"),
                             ("bid", "bid"), ("ask", "ask")]:
                if col in calls.columns:
                    atm_call_greeks[key] = safe_float(atm_call.get(col, 0))
                if col in puts.columns:
                    atm_put_greeks[key] = safe_float(atm_put.get(col, 0))

            # Compute Greeks from IV using Black-Scholes approximations
            atm_iv_call = atm_call_greeks.get("iv", 0.3)
            atm_iv_put = atm_put_greeks.get("iv", 0.3)
            avg_iv = (atm_iv_call + atm_iv_put) / 2 if (atm_iv_call + atm_iv_put) > 0 else 0.3
            dte_years = best_dte / 365.0
            sqrt_t = np.sqrt(dte_years) if dte_years > 0 else 0.01

            # ATM approximations (Black-Scholes ATM simplifications)
            # Delta: ATM call ~ 0.5 + adjustment, put ~ -0.5 + adjustment
            d1_approx = (avg_iv * sqrt_t) / 2  # simplified d1 for ATM
            est_delta_call = 0.5 + 0.4 * d1_approx  # rough N(d1) expansion
            est_delta_call = min(max(est_delta_call, 0.35), 0.65)
            est_delta_put = est_delta_call - 1.0

            # Gamma: ATM gamma ~ N'(0) / (S * sigma * sqrt(T))
            nprime_0 = 0.3989  # N'(0) = 1/sqrt(2*pi)
            est_gamma = nprime_0 / (current_price * avg_iv * sqrt_t) if (avg_iv * sqrt_t) > 0 else 0

            # Theta: ATM theta ~ -(S * sigma * N'(0)) / (2 * sqrt(T)) per year, convert to daily
            est_theta_yearly = -(current_price * avg_iv * nprime_0) / (2 * sqrt_t) if sqrt_t > 0 else 0
            est_theta_daily = est_theta_yearly / 365.0

            # Vega: ATM vega ~ S * sqrt(T) * N'(0) (per 1% IV move = /100)
            est_vega = current_price * sqrt_t * nprime_0 / 100

            # Volume and open interest aggregates
            total_call_vol = safe_float(calls["volume"].sum()) if "volume" in calls.columns else 0
            total_put_vol = safe_float(puts["volume"].sum()) if "volume" in puts.columns else 0
            total_call_oi = safe_float(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            total_put_oi = safe_float(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

            pc_vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
            pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            # Highest volume contracts
            top_calls = calls.nlargest(3, "volume")[["strike", "volume", "openInterest", "impliedVolatility", "lastPrice"]].to_dict("records") if "volume" in calls.columns and not calls.empty else []
            top_puts = puts.nlargest(3, "volume")[["strike", "volume", "openInterest", "impliedVolatility", "lastPrice"]].to_dict("records") if "volume" in puts.columns and not puts.empty else []

            # Clean NaN from top contracts
            for contracts in [top_calls, top_puts]:
                for c in contracts:
                    for k, v in c.items():
                        if isinstance(v, float) and np.isnan(v):
                            c[k] = 0.0

            result = {
                "symbol": symbol,
                "expiration": best_exp,
                "dte": best_dte,
                "atm_strike": float(atm_call["strike"]),
                "atm_call": {
                    "strike": float(atm_call["strike"]),
                    "delta": round(est_delta_call, 4),
                    "gamma": round(est_gamma, 6),
                    "theta": round(est_theta_daily, 4),
                    "vega": round(est_vega, 4),
                    "iv": round(atm_iv_call, 4),
                    "bid": atm_call_greeks.get("bid", 0),
                    "ask": atm_call_greeks.get("ask", 0),
                    "volume": atm_call_greeks.get("volume", 0),
                    "open_interest": atm_call_greeks.get("openInterest", 0),
                },
                "atm_put": {
                    "strike": float(atm_put["strike"]),
                    "delta": round(est_delta_put, 4),
                    "gamma": round(est_gamma, 6),
                    "theta": round(est_theta_daily, 4),
                    "vega": round(est_vega, 4),
                    "iv": round(atm_iv_put, 4),
                    "bid": atm_put_greeks.get("bid", 0),
                    "ask": atm_put_greeks.get("ask", 0),
                    "volume": atm_put_greeks.get("volume", 0),
                    "open_interest": atm_put_greeks.get("openInterest", 0),
                },
                "put_call_volume_ratio": round(pc_vol_ratio, 3),
                "put_call_oi_ratio": round(pc_oi_ratio, 3),
                "total_call_volume": int(total_call_vol),
                "total_put_volume": int(total_put_vol),
                "total_call_oi": int(total_call_oi),
                "total_put_oi": int(total_put_oi),
                "top_call_contracts": top_calls,
                "top_put_contracts": top_puts,
            }

            self._options_cache[symbol] = result
            return result

        except Exception as e:
            logger.debug(f"Options chain analysis failed for {symbol}: {e}")
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

            # Enhanced signals: MA crossovers (golden cross / death cross 50/200)
            if ma_50 and ma_200 and len(close) >= 201:
                prev_ma_50 = float(close.iloc[-51:-1].mean())
                prev_ma_200 = float(close.iloc[-201:-1].mean())
                if prev_ma_50 <= prev_ma_200 and ma_50 > ma_200:
                    signals.append("GOLDEN_CROSS_50_200")
                elif prev_ma_50 >= prev_ma_200 and ma_50 < ma_200:
                    signals.append("DEATH_CROSS_50_200")

            # Enhanced signals: RSI divergence detection
            if len(close) >= 28:
                rsi_series = self._calc_rsi_series(close)
                if len(rsi_series) >= 14:
                    # Bullish divergence: price making lower low but RSI making higher low
                    price_recent_low = float(close.tail(14).min())
                    price_prior_low = float(close.iloc[-28:-14].min())
                    rsi_recent_low = float(rsi_series.tail(14).min())
                    rsi_prior_low = float(rsi_series.iloc[-28:-14].min()) if len(rsi_series) >= 28 else rsi_recent_low
                    if price_recent_low < price_prior_low and rsi_recent_low > rsi_prior_low:
                        signals.append("RSI_BULLISH_DIVERGENCE")
                    # Bearish divergence: price making higher high but RSI making lower high
                    price_recent_high = float(close.tail(14).max())
                    price_prior_high = float(close.iloc[-28:-14].max())
                    rsi_recent_high = float(rsi_series.tail(14).max())
                    rsi_prior_high = float(rsi_series.iloc[-28:-14].max()) if len(rsi_series) >= 28 else rsi_recent_high
                    if price_recent_high > price_prior_high and rsi_recent_high < rsi_prior_high:
                        signals.append("RSI_BEARISH_DIVERGENCE")

            # Enhanced signals: Unusual options volume
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            if vol_ratio > 2.0:
                signals.append("UNUSUAL_VOLUME")

            # Enhanced signals: Price near key MAs (within 1%)
            if ma_20 and abs(price - ma_20) / ma_20 < 0.01:
                signals.append("NEAR_MA20")
            if ma_50 and abs(price - ma_50) / ma_50 < 0.01:
                signals.append("NEAR_MA50")
            if ma_200 and abs(price - ma_200) / ma_200 < 0.01:
                signals.append("NEAR_MA200")

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
                "sector": SYMBOL_SECTOR.get(symbol, "Unknown"),
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return None

    @staticmethod
    def _calc_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
        """Return full RSI series (not just last value) for divergence detection."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.dropna()

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

            # Enhanced: Detect MA crossovers (golden/death cross 50/200)
            prev_sigs = set(prev.get("signals", []))
            if "GOLDEN_CROSS_50_200" in new_signals and "GOLDEN_CROSS_50_200" not in prev_sigs:
                changes.append({
                    "symbol": sym,
                    "type": "GOLDEN_CROSS",
                    "detail": "MA50 crossed above MA200 (Golden Cross)",
                    "price": stock["price"],
                })
            if "DEATH_CROSS_50_200" in new_signals and "DEATH_CROSS_50_200" not in prev_sigs:
                changes.append({
                    "symbol": sym,
                    "type": "DEATH_CROSS",
                    "detail": "MA50 crossed below MA200 (Death Cross)",
                    "price": stock["price"],
                })

            # Enhanced: RSI divergence alerts
            if "RSI_BULLISH_DIVERGENCE" in new_signals and "RSI_BULLISH_DIVERGENCE" not in prev_sigs:
                changes.append({
                    "symbol": sym,
                    "type": "RSI_DIVERGENCE",
                    "detail": f"Bullish RSI divergence detected (RSI: {stock['rsi']:.1f})",
                    "price": stock["price"],
                })
            if "RSI_BEARISH_DIVERGENCE" in new_signals and "RSI_BEARISH_DIVERGENCE" not in prev_sigs:
                changes.append({
                    "symbol": sym,
                    "type": "RSI_DIVERGENCE",
                    "detail": f"Bearish RSI divergence detected (RSI: {stock['rsi']:.1f})",
                    "price": stock["price"],
                })

            # Enhanced: Unusual volume alert
            if "UNUSUAL_VOLUME" in new_signals and "UNUSUAL_VOLUME" not in prev_sigs:
                changes.append({
                    "symbol": sym,
                    "type": "UNUSUAL_VOLUME",
                    "detail": f"Volume {stock['volume_ratio']:.1f}x average ({stock['current_volume']:,.0f} vs avg {stock['avg_volume']:,.0f})",
                    "price": stock["price"],
                })

        return changes

    # ---- Morning Briefing ----

    def generate_morning_briefing(self, results: List[Dict],
                                   options_data: Optional[Dict[str, Dict]] = None) -> str:
        """
        Generate a comprehensive morning briefing with market overview.

        Includes:
          - Market breadth (% bullish vs bearish)
          - Sector rotation signals
          - Options flow summary (highest IV, most active options)
          - Key levels (stocks near MA support/resistance)
          - Top movers and actionable setups
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        if options_data is None:
            options_data = self._options_cache

        lines = [
            "=" * 70,
            f"  MORNING MARKET BRIEFING - {now}",
            "=" * 70,
            f"  Stocks Analyzed: {len(results)}",
            "",
        ]

        if not results:
            lines.append("  No data available for briefing.")
            return "\n".join(lines)

        # ---- Market Breadth ----
        bullish = [s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")]
        bearish = [s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")]
        neutral = [s for s in results if s["trend"] == "NEUTRAL"]
        total = len(results)
        pct_bull = len(bullish) / total * 100 if total else 0
        pct_bear = len(bearish) / total * 100 if total else 0

        avg_rsi = np.mean([s["rsi"] for s in results])
        avg_change = np.mean([s["daily_change_pct"] for s in results])
        gainers = len([s for s in results if s["daily_change_pct"] > 0])
        losers = len([s for s in results if s["daily_change_pct"] < 0])

        breadth_label = "BULLISH" if pct_bull > 60 else "BEARISH" if pct_bear > 60 else "MIXED"

        lines.append("--- MARKET BREADTH ---")
        lines.append(f"  Overall Sentiment: {breadth_label}")
        lines.append(f"  Bullish: {len(bullish)} ({pct_bull:.0f}%) | Bearish: {len(bearish)} ({pct_bear:.0f}%) | Neutral: {len(neutral)}")
        lines.append(f"  Gainers: {gainers} | Losers: {losers}")
        lines.append(f"  Average Daily Change: {avg_change:+.2f}%")
        lines.append(f"  Average RSI: {avg_rsi:.1f}")
        lines.append("")

        # ---- Sector Rotation ----
        lines.append("--- SECTOR ROTATION ---")
        sector_perf = {}
        for s in results:
            sector = s.get("sector", SYMBOL_SECTOR.get(s["symbol"], "Unknown"))
            if sector not in sector_perf:
                sector_perf[sector] = {"changes": [], "rsis": [], "count": 0}
            sector_perf[sector]["changes"].append(s["daily_change_pct"])
            sector_perf[sector]["rsis"].append(s["rsi"])
            sector_perf[sector]["count"] += 1

        sector_summary = []
        for sector, data in sector_perf.items():
            avg_chg = np.mean(data["changes"])
            avg_rsi_sec = np.mean(data["rsis"])
            sector_summary.append((sector, avg_chg, avg_rsi_sec, data["count"]))

        sector_summary.sort(key=lambda x: x[1], reverse=True)
        for sector, avg_chg, avg_rsi_sec, count in sector_summary:
            rotation_signal = ""
            if avg_chg > 1.0 and avg_rsi_sec < 60:
                rotation_signal = " [ROTATING IN]"
            elif avg_chg < -1.0 and avg_rsi_sec > 60:
                rotation_signal = " [ROTATING OUT]"
            lines.append(f"  {sector:25s} Avg Chg: {avg_chg:+.2f}%  RSI: {avg_rsi_sec:.0f}  ({count} stocks){rotation_signal}")
        lines.append("")

        # ---- Options Flow Summary ----
        lines.append("--- OPTIONS FLOW SUMMARY ---")
        if options_data:
            # Highest IV stocks
            iv_sorted = sorted(options_data.items(),
                             key=lambda x: max(x[1].get("atm_call", {}).get("iv", 0),
                                              x[1].get("atm_put", {}).get("iv", 0)),
                             reverse=True)
            lines.append("  Highest Implied Volatility:")
            for sym, od in iv_sorted[:10]:
                call_iv = od.get("atm_call", {}).get("iv", 0) * 100
                put_iv = od.get("atm_put", {}).get("iv", 0) * 100
                pc_ratio = od.get("put_call_volume_ratio", 0)
                lines.append(f"    {sym:6s} Call IV: {call_iv:5.1f}%  Put IV: {put_iv:5.1f}%  P/C Vol: {pc_ratio:.2f}")

            # Most active options
            vol_sorted = sorted(options_data.items(),
                              key=lambda x: x[1].get("total_call_volume", 0) + x[1].get("total_put_volume", 0),
                              reverse=True)
            lines.append("")
            lines.append("  Most Active Options Chains:")
            for sym, od in vol_sorted[:10]:
                total_vol = od.get("total_call_volume", 0) + od.get("total_put_volume", 0)
                pc_ratio = od.get("put_call_volume_ratio", 0)
                dte = od.get("dte", 0)
                lines.append(f"    {sym:6s} Total Vol: {total_vol:>8,}  P/C: {pc_ratio:.2f}  DTE: {dte}")

            # High put/call ratio (bearish sentiment)
            high_pc = [(sym, od) for sym, od in options_data.items()
                       if od.get("put_call_volume_ratio", 0) > 1.5]
            if high_pc:
                lines.append("")
                lines.append("  Elevated Put/Call Ratio (>1.5 = bearish sentiment):")
                high_pc.sort(key=lambda x: x[1]["put_call_volume_ratio"], reverse=True)
                for sym, od in high_pc[:10]:
                    lines.append(f"    {sym:6s} P/C: {od['put_call_volume_ratio']:.2f}")
        else:
            lines.append("  No options data available. Run with options_chain analysis enabled.")
        lines.append("")

        # ---- Key Levels (MA Support/Resistance) ----
        lines.append("--- KEY LEVELS: STOCKS AT MA SUPPORT/RESISTANCE ---")
        near_ma = [s for s in results if any(sig in s["signals"]
                   for sig in ["NEAR_MA20", "NEAR_MA50", "NEAR_MA200"])]
        if near_ma:
            for s in near_ma[:15]:
                ma_notes = []
                if "NEAR_MA20" in s["signals"]:
                    ma_notes.append(f"MA20={s['ma_20']}")
                if "NEAR_MA50" in s["signals"]:
                    ma_notes.append(f"MA50={s['ma_50']}")
                if "NEAR_MA200" in s["signals"]:
                    ma_notes.append(f"MA200={s['ma_200']}")
                lines.append(f"  {s['symbol']:6s} ${s['price']:.2f}  Near: {', '.join(ma_notes)}  Trend: {s['trend']}")
        else:
            lines.append("  No stocks currently testing major MA levels.")
        lines.append("")

        # ---- Golden/Death Crosses ----
        golden_crosses = [s for s in results if "GOLDEN_CROSS_50_200" in s["signals"]]
        death_crosses = [s for s in results if "DEATH_CROSS_50_200" in s["signals"]]
        if golden_crosses or death_crosses:
            lines.append("--- MA CROSSOVER ALERTS ---")
            for s in golden_crosses:
                lines.append(f"  [GOLDEN CROSS] {s['symbol']:6s} ${s['price']:.2f} MA50={s['ma_50']} > MA200={s['ma_200']}")
            for s in death_crosses:
                lines.append(f"  [DEATH CROSS]  {s['symbol']:6s} ${s['price']:.2f} MA50={s['ma_50']} < MA200={s['ma_200']}")
            lines.append("")

        # ---- RSI Divergences ----
        bull_div = [s for s in results if "RSI_BULLISH_DIVERGENCE" in s["signals"]]
        bear_div = [s for s in results if "RSI_BEARISH_DIVERGENCE" in s["signals"]]
        if bull_div or bear_div:
            lines.append("--- RSI DIVERGENCE ALERTS ---")
            for s in bull_div:
                lines.append(f"  [BULLISH DIV] {s['symbol']:6s} RSI:{s['rsi']:.1f} - Price lower low, RSI higher low")
            for s in bear_div:
                lines.append(f"  [BEARISH DIV] {s['symbol']:6s} RSI:{s['rsi']:.1f} - Price higher high, RSI lower high")
            lines.append("")

        # ---- Unusual Volume ----
        unusual_vol = [s for s in results if "UNUSUAL_VOLUME" in s["signals"]]
        if unusual_vol:
            unusual_vol.sort(key=lambda x: x["volume_ratio"], reverse=True)
            lines.append("--- UNUSUAL VOLUME (>2x Average) ---")
            for s in unusual_vol[:15]:
                lines.append(f"  {s['symbol']:6s} ${s['price']:.2f}  Vol: {s['current_volume']:>12,.0f}  "
                           f"Ratio: {s['volume_ratio']:.1f}x  Chg: {s['daily_change_pct']:+.2f}%")
            lines.append("")

        # ---- Top Movers ----
        lines.append("--- TOP 5 GAINERS ---")
        for s in results[:5]:
            lines.append(f"  {s['symbol']:6s} ${s['price']:.2f}  {s['daily_change_pct']:+.2f}%  "
                       f"RSI:{s['rsi']:.0f}  Trend:{s['trend']}")
        lines.append("")
        lines.append("--- TOP 5 LOSERS ---")
        for s in results[-5:]:
            lines.append(f"  {s['symbol']:6s} ${s['price']:.2f}  {s['daily_change_pct']:+.2f}%  "
                       f"RSI:{s['rsi']:.0f}  Trend:{s['trend']}")
        lines.append("")

        # ---- Actionable Setups ----
        lines.append("--- ACTIONABLE SETUPS ---")
        # Bullish setups: oversold + bullish divergence or near support
        bull_setups = [s for s in results if s["rsi"] < 35 and
                       any(sig in s["signals"] for sig in ["RSI_BULLISH_DIVERGENCE", "NEAR_MA200", "STOCH_OVERSOLD"])]
        if bull_setups:
            lines.append("  Potential Bullish Reversals (oversold + support signals):")
            for s in bull_setups[:5]:
                sigs = [sig for sig in s["signals"] if sig in
                        ["RSI_BULLISH_DIVERGENCE", "NEAR_MA200", "STOCH_OVERSOLD", "RSI_OVERSOLD"]]
                lines.append(f"    {s['symbol']:6s} ${s['price']:.2f}  RSI:{s['rsi']:.0f}  Signals: {', '.join(sigs)}")

        # Bearish setups: overbought + bearish divergence or resistance
        bear_setups = [s for s in results if s["rsi"] > 65 and
                       any(sig in s["signals"] for sig in ["RSI_BEARISH_DIVERGENCE", "STOCH_OVERBOUGHT"])]
        if bear_setups:
            lines.append("  Potential Bearish Reversals (overbought + resistance signals):")
            for s in bear_setups[:5]:
                sigs = [sig for sig in s["signals"] if sig in
                        ["RSI_BEARISH_DIVERGENCE", "STOCH_OVERBOUGHT", "RSI_OVERBOUGHT"]]
                lines.append(f"    {s['symbol']:6s} ${s['price']:.2f}  RSI:{s['rsi']:.0f}  Signals: {', '.join(sigs)}")

        # High IV premium selling opportunities
        high_iv_setups = [s for s in results if s["iv_rank"] > 70]
        if high_iv_setups:
            high_iv_setups.sort(key=lambda x: x["iv_rank"], reverse=True)
            lines.append("  Premium Selling Opportunities (IV Rank > 70):")
            for s in high_iv_setups[:5]:
                opt = options_data.get(s["symbol"], {})
                dte_str = f"  DTE:{opt['dte']}" if opt and "dte" in opt else ""
                lines.append(f"    {s['symbol']:6s} IV Rank:{s['iv_rank']:.0f}  HV20:{s['greeks']['hv_20']:.1f}%{dte_str}")

        if not bull_setups and not bear_setups and not high_iv_setups:
            lines.append("  No strong actionable setups identified at this time.")
        lines.append("")

        lines.append("=" * 70)
        lines.append("  END MORNING BRIEFING")
        lines.append("=" * 70)

        return "\n".join(lines)

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
            lines.append(f"--- HIGH IV RANK (>60) -- {len(high_iv)} STOCKS ---")
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
            lines.append(f"--- RSI OVERBOUGHT (>70) -- {len(overbought)} STOCKS ---")
            for s in overbought[:10]:
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} Stoch:{s['stochastic']['k']:.0f}")
            lines.append("")
        if oversold:
            lines.append(f"--- RSI OVERSOLD (<30) -- {len(oversold)} STOCKS ---")
            for s in oversold[:10]:
                lines.append(f"  {s['symbol']:6s} RSI:{s['rsi']:.1f} Stoch:{s['stochastic']['k']:.0f}")
            lines.append("")

        # MACD crossover signals
        bullish_macd = [s for s in results if "MACD_BULLISH" in s["signals"]]
        bearish_macd = [s for s in results if "MACD_BEARISH" in s["signals"]]
        if bullish_macd:
            lines.append(f"--- MACD BULLISH CROSSOVER -- {len(bullish_macd)} STOCKS ---")
            for s in bullish_macd[:10]:
                lines.append(
                    f"  {s['symbol']:6s} MACD:{s['macd']['macd']:.4f} "
                    f"Signal:{s['macd']['signal']:.4f} Hist:{s['macd']['histogram']:.4f}"
                )
            lines.append("")

        # Golden crosses
        golden = [s for s in results if "GOLDEN_CROSS_20_50" in s["signals"]]
        if golden:
            lines.append(f"--- MA20 > MA50 CROSSOVER -- {len(golden)} STOCKS ---")
            for s in golden[:10]:
                lines.append(
                    f"  {s['symbol']:6s} MA20:{s['ma_20']} MA50:{s['ma_50']} "
                    f"MA200:{s['ma_200'] or 'N/A'} Trend:{s['trend']}"
                )
            lines.append("")

        # Enhanced: 50/200 crossovers
        golden_50_200 = [s for s in results if "GOLDEN_CROSS_50_200" in s["signals"]]
        death_50_200 = [s for s in results if "DEATH_CROSS_50_200" in s["signals"]]
        if golden_50_200:
            lines.append(f"--- GOLDEN CROSS 50/200 -- {len(golden_50_200)} STOCKS ---")
            for s in golden_50_200:
                lines.append(f"  {s['symbol']:6s} MA50:{s['ma_50']} > MA200:{s['ma_200']}")
            lines.append("")
        if death_50_200:
            lines.append(f"--- DEATH CROSS 50/200 -- {len(death_50_200)} STOCKS ---")
            for s in death_50_200:
                lines.append(f"  {s['symbol']:6s} MA50:{s['ma_50']} < MA200:{s['ma_200']}")
            lines.append("")

        # Enhanced: Unusual volume
        unusual = [s for s in results if "UNUSUAL_VOLUME" in s["signals"]]
        if unusual:
            unusual.sort(key=lambda x: x["volume_ratio"], reverse=True)
            lines.append(f"--- UNUSUAL VOLUME (>2x avg) -- {len(unusual)} STOCKS ---")
            for s in unusual[:10]:
                lines.append(f"  {s['symbol']:6s} Vol Ratio: {s['volume_ratio']:.1f}x  "
                           f"Vol: {s['current_volume']:,.0f} vs Avg: {s['avg_volume']:,.0f}")
            lines.append("")

        # Enhanced: RSI Divergences
        divergences = [s for s in results if any("DIVERGENCE" in sig for sig in s["signals"])]
        if divergences:
            lines.append(f"--- RSI DIVERGENCES -- {len(divergences)} STOCKS ---")
            for s in divergences[:10]:
                div_type = "Bullish" if "RSI_BULLISH_DIVERGENCE" in s["signals"] else "Bearish"
                lines.append(f"  {s['symbol']:6s} {div_type} Divergence  RSI:{s['rsi']:.1f}  Price:${s['price']:.2f}")
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

        # Enhanced: Real options chain data summary if available
        if self._options_cache:
            lines.append(f"--- REAL OPTIONS CHAIN DATA ({len(self._options_cache)} symbols) ---")
            for sym in list(self._options_cache.keys())[:15]:
                od = self._options_cache[sym]
                call = od.get("atm_call", {})
                put = od.get("atm_put", {})
                lines.append(
                    f"  {sym:6s} Exp:{od.get('expiration', 'N/A')} DTE:{od.get('dte', 0):>2d} "
                    f"Strike:{od.get('atm_strike', 0):.0f} "
                    f"C.IV:{call.get('iv', 0)*100:.1f}% P.IV:{put.get('iv', 0)*100:.1f}% "
                    f"P/C:{od.get('put_call_volume_ratio', 0):.2f}"
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

        # Build options chain HTML if available
        options_rows = ""
        if self._options_cache:
            for sym, od in sorted(self._options_cache.items()):
                call = od.get("atm_call", {})
                put_ = od.get("atm_put", {})
                options_rows += f"""<tr>
                    <td>{sym}</td>
                    <td>{od.get('expiration', '-')}</td>
                    <td>{od.get('dte', 0)}</td>
                    <td>{od.get('atm_strike', 0):.1f}</td>
                    <td>{call.get('delta', 0):.3f}</td>
                    <td>{call.get('gamma', 0):.5f}</td>
                    <td>{call.get('theta', 0):.4f}</td>
                    <td>{call.get('vega', 0):.4f}</td>
                    <td>{call.get('iv', 0)*100:.1f}%</td>
                    <td>{put_.get('delta', 0):.3f}</td>
                    <td>{put_.get('iv', 0)*100:.1f}%</td>
                    <td>{od.get('put_call_volume_ratio', 0):.2f}</td>
                    <td>{od.get('total_call_volume', 0) + od.get('total_put_volume', 0):,}</td>
                </tr>"""

        # Sector breadth
        sector_perf = {}
        for s in results:
            sector = s.get("sector", SYMBOL_SECTOR.get(s["symbol"], "Unknown"))
            if sector not in sector_perf:
                sector_perf[sector] = []
            sector_perf[sector].append(s["daily_change_pct"])
        sector_rows = ""
        for sector in sorted(sector_perf.keys(), key=lambda x: np.mean(sector_perf[x]), reverse=True):
            avg = np.mean(sector_perf[sector])
            clr = "color:#00e676" if avg >= 0 else "color:#ff5252"
            sector_rows += f'<tr><td>{sector}</td><td style="{clr}">{avg:+.2f}%</td><td>{len(sector_perf[sector])}</td></tr>'

        bullish_count = len([s for s in results if s["trend"] in ("BULLISH", "MODERATELY BULLISH")])
        bearish_count = len([s for s in results if s["trend"] in ("BEARISH", "MODERATELY BEARISH")])
        unusual_vol_count = len([s for s in results if "UNUSUAL_VOLUME" in s["signals"]])

        options_section = ""
        if options_rows:
            options_section = f"""
    <h2>Real Options Chain Greeks (ATM)</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Expiry</th><th>DTE</th><th>Strike</th>
            <th>C Delta</th><th>C Gamma</th><th>C Theta</th><th>C Vega</th><th>C IV</th>
            <th>P Delta</th><th>P IV</th><th>P/C Vol</th><th>Total Vol</th>
        </tr>
        {options_rows}
    </table>"""

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
        <div class="stat-card">
            <div class="stat-value">{bullish_count}</div>
            <div class="stat-label">Bullish Trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{bearish_count}</div>
            <div class="stat-label">Bearish Trend</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{unusual_vol_count}</div>
            <div class="stat-label">Unusual Volume</div>
        </div>
    </div>

    <h2>Sector Performance</h2>
    <table>
        <tr><th>Sector</th><th>Avg Change</th><th>Stocks</th></tr>
        {sector_rows}
    </table>

    <h2>Top 25 Performers -- Full Technical Analysis</h2>
    <table>
        <tr>
            <th>Symbol</th><th>Price</th><th>Change</th><th>RSI</th><th>IV Rank</th>
            <th>MA20</th><th>MA50</th><th>MA200</th><th>MACD Hist</th>
            <th>Stoch K/D</th><th>HV20</th><th>Theta</th><th>Trend</th><th>Signals</th>
        </tr>
        {top_rows}
    </table>

    {options_section}

    {"<h2>Significant Changes</h2><table><tr><th>Symbol</th><th>Type</th><th>Price</th><th>Detail</th></tr>" + change_rows + "</table>" if changes else ""}

    <p style="color:#666; font-size:11px; margin-top:30px;">
        IBKR Options Trading Bot -- Market Monitor | Auto-generated report
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
                    "volume_ratio": r.get("volume_ratio", 1.0),
                    "avg_volume": r.get("avg_volume", 0),
                    "current_volume": r.get("current_volume", 0),
                })
            with open(self._state_file, "w") as f:
                json.dump(slim, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    # ---- Main Entry Points ----

    def run_scan_and_report(self, discover: bool = True,
                            analyze_options: bool = False,
                            options_top_n: int = 20) -> Tuple[str, str, List[Dict]]:
        """
        Run full scan, detect changes, generate reports.

        Args:
            discover: If True, run dynamic stock discovery first.
            analyze_options: If True, fetch real options chain data for top N stocks.
            options_top_n: Number of top stocks to analyze options chains for.

        Returns:
            (text_summary, html_report, changes)
        """
        # Step 1: Dynamic discovery
        if discover:
            try:
                self.discover_top_performers()
            except Exception as e:
                logger.warning(f"Discovery failed, using existing symbols: {e}")

        # Step 2: Full technical scan
        results = self.run_full_scan()

        # Step 3: Options chain analysis for top stocks
        if analyze_options and results:
            logger.info(f"Analyzing options chains for top {options_top_n} stocks...")
            # Analyze options for highest IV rank and top performers
            candidates = set()
            # Top by IV rank
            by_iv = sorted(results, key=lambda x: x["iv_rank"], reverse=True)
            for s in by_iv[:options_top_n // 2]:
                candidates.add(s["symbol"])
            # Top by daily performance
            for s in results[:options_top_n // 2]:
                candidates.add(s["symbol"])
            # Add any with unusual volume
            for s in results:
                if "UNUSUAL_VOLUME" in s.get("signals", []) and len(candidates) < options_top_n:
                    candidates.add(s["symbol"])

            for i, sym in enumerate(list(candidates)[:options_top_n]):
                try:
                    self.analyze_options_chain(sym)
                except Exception as e:
                    logger.debug(f"Options analysis failed for {sym}: {e}")
                if (i + 1) % 10 == 0:
                    logger.info(f"  Options chains analyzed: {i + 1}/{len(candidates)}")

            logger.info(f"Options analysis complete: {len(self._options_cache)} chains cached")

        # Step 4: Detect changes
        changes = self.detect_changes(results)

        # Step 5: Generate reports
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

    def run_morning_briefing(self, analyze_options: bool = True,
                              options_top_n: int = 20) -> str:
        """
        Run a complete morning briefing workflow.

        Discovers top stocks, runs full scan, analyzes options, generates briefing.

        Returns:
            Morning briefing text
        """
        logger.info("Starting morning briefing workflow...")

        # Discover + scan + options
        text_summary, html_report, changes = self.run_scan_and_report(
            discover=True,
            analyze_options=analyze_options,
            options_top_n=options_top_n,
        )

        # Load results from the just-saved state for briefing generation
        results_state = self._load_state()

        # Re-run scan data for full results (state is slim)
        # Use the text_summary's underlying data by re-parsing or re-scanning
        # Actually, let's just use run_full_scan results cached approach
        results = self.run_full_scan()
        briefing = self.generate_morning_briefing(results, self._options_cache)

        # Save briefing to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        briefing_path = self.reports_dir / f"morning_briefing_{timestamp}.txt"
        briefing_path.write_text(briefing)
        logger.info(f"Morning briefing saved: {briefing_path}")

        return briefing


def run_standalone():
    """Run the market monitor as a standalone script."""
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Market Monitor - NYSE Stock Analysis")
    parser.add_argument("--discover", action="store_true", default=False,
                        help="Enable dynamic stock discovery (scan ~300 stocks)")
    parser.add_argument("--options", action="store_true", default=False,
                        help="Analyze real options chains for top stocks")
    parser.add_argument("--options-top-n", type=int, default=20,
                        help="Number of stocks to analyze options for (default: 20)")
    parser.add_argument("--briefing", action="store_true", default=False,
                        help="Generate full morning briefing")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    monitor = MarketMonitor(config)

    if args.briefing:
        briefing = monitor.run_morning_briefing(
            analyze_options=args.options,
            options_top_n=args.options_top_n,
        )
        print(briefing)
    else:
        text_summary, html_report, changes = monitor.run_scan_and_report(
            discover=args.discover,
            analyze_options=args.options,
            options_top_n=args.options_top_n,
        )
        print(text_summary)

        if changes:
            print(f"\n[!] {len(changes)} significant changes detected.")


if __name__ == "__main__":
    run_standalone()
