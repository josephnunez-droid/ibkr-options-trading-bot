"""
Dynamic NYSE Screener: Discovers the top 100 best-performing NYSE stocks.

Uses yfinance to screen a broad universe of NYSE-listed stocks, ranking them
by recent performance (price momentum, volume surge, IV rank) to dynamically
build the watchlist for the market monitor.

This replaces the static top-100 list with a live-screened one that adapts
to current market conditions.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Broad NYSE universe (~400 liquid names) to screen down to top 100.
# Covers all major sectors for comprehensive market coverage.
NYSE_BROAD_UNIVERSE = [
    # ---- Mega-Cap Tech (NYSE-listed) ----
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "TSM", "V", "MA", "AVGO", "ORCL", "CRM", "ACN", "ADBE",
    "IBM", "INTU", "NOW", "SHOP", "SNOW", "PLTR", "UBER", "SQ",
    "COIN", "CRWD", "PANW", "NET", "DDOG", "ZS", "ABNB", "DASH",
    # ---- Semiconductors ----
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "SNPS",
    "CDNS", "MRVL", "ON", "NXPI", "SWKS", "KLAC", "MPWR",
    # ---- Financials ----
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    "USB", "PNC", "TFC", "COF", "ICE", "CME", "MCO", "SPGI",
    "MMC", "AON", "AIG", "MET", "PRU", "ALL", "TRV", "CB",
    "AFL", "FITB", "HBAN", "RF", "KEY", "CFG",
    # ---- Healthcare ----
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "BSX",
    "EW", "ZTS", "REGN", "VRTX", "CI", "HUM", "ELV", "CNC",
    "HCA", "IQV", "DXCM", "IDXX", "ALGN", "BIO",
    # ---- Consumer Staples ----
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "GIS",
    "KMB", "SJM", "HSY", "MNST", "STZ", "KHC", "K", "CAG",
    "TSN", "HRL", "WBA", "KR", "SYY",
    # ---- Consumer Discretionary ----
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST",
    "DHI", "LEN", "PHM", "NVR", "BKNG", "MAR", "HLT", "LVS",
    "WYNN", "MGM", "YUM", "DPZ", "CMG", "DKNG", "RBLX",
    "GM", "F", "RIVN",
    # ---- Industrials ----
    "CAT", "DE", "HON", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "UNP", "FDX", "EMR", "ITW", "ROK", "PH", "ETN", "IR",
    "GD", "NOC", "TDG", "WM", "RSG", "VRSK", "CSGP",
    "FAST", "PAYX", "CPRT", "ODFL", "CSX", "NSC",
    # ---- Energy ----
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    "OXY", "PXD", "DVN", "HAL", "FANG", "HES", "BKR",
    "TRGP", "WMB", "KMI", "OKE", "ET",
    # ---- Communication / Media ----
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "WBD", "PARA", "LYV", "MTCH",
    # ---- Utilities ----
    "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL",
    "WEC", "ED", "PCG", "ES", "FE", "PPL", "CEG",
    # ---- REITs ----
    "AMT", "PLD", "CCI", "SPG", "O", "EQIX", "PSA", "DLR",
    "WELL", "AVB", "EQR", "VICI", "IRM", "SBAC", "ARE",
    # ---- Materials ----
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE",
    "DOW", "PPG", "VMC", "MLM", "CF", "MOS", "FMC",
    # ---- ETFs for broad market reference ----
    "SPY", "QQQ", "IWM", "DIA",
]


class NYSEScreener:
    """
    Dynamically screens the broad NYSE universe to find the top 100
    best-performing stocks based on multiple factors.
    """

    def __init__(self, config: dict, cache_dir: str = "reports"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "screener_cache.json"
        self._cache_ttl_hours = config.get("market_monitor", {}).get(
            "screener_cache_hours", 4
        )
        self._max_workers = config.get("market_monitor", {}).get(
            "screener_threads", 8
        )

    def get_top_100(self, force_refresh: bool = False) -> List[str]:
        """
        Return the top 100 best-performing NYSE stocks.

        Uses a cached result if available and fresh (< cache_ttl_hours old).
        Otherwise runs a full screen.
        """
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                logger.info(
                    f"Using cached top-100 screener results "
                    f"({len(cached)} symbols)"
                )
                return cached

        ranked = self._screen_all()
        symbols = [r["symbol"] for r in ranked[:100]]
        self._save_cache(symbols)
        return symbols

    def get_top_100_with_scores(self, force_refresh: bool = False) -> List[Dict]:
        """Return top 100 with full scoring details."""
        ranked = self._screen_all()
        return ranked[:100]

    def _screen_all(self) -> List[Dict]:
        """Screen the full broad universe in parallel and rank by composite score."""
        logger.info(
            f"Screening {len(NYSE_BROAD_UNIVERSE)} NYSE stocks "
            f"(threads={self._max_workers})..."
        )
        start = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._score_stock, sym): sym
                for sym in NYSE_BROAD_UNIVERSE
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                sym = futures[future]
                try:
                    score = future.result()
                    if score:
                        results.append(score)
                except Exception as e:
                    logger.debug(f"Screen failed for {sym}: {e}")
                if done_count % 50 == 0:
                    logger.info(
                        f"  Screened {done_count}/{len(NYSE_BROAD_UNIVERSE)}..."
                    )

        if not results:
            logger.warning("Screener returned 0 results — falling back to defaults")
            return [{"symbol": s, "composite": 0} for s in NYSE_BROAD_UNIVERSE[:100]]

        # Normalize each factor to 0-100 and compute composite
        df = pd.DataFrame(results)
        for col in ["perf_1m", "perf_5d", "volume_surge", "iv_rank"]:
            if col in df.columns and df[col].std() > 0:
                df[f"{col}_norm"] = (
                    (df[col] - df[col].min())
                    / (df[col].max() - df[col].min())
                    * 100
                )
            else:
                df[f"{col}_norm"] = 50.0

        # Composite: blend of recent performance + volume + IV activity
        df["composite"] = (
            df["perf_1m_norm"] * 0.30
            + df["perf_5d_norm"] * 0.25
            + df["volume_surge_norm"] * 0.20
            + df["iv_rank_norm"] * 0.25
        )

        df = df.sort_values("composite", ascending=False).reset_index(drop=True)
        elapsed = time.time() - start
        logger.info(
            f"Screening complete: {len(df)} stocks scored in {elapsed:.1f}s"
        )

        # Log top 10
        for _, row in df.head(10).iterrows():
            logger.info(
                f"  #{int(row.name)+1} {row['symbol']:6s} "
                f"Composite:{row['composite']:.1f} "
                f"1M:{row['perf_1m']:+.1f}% "
                f"5D:{row['perf_5d']:+.1f}% "
                f"VolSurge:{row['volume_surge']:.1f}x "
                f"IV:{row['iv_rank']:.0f}"
            )

        return df.to_dict("records")

    def _score_stock(self, symbol: str) -> Optional[Dict]:
        """Calculate screening scores for a single stock."""
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="3mo")
            if hist.empty or len(hist) < 25:
                return None

            close = hist["Close"]
            volume = hist["Volume"]
            price = float(close.iloc[-1])

            # Skip penny stocks and illiquid names
            if price < 5.0:
                return None
            avg_vol_20 = float(volume.tail(20).mean())
            if avg_vol_20 < 500_000:
                return None

            # Performance: 1-month and 5-day returns
            perf_1m = (
                (close.iloc[-1] / close.iloc[-22] - 1) * 100
                if len(close) >= 22
                else 0
            )
            perf_5d = (
                (close.iloc[-1] / close.iloc[-5] - 1) * 100
                if len(close) >= 5
                else 0
            )

            # Volume surge: current vs 20-day average
            current_vol = float(volume.iloc[-1])
            volume_surge = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

            # IV Rank approximation from HV
            returns = np.log(close / close.shift(1)).dropna()
            if len(returns) < 20:
                return None
            hv_20 = float(returns.tail(20).std() * np.sqrt(252) * 100)

            # Need longer history for IV rank
            hist_long = tk.history(period="1y")
            if len(hist_long) >= 60:
                ret_long = np.log(
                    hist_long["Close"] / hist_long["Close"].shift(1)
                ).dropna()
                rolling_hv = ret_long.rolling(20).std() * np.sqrt(252) * 100
                rolling_hv = rolling_hv.dropna()
                hv_low = float(rolling_hv.min())
                hv_high = float(rolling_hv.max())
                iv_rank = (
                    (hv_20 - hv_low) / (hv_high - hv_low) * 100
                    if hv_high != hv_low
                    else 50.0
                )
            else:
                iv_rank = 50.0

            return {
                "symbol": symbol,
                "price": round(price, 2),
                "perf_1m": round(float(perf_1m), 2),
                "perf_5d": round(float(perf_5d), 2),
                "volume_surge": round(float(volume_surge), 2),
                "avg_volume": round(avg_vol_20),
                "iv_rank": round(float(max(0, min(100, iv_rank))), 1),
                "hv_20": round(hv_20, 2),
            }
        except Exception as e:
            logger.debug(f"Score failed for {symbol}: {e}")
            return None

    # ---- Cache ----

    def _load_cache(self) -> Optional[List[str]]:
        """Load cached screener results if fresh."""
        try:
            if not self._cache_file.exists():
                return None
            data = json.loads(self._cache_file.read_text())
            cached_at = datetime.fromisoformat(data["timestamp"])
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            if age_hours > self._cache_ttl_hours:
                logger.info(
                    f"Screener cache expired ({age_hours:.1f}h old, "
                    f"ttl={self._cache_ttl_hours}h)"
                )
                return None
            return data["symbols"]
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return None

    def _save_cache(self, symbols: List[str]):
        """Save screener results to cache."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "count": len(symbols),
                "symbols": symbols,
            }
            self._cache_file.write_text(json.dumps(data, indent=2))
            logger.info(f"Screener cache saved ({len(symbols)} symbols)")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
