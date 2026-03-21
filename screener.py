"""Dynamic universe builder: screens stocks by IV rank, volume, and momentum."""

import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Large-cap universe to screen from
SCREENING_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "BAC",
    "NFLX", "CRM", "AMD", "INTC", "COST", "DIS", "PYPL",
    "PFE", "ABBV", "MRK", "KO", "PEP", "TMO", "AVGO", "ORCL",
    "CSCO", "ACN", "TXN", "QCOM", "INTU", "AMAT", "ISRG",
    "NOW", "ADP", "LRCX", "MU", "PANW", "SNPS", "CDNS",
    "COIN", "MARA", "PLTR", "RIVN", "SOFI", "SQ", "UBER",
    "ABNB", "DASH", "DKNG", "RBLX", "SNOW", "NET", "CRWD",
]


class Screener:
    """Screens and ranks stocks for options trading opportunities."""

    def __init__(self, config: dict, data_feed=None):
        self.config = config
        self.data_feed = data_feed
        self.screener_config = config.get("universe", {}).get("dynamic_screener", {})
        self.top_n = self.screener_config.get("top_n", 20)
        self.min_option_volume = self.screener_config.get("min_option_volume", 1000)
        self._last_screen: Dict[str, pd.DataFrame] = {}
        self._last_screen_time = None

    def build_universe(self) -> List[str]:
        """
        Build the complete trading universe:
        1. Static Mag7 + ETFs from config
        2. Dynamic top-N from screener
        """
        universe = set()

        mag7 = self.config.get("universe", {}).get("mag7", [])
        etfs = self.config.get("universe", {}).get("etfs", [])
        universe.update(mag7)
        universe.update(etfs)

        if self.screener_config.get("enabled", True):
            try:
                dynamic = self.screen_top_stocks()
                universe.update(dynamic)
                logger.info(f"Dynamic screener added {len(dynamic)} stocks")
            except Exception as e:
                logger.error(f"Dynamic screener failed: {e}")

        result = sorted(universe)
        logger.info(f"Trading universe: {len(result)} symbols")
        return result

    def screen_top_stocks(self) -> List[str]:
        """
        Screen for top stocks by combined score of:
        - IV Rank (40% weight)
        - Options volume (30% weight)
        - Price momentum (30% weight)
        """
        logger.info(f"Screening {len(SCREENING_UNIVERSE)} stocks...")

        results = []
        for symbol in SCREENING_UNIVERSE:
            try:
                score = self._score_stock(symbol)
                if score is not None:
                    results.append(score)
            except Exception as e:
                logger.debug(f"Screening failed for {symbol}: {e}")
                continue

        if not results:
            logger.warning("Screener returned no results")
            return []

        df = pd.DataFrame(results)
        self._last_screen["results"] = df
        self._last_screen_time = datetime.now()

        for col in ["iv_rank", "volume_score", "momentum_score"]:
            if df[col].std() > 0:
                df[f"{col}_norm"] = (
                    (df[col] - df[col].min()) /
                    (df[col].max() - df[col].min()) * 100
                )
            else:
                df[f"{col}_norm"] = 50

        df["composite"] = (
            df["iv_rank_norm"] * 0.40 +
            df["volume_score_norm"] * 0.30 +
            df["momentum_score_norm"] * 0.30
        )

        df = df.sort_values("composite", ascending=False)
        top = df.head(self.top_n)

        logger.info(f"Top {self.top_n} screened stocks:")
        for _, row in top.iterrows():
            logger.info(
                f"  {row['symbol']:6s} | IV Rank: {row['iv_rank']:.0f} | "
                f"Vol Score: {row['volume_score']:.0f} | "
                f"Mom: {row['momentum_score']:.1f} | "
                f"Composite: {row['composite']:.1f}"
            )

        return top["symbol"].tolist()

    def _score_stock(self, symbol: str) -> dict:
        """Calculate screening scores for a single stock."""
        tk = yf.Ticker(symbol)
        hist = tk.history(period="3mo")
        if hist.empty or len(hist) < 20:
            return None

        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(returns) < 20:
            return None

        hv_20 = returns.tail(20).std() * np.sqrt(252) * 100

        hist_1y = tk.history(period="1y")
        if len(hist_1y) < 60:
            return None

        returns_1y = np.log(hist_1y["Close"] / hist_1y["Close"].shift(1)).dropna()
        rolling_hv = returns_1y.rolling(20).std() * np.sqrt(252) * 100
        rolling_hv = rolling_hv.dropna()

        hv_low = rolling_hv.min()
        hv_high = rolling_hv.max()
        iv_rank = (
            (hv_20 - hv_low) / (hv_high - hv_low) * 100
            if hv_high != hv_low else 50
        )

        avg_volume = hist["Volume"].tail(20).mean()
        volume_score = min(avg_volume / 1_000_000, 100)

        momentum = (
            (hist["Close"].iloc[-1] / hist["Close"].iloc[-20] - 1) * 100
        )

        return {
            "symbol": symbol,
            "price": float(hist["Close"].iloc[-1]),
            "iv_rank": float(iv_rank),
            "volume_score": float(volume_score),
            "momentum_score": float(abs(momentum)),
            "momentum_direction": "UP" if momentum > 0 else "DOWN",
            "hv_20": float(hv_20),
            "avg_volume": float(avg_volume),
        }

    def get_top_by_iv_rank(self, n: int = 10) -> List[Dict]:
        """Get top N stocks by IV rank."""
        results = []
        for symbol in SCREENING_UNIVERSE:
            try:
                score = self._score_stock(symbol)
                if score:
                    results.append(score)
            except Exception:
                continue

        df = pd.DataFrame(results)
        if df.empty:
            return []

        return (
            df.sort_values("iv_rank", ascending=False)
            .head(n)
            .to_dict("records")
        )

    def get_high_iv_range_bound(
        self,
        symbols: List[str],
        min_iv_rank: float = 40,
    ) -> List[str]:
        """Find stocks with high IV that are range-bound (good for iron condors)."""
        candidates = []
        for symbol in symbols:
            try:
                score = self._score_stock(symbol)
                if score is None:
                    continue
                if score["iv_rank"] < min_iv_rank:
                    continue
                if abs(score["momentum_score"]) < 5:
                    candidates.append(symbol)
            except Exception:
                continue
        return candidates

    def get_directional_candidates(
        self,
        symbols: List[str],
    ) -> Dict[str, str]:
        """
        Determine directional bias using SMA crossover.
        Returns dict of {symbol: "BULL" or "BEAR"}
        """
        directions = {}
        for symbol in symbols:
            try:
                tk = yf.Ticker(symbol)
                hist = tk.history(period="3mo")
                if hist.empty or len(hist) < 50:
                    continue

                sma_20 = hist["Close"].tail(20).mean()
                sma_50 = hist["Close"].tail(50).mean()

                if sma_20 > sma_50:
                    directions[symbol] = "BULL"
                else:
                    directions[symbol] = "BEAR"
            except Exception:
                continue
        return directions

    def get_earnings_plays(
        self,
        symbols: List[str],
        days_before: int = 7,
    ) -> List[Dict]:
        """Find stocks with earnings coming up within specified days."""
        plays = []
        for symbol in symbols:
            try:
                if self.data_feed:
                    days = self.data_feed.days_to_earnings(symbol)
                else:
                    tk = yf.Ticker(symbol)
                    dates = tk.earnings_dates
                    if dates is None or dates.empty:
                        continue
                    future = dates[dates.index >= pd.Timestamp.now()]
                    if future.empty:
                        continue
                    days = (future.index[0].date() - pd.Timestamp.now().date()).days

                if days is not None and 0 < days <= days_before:
                    plays.append({
                        "symbol": symbol,
                        "days_to_earnings": days,
                    })
            except Exception:
                continue

        return sorted(plays, key=lambda x: x["days_to_earnings"])
