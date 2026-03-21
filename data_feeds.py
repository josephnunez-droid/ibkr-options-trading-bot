"""Market data feeds: options chains, IV rank, historical data, earnings."""

import logging
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from ib_insync import IB, Stock, Option, Index, util
import yfinance as yf

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ETFs don't have earnings — skip fundamentals lookups for these
KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "GLD", "XLK", "XLF", "SOXX", "VXX",
    "DIA", "TLT", "HYG", "EEM", "XLE", "XLP", "XLV", "XLI",
    "XLU", "XLB", "XLRE", "XLC", "ARKK", "SQQQ", "TQQQ",
}


class DataFeed:
    """Provides market data via IBKR TWS API with yfinance fallback."""

    def __init__(self, ib: IB, config: dict):
        self.ib = ib
        self.config = config
        self._iv_cache: Dict[str, Tuple[float, datetime]] = {}
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = 60  # seconds

    def get_contract(self, symbol: str) -> Stock:
        """Create and qualify a stock contract."""
        contract = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
        raise ValueError(f"Could not qualify contract for {symbol}")

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        # Check cache
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if (datetime.now() - ts).seconds < self._cache_ttl:
                return price

        try:
            contract = self.get_contract(symbol)
            self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1)
            ticker = self.ib.ticker(contract)

            price = ticker.marketPrice()
            if np.isnan(price):
                price = ticker.close
            if np.isnan(price):
                price = self._yf_price(symbol)

            self._price_cache[symbol] = (price, datetime.now())
            return price

        except Exception as e:
            logger.warning(f"IBKR price failed for {symbol}: {e}, using yfinance")
            return self._yf_price(symbol)

    def _yf_price(self, symbol: str) -> float:
        """Fallback price from yfinance."""
        try:
            tk = yf.Ticker(symbol)
            data = tk.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            logger.error(f"yfinance price failed for {symbol}: {e}")
        return 0.0

    def get_options_chain(
        self,
        symbol: str,
        min_dte: int = 25,
        max_dte: int = 50,
        right: str = None,
    ) -> pd.DataFrame:
        """
        Fetch options chain for a symbol within DTE range.

        Returns DataFrame with columns:
            symbol, strike, expiry, right, bid, ask, mid, last,
            volume, openInterest, impliedVol, delta, gamma, theta, vega, dte
        """
        contract = self.get_contract(symbol)
        chains = self.ib.reqSecDefOptParams(
            contract.symbol, "", contract.secType, contract.conId
        )

        if not chains:
            logger.warning(f"No options chains found for {symbol}")
            return pd.DataFrame()

        # Use SMART exchange chain
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        today = date.today()
        target_expiries = []
        for exp_str in chain.expirations:
            exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                target_expiries.append(exp_str)

        if not target_expiries:
            logger.warning(f"No expiries in DTE range [{min_dte}, {max_dte}] for {symbol}")
            return pd.DataFrame()

        # Get current price to filter relevant strikes
        price = self.get_current_price(symbol)
        strike_range = price * 0.20
        relevant_strikes = [
            s for s in chain.strikes
            if price - strike_range <= s <= price + strike_range
        ]

        rows = []
        rights = [right] if right else ["C", "P"]

        for expiry in target_expiries:
            for r in rights:
                contracts = [
                    Option(symbol, expiry, strike, r, "SMART")
                    for strike in relevant_strikes
                ]

                # Qualify in batches
                qualified = []
                batch_size = 50
                for i in range(0, len(contracts), batch_size):
                    batch = contracts[i:i + batch_size]
                    qualified.extend(self.ib.qualifyContracts(*batch))

                if not qualified:
                    continue

                # Request market data
                tickers = []
                for c in qualified:
                    ticker = self.ib.reqMktData(c, "100,101,104,106", False, False)
                    tickers.append((c, ticker))

                self.ib.sleep(2)

                exp_date = datetime.strptime(expiry, "%Y%m%d").date()
                dte = (exp_date - today).days

                for c, t in tickers:
                    bid = t.bid if not np.isnan(t.bid) else 0
                    ask = t.ask if not np.isnan(t.ask) else 0
                    mid = (bid + ask) / 2 if bid and ask else 0

                    row = {
                        "symbol": symbol,
                        "strike": c.strike,
                        "expiry": exp_date,
                        "right": r,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": t.last if not np.isnan(t.last) else 0,
                        "volume": t.volume if t.volume else 0,
                        "openInterest": (
                            getattr(t, "callOpenInterest", 0) if r == "C"
                            else getattr(t, "putOpenInterest", 0)
                        ),
                        "impliedVol": (
                            t.impliedVolatility
                            if hasattr(t, "impliedVolatility") and t.impliedVolatility
                            else 0
                        ),
                        "delta": getattr(t.modelGreeks, "delta", 0) if t.modelGreeks else 0,
                        "gamma": getattr(t.modelGreeks, "gamma", 0) if t.modelGreeks else 0,
                        "theta": getattr(t.modelGreeks, "theta", 0) if t.modelGreeks else 0,
                        "vega": getattr(t.modelGreeks, "vega", 0) if t.modelGreeks else 0,
                        "dte": dte,
                    }
                    rows.append(row)

                # Cancel market data
                for c, _ in tickers:
                    self.ib.cancelMktData(c)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["expiry", "strike", "right"]).reset_index(drop=True)
        return df

    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate IV Rank: where current IV sits relative to past year.
        IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) * 100
        """
        if symbol in self._iv_cache:
            rank, ts = self._iv_cache[symbol]
            if (datetime.now() - ts).seconds < 300:
                return rank

        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1y")
            if hist.empty or len(hist) < 20:
                return 50.0

            returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100

            current_vol = rolling_vol.iloc[-1]
            low_vol = rolling_vol.min()
            high_vol = rolling_vol.max()

            if high_vol == low_vol:
                rank = 50.0
            else:
                rank = float((current_vol - low_vol) / (high_vol - low_vol) * 100)

            rank = max(0, min(100, rank))
            self._iv_cache[symbol] = (rank, datetime.now())
            return rank

        except Exception as e:
            logger.error(f"IV rank calculation failed for {symbol}: {e}")
            return 50.0

    def get_iv_percentile(self, symbol: str, lookback_days: int = 252) -> float:
        """Calculate IV Percentile: % of days in past year with lower IV."""
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1y")
            if hist.empty or len(hist) < 20:
                return 50.0

            returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()

            current_vol = rolling_vol.iloc[-1]
            pct = float((rolling_vol < current_vol).sum() / len(rolling_vol) * 100)
            return max(0, min(100, pct))

        except Exception as e:
            logger.error(f"IV percentile failed for {symbol}: {e}")
            return 50.0

    def get_vix(self) -> float:
        """Get current VIX level."""
        try:
            contract = Index("VIX", "CBOE")
            self.ib.qualifyContracts(contract)
            self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1)
            ticker = self.ib.ticker(contract)
            vix = ticker.marketPrice()

            if np.isnan(vix):
                vix = ticker.close
            if np.isnan(vix):
                return self._yf_vix()

            self.ib.cancelMktData(contract)
            return float(vix)

        except Exception as e:
            logger.warning(f"IBKR VIX failed: {e}, using yfinance")
            return self._yf_vix()

    def _yf_vix(self) -> float:
        """Fallback VIX from yfinance."""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return 20.0

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "60 D",
        bar_size: str = "1 day",
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        try:
            contract = self.get_contract(symbol)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                return self._yf_history(symbol, duration)

            df = util.df(bars)
            df = df.rename(columns={
                "date": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume"
            })
            return df

        except Exception as e:
            logger.warning(f"IBKR history failed for {symbol}: {e}")
            return self._yf_history(symbol, duration)

    def _yf_history(self, symbol: str, duration: str) -> pd.DataFrame:
        """Fallback history from yfinance."""
        days = int(duration.split()[0])
        period = f"{days}d" if days <= 30 else f"{days // 30}mo"
        try:
            tk = yf.Ticker(symbol)
            return tk.history(period=period)
        except Exception:
            return pd.DataFrame()

    def get_sma(self, symbol: str, period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        df = self.get_historical_data(symbol, f"{period * 2} D", "1 day")
        if df.empty or len(df) < period:
            return None
        close_col = "Close" if "Close" in df.columns else "close"
        return float(df[close_col].tail(period).mean())

    def get_earnings_date(self, symbol: str) -> Optional[date]:
        """Get next earnings date for a symbol."""
        if symbol.upper() in KNOWN_ETFS:
            return None
        try:
            tk = yf.Ticker(symbol)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                if hasattr(cal, "iloc"):
                    earnings_date = pd.Timestamp(cal.iloc[0, 0])
                    return earnings_date.date()
            dates = tk.earnings_dates
            if dates is not None and not dates.empty:
                future_dates = dates[dates.index >= pd.Timestamp.now()]
                if not future_dates.empty:
                    return future_dates.index[0].date()
        except Exception as e:
            logger.debug(f"Earnings date lookup failed for {symbol}: {e}")
        return None

    def days_to_earnings(self, symbol: str) -> Optional[int]:
        """Get number of days until next earnings."""
        ed = self.get_earnings_date(symbol)
        if ed:
            delta = (ed - date.today()).days
            return delta if delta >= 0 else None
        return None

    def get_option_by_delta(
        self,
        symbol: str,
        right: str,
        target_delta: float,
        min_dte: int = 30,
        max_dte: int = 45,
    ) -> Optional[dict]:
        """
        Find the option contract closest to target delta.

        Args:
            symbol: Underlying symbol
            right: "C" for call, "P" for put
            target_delta: Target delta (positive for calls, negative for puts)
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry

        Returns:
            Dict with option details or None
        """
        chain = self.get_options_chain(symbol, min_dte, max_dte, right)
        if chain.empty:
            return None

        chain = chain[chain["delta"] != 0].copy()
        if chain.empty:
            return None

        chain["delta_diff"] = abs(chain["delta"] - target_delta)
        best = chain.loc[chain["delta_diff"].idxmin()]

        return best.to_dict()

    def get_portfolio_positions(self) -> List[dict]:
        """Get current portfolio positions from IBKR."""
        positions = []
        for pos in self.ib.positions():
            contract = pos.contract
            positions.append({
                "symbol": contract.symbol,
                "secType": contract.secType,
                "strike": getattr(contract, "strike", None),
                "right": getattr(contract, "right", None),
                "expiry": getattr(contract, "lastTradeDateOrContractMonth", None),
                "quantity": pos.position,
                "avgCost": pos.avgCost,
                "account": pos.account,
            })
        return positions

    def get_account_values(self) -> dict:
        """Get key account values."""
        values = {}
        for av in self.ib.accountSummary():
            if av.currency == "USD":
                try:
                    values[av.tag] = float(av.value)
                except (ValueError, TypeError):
                    values[av.tag] = av.value
        return values
