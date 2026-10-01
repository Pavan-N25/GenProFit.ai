# agents/market_data_agent.py
import pandas as pd
import numpy as np

class MarketDataAgent:
    """
    Responsible for fetching or loading historical price data.
    For this prototype there are two modes:
      - load from CSV (if path provided)
      - generate synthetic series (demo mode)
    """

    def load_from_csv(self, path, price_col="close", date_col="date"):
        df = pd.read_csv(path)
        missing = {date_col, price_col} - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
        df = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().sort_values("date").drop_duplicates("date")
        df = df[df["close"] > 0].reset_index(drop=True)
        if len(df) < 2:
            raise ValueError("CSV must contain at least two valid, positive closing prices")
        return df

    def fetch_yahoo(self, ticker, period="2y"):
        """Fetch adjusted daily closes from Yahoo Finance via the optional yfinance package."""
        try:
            import yfinance as yf
        except ImportError as error:
            raise RuntimeError("Yahoo data requires yfinance. Install project requirements first.") from error

        history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if history.empty or "Close" not in history:
            raise ValueError(f"No Yahoo Finance price history was returned for {ticker}")
        result = history[["Close"]].reset_index()
        result.columns = ["date", "close"]
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["close"] = pd.to_numeric(result["close"], errors="coerce")
        result = result.dropna().sort_values("date").reset_index(drop=True)
        if len(result) < 2:
            raise ValueError(f"Not enough Yahoo Finance price history was returned for {ticker}")
        return result

    def generate_synthetic(self, days=365, seed=0, start_price=100.0, mu=0.0002, sigma=0.02):
        """Generate synthetic daily close prices using geometric Brownian motion."""
        if days < 2:
            raise ValueError("Synthetic price history must contain at least two days")
        rng = np.random.default_rng(seed)
        dt = 1/252
        prices = [start_price]
        for _ in range(days-1):
            shock = rng.normal(loc=mu*dt, scale=sigma*(dt**0.5))
            next_price = prices[-1] * (1 + shock)
            prices.append(max(0.01, next_price))
        import pandas as pd
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        return pd.DataFrame({"date": dates, "close": prices})
