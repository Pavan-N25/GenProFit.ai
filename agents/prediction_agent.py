# agents/prediction_agent.py
import numpy as np
import pandas as pd

class PredictionAgent:
    """
    Uses Monte Carlo simulation on historical returns to estimate
    P(>= target%) within various time horizons.
    """

    def __init__(self, n_sims=5000, seed=0):
        if n_sims < 1:
            raise ValueError("n_sims must be at least 1")
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)

    def compute_log_returns(self, price_series):
        # price_series: pd.Series of closes
        return np.log(price_series).diff().dropna()

    def monte_carlo_probabilities(self, price_series, horizons_days=[3,7,14,30,90], targets_pct=[0.05, 0.10, 0.20]):
        """
        Returns dictionary mapping each target to horizon -> probability.
        price_series: pd.Series (close prices indexed by date)
        """
        logrets = self.compute_log_returns(price_series)
        logrets = logrets.replace([np.inf, -np.inf], np.nan).dropna()
        if len(logrets) < 2 or not np.isfinite(logrets).all():
            raise ValueError("At least three valid positive closing prices are required for prediction")
        if not horizons_days or any(horizon < 1 for horizon in horizons_days):
            raise ValueError("Prediction horizons must be positive trading-day counts")
        if any(target <= 0 for target in targets_pct):
            raise ValueError("Return targets must be positive percentages")
        daily_mu = float(logrets.mean())
        daily_sigma = float(logrets.std())
        last_price = price_series.iloc[-1]
        results = {int(t*100): {} for t in targets_pct}

        # Estimate whether simulated daily paths touch each target before the horizon.
        simulated_returns = self.rng.normal(
            loc=daily_mu, scale=daily_sigma,
            size=(self.n_sims, max(horizons_days)),
        ).cumsum(axis=1)
        peak_returns = np.expm1(simulated_returns)
        for horizon in sorted(set(horizons_days)):
            for target in targets_pct:
                prob = float((peak_returns[:, :horizon].max(axis=1) >= target).mean())
                results[int(target*100)][horizon] = round(prob, 4)
        return results

    def best_horizon_for_target(self, probs_dict, target_pct, min_prob_threshold=0.6):
        """
        Choose smallest horizon where P(target) >= threshold.
        probs_dict: output of monte_carlo_probabilities
        """
        target_key = int(target_pct*100)
        horizon_probs = probs_dict.get(target_key, {})
        sorted_horizons = sorted(horizon_probs.keys())
        for h in sorted_horizons:
            if horizon_probs[h] >= min_prob_threshold:
                return h, horizon_probs[h]
        # if none meet threshold, return highest prob at longest horizon
        if sorted_horizons:
            last = sorted_horizons[-1]
            return last, horizon_probs[last]
        return None, 0.0

    def backtest_hit_rate(self, price_series, target_pct=0.10, horizon_days=30):
        """Measure historical forward-window target hits; this is descriptive, not predictive."""
        if target_pct <= 0 or horizon_days < 1:
            raise ValueError("target_pct and horizon_days must be positive")
        prices = pd.to_numeric(price_series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(prices) <= horizon_days or np.any(prices <= 0):
            return {"observations": 0, "hit_rate": None}
        hits = 0
        observations = len(prices) - horizon_days
        for index in range(observations):
            forward_peak = prices[index + 1:index + horizon_days + 1].max()
            hits += forward_peak >= prices[index] * (1 + target_pct)
        return {"observations": observations, "hit_rate": round(hits / observations, 4)}
