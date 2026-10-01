import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agents.memory_agent import MemoryAgent
from agents.fee_tax_agent import FeeTaxAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.prediction_agent import PredictionAgent
from main import run_analysis


class AgentTests(unittest.TestCase):
    def test_broker_profiles_apply_currency_and_commission_models(self):
        agent = FeeTaxAgent()
        robinhood = agent.compute_fees("robinhood", "delivery", 100, 10)
        interactive_brokers = agent.compute_fees("interactive_brokers", "delivery", 100, 10)
        self.assertEqual(robinhood["currency"], "USD")
        self.assertEqual(robinhood["total_fees"], 0)
        self.assertEqual(interactive_brokers["currency"], "USD")
        self.assertEqual(interactive_brokers["brokerage"], 1.0)

    def test_sentiment_is_deterministic_and_neutral_without_headlines(self):
        agent = NewsSentimentAgent()
        headlines = [{"title": "Strong profit growth"}, {"title": "Fraud investigation"}]
        self.assertEqual(agent.sentiment_score(headlines), 0.0)
        self.assertEqual(agent.sentiment_score([]), 0.0)

    def test_path_probabilities_increase_with_horizon_and_fall_with_target(self):
        prices = pd.Series([100 * (1.001 ** day) for day in range(200)])
        probabilities = PredictionAgent(n_sims=1000, seed=7).monte_carlo_probabilities(
            prices, horizons_days=[3, 14, 30], targets_pct=[0.05, 0.10, 0.20],
        )
        for target in (5, 10, 20):
            self.assertLessEqual(probabilities[target][3], probabilities[target][14])
            self.assertLessEqual(probabilities[target][14], probabilities[target][30])
        for horizon in (3, 14, 30):
            self.assertGreaterEqual(probabilities[5][horizon], probabilities[10][horizon])
            self.assertGreaterEqual(probabilities[10][horizon], probabilities[20][horizon])

    def test_memory_persists_sessions_and_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryAgent(Path(directory) / "memory.json")
            memory.save_session("run-1", {"decision": "HOLD"})
            memory.set_preference("platform", "zerodha")
            self.assertEqual(memory.get_session("run-1"), {"decision": "HOLD"})
            self.assertEqual(memory.get_preferences(), {"platform": "zerodha"})

    def test_synthetic_analysis_completes_and_saves_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(NewsSentimentAgent, "fetch_news", return_value=[]):
                report = run_analysis(
                    ticker="TEST", source="synthetic", n_sims=100, seed=3,
                    beginner_mode=True, backtest=True,
                    memory_path=Path(directory) / "memory.json",
                )
            self.assertEqual(report["data_source"], "synthetic")
            self.assertEqual(report["beginner_decision"], "Avoid")
            self.assertIn("trace_id", report["trace"])
            self.assertEqual(report["historical_backtest"]["observations"], 474)
            self.assertTrue((Path(directory) / "memory.json").exists())


if __name__ == "__main__":
    unittest.main()