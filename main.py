# main.py
import argparse
import json

from agents.decision_agent import DecisionAgent
from agents.fee_tax_agent import FeeTaxAgent
from agents.financials_agent import FinancialsAgent
from agents.logging_agent import LoggingAgent
from agents.market_data_agent import MarketDataAgent
from agents.memory_agent import MemoryAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import risk_score_from_vol_sentiment


def run_analysis(ticker="SYNTH", source="synthetic", csv_path=None, user_platform="zerodha",
                 quantity=10, n_sims=3000, seed=42, threshold=0.6, beginner_mode=False,
                 backtest=False, trade_type="delivery", memory_path="data/memory.json"):
    logger = LoggingAgent()
    trace_id = logger.start_trace(f"analysis-{ticker}")
    market = MarketDataAgent()

    logger.log_event(trace_id, "load_market_data", {"source": source, "ticker": ticker})
    if source == "csv":
        if not csv_path:
            raise ValueError("Provide --csv-path when using --source csv")
        prices_frame = market.load_from_csv(csv_path)
    elif source == "yahoo":
        prices_frame = market.fetch_yahoo(ticker)
    elif source == "synthetic":
        prices_frame = market.generate_synthetic(days=504, seed=seed)
    else:
        raise ValueError(f"Unsupported data source: {source}")
    prices = prices_frame["close"].astype(float).reset_index(drop=True)
    current_price = float(prices.iloc[-1])

    logger.log_event(trace_id, "compute_financials")
    if source == "yahoo":
        financials = FinancialsAgent().fetch_yahoo(ticker)
    else:
        financials = {"note": f"Financial statements are not available for {source} price data"}

    logger.log_event(trace_id, "fetch_news")
    news_agent = NewsSentimentAgent()
    headlines = news_agent.fetch_news(ticker, limit=5)
    sentiment = news_agent.sentiment_score(headlines)

    logger.log_event(trace_id, "simulate_target_probabilities", {"simulations": n_sims})
    prediction = PredictionAgent(n_sims=n_sims, seed=seed)
    horizons = [3, 7, 14, 30, 90]
    targets = [0.05, 0.10, 0.20]
    probabilities = prediction.monte_carlo_probabilities(prices, horizons, targets)
    best_horizons = {
        int(target * 100): prediction.best_horizon_for_target(probabilities, target, threshold)
        for target in targets
    }

    log_returns = prediction.compute_log_returns(prices)
    annualized_volatility = float(log_returns.std() * (252 ** 0.5))
    risk_rating, risk_score = risk_score_from_vol_sentiment(annualized_volatility, sentiment)
    fee_report = FeeTaxAgent().compute_net_after_profit(
        user_platform, trade_type, current_price, current_price * 1.10, quantity,
    )

    logger.log_event(trace_id, "make_decision")
    report = DecisionAgent().make_decision(
        ticker=ticker,
        financials_report=financials,
        predict_probs=probabilities,
        best_horizons=best_horizons,
        risk_rating=risk_rating,
        fee_tax_report=fee_report,
        sentiment_score=sentiment,
        user_threshold=threshold,
        beginner_mode=beginner_mode,
    )
    report.update({
        "data_source": source,
        "price_as_of": str(prices_frame["date"].iloc[-1]),
        "current_price": current_price,
        "price_observations": len(prices),
        "annualized_volatility": annualized_volatility,
        "risk_score": round(risk_score, 4),
        "headlines": headlines,
        "news_note": None if headlines else "No headlines were available; sentiment was treated as neutral.",
        "fee_assumption": f"Estimated {trade_type} charges and a simplified tax rate; verify current broker and local tax rules.",
        "trace_id": trace_id,
    })
    if backtest:
        report["historical_backtest"] = {
            "target_pct": 0.10,
            "horizon_days": 30,
            **prediction.backtest_hit_rate(prices, target_pct=0.10, horizon_days=30),
            "note": "Historical forward-window hit rate only; it is not a measure of future performance.",
        }

    logger.log_event(trace_id, "save_session")
    report["trace"] = logger.end_trace(trace_id)
    MemoryAgent(storage_path=memory_path).save_session(f"{ticker}-{trace_id[:8]}", report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Beginner-friendly stock analysis research prototype")
    parser.add_argument("ticker", nargs="?", default="SYNTH", help="Ticker symbol, e.g. TCS.NS")
    parser.add_argument("--source", choices=("synthetic", "csv", "yahoo"), default="synthetic")
    parser.add_argument("--csv-path", help="CSV with date and close columns (for --source csv)")
    parser.add_argument("--platform", default="zerodha", help="Fee profile key from tools/platform_profiles.json")
    parser.add_argument("--quantity", type=int, default=10, help="Number of shares used for the fee estimate")
    parser.add_argument("--simulations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold from 0 to 1 (0%% to 100%%)")
    parser.add_argument("--trade-type", choices=("delivery", "intraday"), default="delivery")
    parser.add_argument("--beginner", action="store_true", help="Include a plain-language Invest/Avoid summary")
    parser.add_argument("--backtest", action="store_true", help="Include historical +10 pct/30-day hit rate")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quantity < 1:
        raise SystemExit("--quantity must be at least 1")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    try:
        report = run_analysis(
            ticker=args.ticker,
            source=args.source,
            csv_path=args.csv_path,
            user_platform=args.platform,
            quantity=args.quantity,
            n_sims=args.simulations,
            seed=args.seed,
            threshold=args.threshold,
            beginner_mode=args.beginner,
            backtest=args.backtest,
            trade_type=args.trade_type,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"GenProFit.ai analysis: {report['ticker']} ({report['data_source']} data)")
        print(f"Decision: {report['decision']} | Risk: {report['risk']}")
        print(f"Last close: {report['current_price']:.2f} as of {report['price_as_of']}")
        print(f"Estimated net profit at +10% target: {report['fee_tax']['net_profit']:.2f}")
        print(report.get("beginner_summary", report["disclaimer"]))
        print(f"Full report saved to local memory. Trace: {report['trace_id']}")


if __name__ == "__main__":
    main()
