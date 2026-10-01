# agents/decision_agent.py
from typing import Dict

class DecisionAgent:
    def __init__(self):
        pass

    def make_decision(self, ticker, financials_report: Dict, predict_probs: Dict, best_horizons: Dict,
                      risk_rating: str, fee_tax_report: Dict, sentiment_score: float, user_threshold=0.6,
                      beginner_mode=False):
        """
        Simple policy:
          - If probability for +10% within recommended horizon >= user_threshold and risk not High -> BUY
          - Else HOLD or AVOID
        """
        decision = "HOLD"
        reasons = []
        horizon, prob = best_horizons.get(10, (None, 0.0))
        if risk_rating == "High":
            decision = "AVOID"
            reasons.append("The estimated risk is high for this analysis.")
        elif fee_tax_report.get("net_profit", 0) <= 0:
            decision = "AVOID"
            reasons.append("Estimated fees and taxes would erase the modeled +10% gain for this trade size.")
        elif prob >= user_threshold:
            decision = "BUY"
            reasons.append(f"Estimated chance of reaching +10% is {prob:.0%} within about {horizon} trading days.")
        else:
            reasons.append(f"Estimated chance of reaching +10% is {prob:.0%}, below the selected {user_threshold:.0%} threshold.")
        report = {
            "ticker": ticker,
            "decision": decision,
            "reasons": reasons,
            "financials": financials_report,
            "probabilities": predict_probs,
            "best_horizons": best_horizons,
            "risk": risk_rating,
            "fee_tax": fee_tax_report,
            "sentiment": sentiment_score
        }
        if beginner_mode:
            report["beginner_summary"] = (
                f"{ticker}: {'Consider investing only money you can afford to lose.' if decision == 'BUY' else 'Avoid investing based on this analysis.'} "
                f"Risk is {risk_rating.lower()}. These estimates can be wrong."
            )
            report["beginner_decision"] = "Invest" if decision == "BUY" else "Avoid"
        report["disclaimer"] = "For education only, not financial advice. Prices, probabilities, fees, and tax estimates are uncertain."
        return report
