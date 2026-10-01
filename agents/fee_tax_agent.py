# agents/fee_tax_agent.py
import json
from pathlib import Path

class FeeTaxAgent:
    """
    Loads platform fee profiles and computes net outcome after fees/taxes.
    """

    def __init__(self, profiles_path="tools/platform_profiles.json", tax_bracket_pct=0.30):
        path = Path(profiles_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        with path.open("r", encoding="utf-8") as f:
            self.profiles = json.load(f)
        self.tax_bracket = tax_bracket_pct

    def compute_fees(self, platform_key, trade_type, price, quantity):
        """
        trade_type: 'delivery' or 'intraday'
        returns dict of fee components
        """
        if platform_key not in self.profiles:
            raise ValueError(f"Unknown platform '{platform_key}'. Choose from: {', '.join(sorted(self.profiles))}")
        if trade_type not in ("delivery", "intraday"):
            raise ValueError("trade_type must be 'delivery' or 'intraday'")
        if price <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        p = self.profiles[platform_key]
        notional = price * quantity
        brk_info = p["brokerage"]
        if "per_share" in brk_info:
            brokerage = max(quantity * brk_info["per_share"], brk_info.get("minimum_per_order", 0.0))
            brokerage = min(brokerage, notional * brk_info.get("maximum_pct", 1.0))
        elif trade_type == "delivery":
            brokerage = notional * brk_info.get("delivery_pct", 0.0)
        else:
            brokerage = max(notional * brk_info.get("intraday_pct", 0.0) / 100.0, brk_info.get("min_intraday", 0.0))
            if "max_intraday" in brk_info:
                brokerage = min(brokerage, brk_info["max_intraday"])
        stt = notional * p.get("stt_pct", 0.0)
        gst = brokerage * p.get("gst_pct", 0.0)
        sebi = notional * p.get("sebi_charges_pct", 0.0)
        stamp = notional * p.get("stamp_duty_pct", 0.0)
        total_fees = brokerage + stt + gst + sebi + stamp
        return {
            "notional": notional,
            "currency": p.get("currency", "INR"),
            "brokerage": brokerage,
            "stt": stt,
            "gst": gst,
            "sebi": sebi,
            "stamp": stamp,
            "total_fees": total_fees
        }

    def compute_net_after_profit(self, platform_key, trade_type, buy_price, sell_price, quantity, is_long_term=False):
        """
        Computes net profit after fees and estimated taxes.
        is_long_term flag toggles capital gains treatment for prototype.
        """
        buy_fees = self.compute_fees(platform_key, trade_type, buy_price, quantity)
        sell_fees = self.compute_fees(platform_key, trade_type, sell_price, quantity)
        gross_profit = (sell_price - buy_price) * quantity
        total_fees = buy_fees["total_fees"] + sell_fees["total_fees"]
        # simple capital gains tax handling
        if gross_profit <= 0:
            tax = 0.0
        else:
            if is_long_term:
                # reduced rate example
                tax = gross_profit * 0.10
            else:
                tax = gross_profit * self.tax_bracket
        net_profit = gross_profit - total_fees - tax
        net_return_pct = (net_profit / (buy_price * quantity)) if (buy_price*quantity) > 0 else 0.0
        return {
            "gross_profit": gross_profit,
            "total_fees": total_fees,
            "tax": tax,
            "net_profit": net_profit,
            "net_return_pct": net_return_pct,
            "currency": buy_fees.get("currency", "INR"),
            "breakdown": {"buy": buy_fees, "sell": sell_fees}
        }
