# agents/financials_agent.py
class FinancialsAgent:
    """
    Compute YoY and CAGR-like metrics from provided financials dictionary.
    For this prototype we accept a small dict of yearly revenue/EPS numbers.
    """

    def compute_cagr(self, start_value, end_value, years):
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return None
        return (end_value / start_value) ** (1.0/years) - 1.0

    def summarize_growth(self, yearly_financials):
        """
        yearly_financials: dict year -> {"revenue": float, "eps": float}
        returns growth report
        """
        years = sorted(yearly_financials.keys())
        if len(years) < 2:
            return {"note": "insufficient data"}
        start_year, end_year = years[0], years[-1]
        years_count = end_year - start_year
        rev_start = yearly_financials[start_year]["revenue"]
        rev_end = yearly_financials[end_year]["revenue"]
        eps_start = yearly_financials[start_year]["eps"]
        eps_end = yearly_financials[end_year]["eps"]
        cagr_rev = self.compute_cagr(rev_start, rev_end, years_count) if rev_start>0 else None
        cagr_eps = self.compute_cagr(eps_start, eps_end, years_count) if eps_start>0 else None
        return {
            "years": years_count,
            "cagr_revenue": cagr_rev,
            "cagr_eps": cagr_eps,
            "rev_start": rev_start,
            "rev_end": rev_end,
            "eps_start": eps_start,
            "eps_end": eps_end
        }

    def fetch_yahoo(self, ticker):
        """Extract annual revenue and EPS when Yahoo Finance provides both."""
        try:
            import yfinance as yf
        except ImportError as error:
            raise RuntimeError("Yahoo financials require yfinance. Install project requirements first.") from error

        statements = yf.Ticker(ticker).financials
        if statements.empty or "Total Revenue" not in statements.index:
            return {"note": "Annual revenue data is unavailable from the selected provider"}
        eps_row = next((name for name in ("Diluted EPS", "Basic EPS") if name in statements.index), None)
        if eps_row is None:
            return {"note": "Annual EPS data is unavailable from the selected provider"}

        yearly = {}
        for column in statements.columns:
            revenue, eps = statements.loc["Total Revenue", column], statements.loc[eps_row, column]
            if revenue is None or eps is None:
                continue
            try:
                revenue, eps = float(revenue), float(eps)
            except (TypeError, ValueError):
                continue
            if revenue > 0:
                yearly[column.year] = {"revenue": revenue, "eps": eps}
        if len(yearly) < 2:
            return {"note": "Fewer than two complete annual revenue and EPS records were available"}
        return self.summarize_growth(yearly)
