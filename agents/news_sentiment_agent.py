# agents/news_sentiment_agent.py
from typing import List, Dict
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

class NewsSentimentAgent:
    """
    Stub for fetching news and computing sentiment.
    Replace fetch_news() with real API calls when integrating.
    """

    POSITIVE_TERMS = {
        "beat", "beats", "growth", "profit", "profits", "strong", "surge", "surges",
        "upgrade", "buyback", "record", "rise", "rises", "gain", "gains", "expand",
    }
    NEGATIVE_TERMS = {
        "miss", "misses", "decline", "loss", "losses", "weak", "fall", "falls", "drop",
        "downgrade", "fraud", "lawsuit", "probe", "investigation", "resign", "warning",
    }

    def fetch_news(self, ticker, limit=5) -> List[Dict]:
        """Fetch recent matching headlines from Google News RSS; returns [] on feed/network errors."""
        url = "https://news.google.com/rss/search?q=" + quote(f'"{ticker}" stock') + "&hl=en-US&gl=US&ceid=US:en"
        request = Request(url, headers={"User-Agent": "GenProFit.ai/1.0"})
        try:
            with urlopen(request, timeout=8) as response:
                root = ET.fromstring(response.read())
        except (OSError, URLError, ET.ParseError, ValueError):
            return []
        headlines = []
        for item in root.findall("./channel/item")[:limit]:
            headlines.append({
                "title": item.findtext("title", default=""),
                "source": "Google News RSS",
                "date": item.findtext("pubDate"),
                "link": item.findtext("link"),
            })
        return headlines

    def sentiment_score(self, headlines) -> float:
        """Return a transparent lexicon score in [-1, 1], or neutral when no headlines exist."""
        scores = []
        for headline in headlines:
            words = set("".join(char.lower() if char.isalnum() else " " for char in headline["title"]).split())
            positive = len(words & self.POSITIVE_TERMS)
            negative = len(words & self.NEGATIVE_TERMS)
            total = positive + negative
            if total:
                scores.append((positive - negative) / total)
        return round(sum(scores) / len(scores), 3) if scores else 0.0
