"""

news_analyzer.py
----------------
Core pipeline for the AI-Powered News Analyzer.
Fetches top headlines via NewsAPI and runs dual sentiment analysis:
- TextBlob (lexicon/rule-based baseline)
- HuggingFace Transformers (contextual transformer model)

Returns structured results for each article showing both methods side by side,
making the comparison between approaches the central output of this pipeline.
"""

import os
from dotenv import load_dotenv
import requests
from textblob import TextBlob
from transformers import pipeline

load_dotenv()

# ============ NEWS FETCHING ============

def fetch_news(country: str = "us", category: str = None, page_size: int = 10) -> list[dict]:
    """
    Fetch top headlines from NewsAPI.
    
    Args:
        country: ISO 3166-1 alpha-2 country code (default: "us")
        category: Optional news category (business, tech, health, etc.)
        page_size: Number of articles to fetch (max: 100)
    
    Returns:
        List of article dicts, or empty list on failure.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY not found. Check your .env file.")

    params = {"country": country, "pageSize": page_size, "apiKey": api_key}
    if category:
        params["category"] = category

    try:
        response = requests.get("https://newsapi.org/v2/top-headlines",
                                params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [a for a in articles if a.get("title") and a["title"] != "Removed"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching news: {e}")
    except requests.exceptions.ConnectionError:
        print("Connection error. Check your internet connection.")
    except requests.exceptions.Timeout:
        print("Request timed out.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return []

# ============ SENTIMENT ANALYSIS ============

def textblob_sentiment(text: str) -> dict:
    print("News fetching module loaded.")
    """
    Lexicon-based sentiment via TextBlob.
    
    Returns:
        polarity: float (-1.0 negative to 1.0 positive)
        subjectivity: float (0.0 objective to 1.0 subjective)
        label: str ("Positive", "Negative", or "Neutral")
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "label": label,
    }

print("TextBlob sentiment module loaded.")

# Load transformer model once at import time
print("Loading transformer model...")
transformer_sentiment = pipeline("sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True, max_length=512)
print("Model loaded.")

def transformer_sentiment_score(text: str) -> dict:
    """
    Contextual sentiment via HuggingFace DistilBERT SST-2.
    
    Returns:
        label: str ("POSITIVE" or "NEGATIVE")
        score: float (confidence score 0.0 - 1.0)
    """
    result = transformer_sentiment(text[:512])[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4),
    }

print("Transformer sentiment module loaded.")

print("News fetching module loaded.")

# ============ ANALYSIS & OUTPUT ============

def analyze_articles(articles: list[dict]) -> list[dict]:
    """
    Run both sentiment methods on each article's title + description.
    
    Returns a list of structured result dicts per article.
    """
    results = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        source = article.get("source", {}).get("name", "Unknown")
        url = article.get("url", "")
        published_at = article.get("publishedAt", "")

        # Combine title + description for richer signal
        full_text = f"{title}. {description.strip()}" if description else title

        tb = textblob_sentiment(full_text)
        tr = transformer_sentiment_score(full_text)

        # Agreement flag: do both methods agree on positive/negative?
        tb_positive = tb["label"] == "Positive"
        tr_positive = tr["label"] == "POSITIVE"
        agreement = "Agree" if tb_positive == tr_positive else "Disagree"

        results.append({
            "title": title,
            "description": description,
            "source": source,
            "url": url,
            "published_at": published_at,
            "textblob": tb,
            "transformer": tr,
            "agreement": agreement,
        })
    return results

def print_results(results: list[dict]) -> None:
    """
    Pretty-print results to terminal.
    """
    for i, r in enumerate(results, 1):
        print(f"{'='*70}")
        print(f"{i}. {r['title']}")
        print(f"   Source: {r['source']} | {r['published_at']}")
        if r["description"]:
            print(f"   {r['description'][:120]}..." if len(r["description"]) > 120 else f"   {r['description']}")
        print()
        print(f"   TextBlob:   {r['textblob']['label']:<10} | polarity: {r['textblob']['polarity']:.3f}, "
              f"subjectivity: {r['textblob']['subjectivity']:.3f}")
        print(f"   Transformer: {r['transformer']['label']:<10} | confidence: {r['transformer']['score']:.3f}")
        print(f"   Methods: {r['agreement']}")
        print()

# ============ CLI ENTRY POINT ============

if __name__ == "__main__":
    print("Fetching news headlines...")
    articles = fetch_news(page_size=10)
    if not articles:
        print("No articles fetched. Check your API key and connection.")
    else:
        print(f"Fetched {len(articles)} articles. Running analysis...")
        results = analyze_articles(articles)
        print_results(results)
