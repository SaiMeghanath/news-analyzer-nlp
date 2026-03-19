"""
app.py
------
Flask web interface for the AI-Powered News Analyzer.

Routes:
- GET / : Homepage with analysis form
- GET /analyze : Fetch & analyze news, render results table
- GET /api/news : JSON API endpoint (programmatic access)
"""

from flask import Flask, render_template, request, jsonify
from news_analyzer import fetch_news, analyze_articles

app = Flask(__name__)

@app.route("/")
def index():
    """
    Homepage: renders the search/filter form.
    """
    return render_template("index.html", results=None, error=None)

@app.route("/analyze")
def analyze():
    """
    Fetch and analyze news articles.

    Query params:
        country: str (ISO country code, default "us")
        category: str (news category, default None = "all")
        count: int (number of articles, default 10)
    """
    country = request.args.get("country", "us")
    category = request.args.get("category", None) or None

    try:
        count = int(request.args.get("count", 10))
        count = max(1, min(count, 20))  # clamp between 1 and 20
    except ValueError:
        count = 10

    articles = fetch_news(country=country, category=category, page_size=count)

    if not articles:
        return render_template(
            "index.html",
            results=None,
            error="No articles found. Check your API key or try different filters.",
        )

    results = analyze_articles(articles)

    return render_template(
        "index.html",
        results=results,
        country=country,
        category=category,
        count=count,
    )

@app.route("/api/news")
def api_news():
    """
    JSON API endpoint.
    Returns structured analysis results for programmatic access.
    """
    country = request.args.get("country", "us")
    category = request.args.get("category", None) or None

    try:
        count = int(request.args.get("count", 10))
        count = max(1, min(count, 20))
    except ValueError:
        count = 10

    articles = fetch_news(country=country, category=category, page_size=count)

    if not articles:
        return jsonify({"error": "No articles found"}), 404

    results = analyze_articles(articles)

    return jsonify({
        "count": len(results),
        "articles": results,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
