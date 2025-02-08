import os
from dotenv import load_dotenv
import requests
from textblob import TextBlob

# Load environment variables
load_dotenv()

# Function to fetch news articles
def fetch_news():
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            raise ValueError("API key not found in .env file")
            
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.json()['articles']
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")
    return None

# Function to analyze sentiment
def analyze_sentiment(text):
    if not text:
        return "Neutral 😐 (No text)"
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.2:
        return "Positive 😊"
    elif polarity < -0.2:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# Main function to display news and sentiment
if __name__ == "__main__":
    print("Fetching news...\n")
    articles = fetch_news()
    
    if articles:
        for idx, article in enumerate(articles[:5], 1):  # Show first 5 articles
            title = article['title']
            description = article['description'] or "No description"
            
            # Analyze sentiment for title and description
            title_sentiment = analyze_sentiment(title)
            desc_sentiment = analyze_sentiment(description)
            
            # Display results
            print(f"\n{idx}. {title}")
            print(f"   Title Sentiment: {title_sentiment}")
            print(f"   Description Sentiment: {desc_sentiment}")
            print("   " + "-" * 30)
    else:
        print("No articles found or error fetching news.")