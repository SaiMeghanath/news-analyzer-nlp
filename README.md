# AI-Powered News Analyzer

Real-time news summarization and sentiment analysis using NLP pipelines comparing lexicon-based and transformer-based approaches on news-domain text.

---

## Motivation

News articles are high-velocity, domain-specific text written in formal register but laden with implicit sentiment and editorial framing. Standard sentiment models trained on review corpora struggle here. This project builds a pipeline that fetches live headlines, runs dual-method sentiment analysis, and surfaces the results through a lightweight web interface making it easy to compare how different NLP approaches interpret the same news text.

---

## What It Does

- Fetches top headlines across categories via NewsAPI
- Runs TextBlob lexicon-rule-based sentiment scoring for baseline comparison
- Applies HuggingFace Transformer models for contextual sentiment classification
- Displays results with polarity scores and visual indicators through a Flask web UI
- Enables side-by-side comparison of lexicon vs. transformer output on identical inputs

---

## Architecture

NewsAPI Article Fetch --> Preprocessing --> TextBlob lexicon-based Polarity Score [-1, 1]
                                       --> HuggingFace Transformer Label + Confidence Score
                                        distilbert-base-uncased-finetuned-sst-2-english
                                       --> Flask Web UI Results Display


## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| NLP (lexicon) | TextBlob |
| NLP (transformer) | HuggingFace Transformers pipeline |
| News Data | NewsAPI |
| HTTP Client | requests, python-dotenv |
| Web Interface | Flask |
| Model | distilbert-base-uncased-finetuned-sst-2-english |

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/SaiMeghanath/news-analyzer-nlp.git
cd news-analyzer-nlp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the root directory:

```env
NEWS_API_KEY=your_news_api_key_here
```

Get a free key at [newsapi.org](https://newsapi.org).

### 4. Run

For the CLI pipeline:
```bash
python news_analyzer.py
```

For the web interface:
```bash
python app.py
```

---

## Method Comparison

| Method | Strength | Limitation |
|--------|----------|------------|
| TextBlob | Fast, interpretable, no GPU needed | Misses irony, sarcasm, domain context |
| DistilBERT | Context-aware, higher accuracy | Heavier, trained on SST-2 movie reviews |

The divergence between the two methods is most pronounced on geopolitical and economic headlines where sentiment is contextually encoded rather than lexically explicit. This gap motivates fine-tuning transformer models specifically on news corpora.

---

## Key Findings

- Fine-tune a transformer on news-domain sentiment datasets (e.g., Financial PhraseBank, SemEval News)
- Add multilingual support (Telugu/Hindi news headline analysis)
- Deploy as a HuggingFace Space for public demo
- Extend to named entity recognition (NER) for entity-level sentiment

---

## Future Directions

- Fine-tune a transformer on news-domain sentiment datasets
- Add multilingual support for Telugu/Hindi news headline analysis
- Deploy as a HuggingFace Space for public demo
- Extend to named entity recognition (NER) for entity-level sentiment

- ---

## Project Structure

```
news-analyzer-nlp/
├── news_analyzer.py    # Core pipeline: fetch, preprocess, analyze
├── app.py              # Flask web interface
├── .env                # API key (not committed)
├── requirements.txt
└── README.md
```

---

## Author

**Aladurthi Sai Meghanath**  
MCA, AI Specialization  
Amrita Vishwa Vidyapeetham  
- [LinkedIn](https://www.linkedin.com/in/sai-meghanath/)  
- [GitHub](https://github.com/SaiMeghanath)  
- saimeghanath052@gmail.com
