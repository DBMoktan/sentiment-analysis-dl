# Aspect-Based Sentiment Analysis (ABSA) on Code-Mixed Nepali-English Product Reviews

A production-grade, research-oriented natural language processing (NLP) framework designed to identify explicit and implicit product aspects and determine their associated sentiment polarities from code-mixed Nepali-English social media comments and reviews.

---

## 🚀 Key Features

* **Zero-Configuration Scraper:** Collects real-world social media and YouTube comments without requiring Google API Keys or complex developer accounts.
* **Specialized Text Normalization:** Cleanses Devanagari unicode structures and standardizes colloquial Romanized (transliterated) Nepali tokens into a normalized form for machine learning pipelines.
* **Multi-Task Deep Learning Architecture:** Adapts state-of-the-art cross-lingual Transformer models (e.g., XLM-RoBERTa) to simultaneously classify Aspect Categories (e.g., *Camera, Battery, Price*) and Sentiment Polarities (e.g., *Positive, Negative, Neutral*).
* **High-Performance API Layer:** Served using a fast, lightweight FastAPI backend providing REST endpoints for single-text and batch inferences.
* **Interactive Analytics Dashboard:** Built with Streamlit to offer users real-time visualization of aspects, sentiment distributions, and batch CSV processing capabilities.

---

## 📂 Project Architecture

```text
sentiment-analysis-dl/
│
├── data/
│   ├── raw/                 # Raw comment exports (ignored by Git)
│   └── processed/           # Normalized and annotated datasets
│
├── notebooks/               # EDA & Model Prototyping
│
├── scripts/
│   └── scrape_youtube.py    # Zero-configuration YouTube scraper
│
├── src/                     # Core NLP Package
│   ├── __init__.py
│   ├── data_pipeline.py     # Cleansing and text normalization
│   ├── tokenizer.py         # Subword tokenization wrappers
│   ├── model.py             # Custom deep learning/transformer architectures
│   ├── train.py             # Model training with WandB/MLflow logs
│   └── evaluate.py          # Metric calculations & benchmarking
│
├── api/                     # Inference REST API
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic validation schemas
│
├── dashboard/               # Analytics Frontend
│   └── app.py               # Streamlit application
│
├── requirements.txt         # Package dependencies
└── .gitignore               # Clean repository filter
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Virtual Environment Setup
Clone this repository and set up a Python virtual environment:

```bash
# Initialize and activate the virtual environment
python -m venv venv

# For Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# For Windows (Command Prompt):
.\venv\Scripts\activate.bat

# For macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries including PyTorch, Transformers, FastAPI, and Streamlit:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Data Collection (YouTube Scraper)
To harvest code-mixed comments and product reviews directly from YouTube videos (e.g., device reviews and tech channels):
```bash
python scripts/scrape_youtube.py
```
This script exports the fetched reviews directly into `data/raw/raw_comments.csv` without requiring any external YouTube API credentials.

### 2. Start the Inference API
To serve the multi-task aspect and sentiment predictions over local REST endpoints:
```bash
python api/main.py
```
The server starts locally at `http://127.0.0.1:8000`. You can access the interactive API docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

### 3. Start the Interactive Dashboard
To launch the frontend dashboard for analyzing individual reviews or visualizing batch uploads:
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your browser (typically at `http://localhost:8501`).
