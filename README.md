# Aspect-Based Sentiment Analysis (ABSA) on Code-Mixed Nepali-English Product Reviews

A production-grade, research-oriented natural language processing (NLP) project focused on identifying explicit and implicit product aspects and their associated sentiment polarities from code-mixed Nepali-English social media comments and reviews.

---

## 🚀 Project Overview

In the Nepalese digital landscape, users communicating on e-commerce (e.g., Daraz) and social media platforms (e.g., YouTube, Facebook, TikTok) rarely write in standard English or pure Devanagari script. Instead, they predominantly use **Code-Mixed Nepali-English** written in either Devanagari or Romanized (transliterated) script.

This project addresses this low-resource NLP challenge by establishing:
1. **Automated Scraping Pipelines:** Gathering real-world code-mixed reviews from tech channels (without requiring Google API Keys).
2. **Specialized Text Normalization:** Cleansing Devanagari unicode structures and standardizing colloquial Romanized Nepali tokens.
3. **Multi-Task Deep Learning:** Fine-tuning transformer models (e.g., XLM-RoBERTa) to simultaneously classify **Aspect Category** (e.g., *Camera, Battery, Price*) and **Sentiment Polarity** (e.g., *Positive, Negative, Neutral*).
4. **Cloud-Ready Deployment:** Serving the model via a **FastAPI** backend and showcasing it with a beautiful, interactive **Streamlit** dashboard.

---

## 📂 Project Architecture

```text
sentiment-analysis-dl/
│
├── data/
│   ├── raw/                 # Raw comment exports (ignored by Git)
│   └── processed/           # Normalised and annotated datasets
│
├── notebooks/               # EDA & Model Prototyping
│
├── scripts/
│   └── scrape_youtube.py    # Zero-configuration YouTube scraper
│
├── src/                     # Production Package Source Code
│   ├── __init__.py
│   ├── data_pipeline.py     # Cleansing and processing logic
│   ├── tokenizer.py         # Subword tokenization wrappers
│   ├── model.py             # Custom deep learning/transformer architectures
│   ├── train.py             # Model training with WandB/MLflow logs
│   └── evaluate.py          # Metric calculations & benchmarking
│
├── api/                     # High-performance REST API
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   └── schemas.py           # Pydantic schemas
│
├── dashboard/               # Frontend Dashboard
│   └── app.py               # Streamlit application
│
├── requirements.txt         # Package dependencies
└── .gitignore               # Clean repository filter
```

---

## 🛠️ Getting Started

### 1. Prerequisite: Local Virtual Environment
Create and activate your Python virtual environment:

```powershell
# In PowerShell:
python -m venv venv
.\venv\Scripts\Activate

# In Command Prompt (CMD):
python -m venv venv
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Collection (Scraping)
To scrape real-world code-mixed comments from popular Nepali tech YouTube videos (e.g., reviews of smartphones, laptops, or gadgets), run:
```bash
python scripts/scrape_youtube.py
```
This automatically parses popular video comments and saves them directly to `data/raw/raw_comments.csv` without requiring any API keys.

---

## 📈 Roadmap & Core Tasks
- [x] Establish directory structure and modular package bindings.
- [x] Deploy zero-config YouTube comments scraper.
- [ ] Build specialized Nepali/Romanized text pre-processing and cleaning engine.
- [ ] Design and implement semi-automated data annotator for ABSA categories.
- [ ] Implement baseline machine learning (TF-IDF + SVM).
- [ ] Fine-tune state-of-the-art Transformer models (XLM-RoBERTa / DeBERTa).
- [ ] Implement FastAPI serving endpoints and deploy Streamlit analytics interface.
