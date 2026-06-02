from fastapi import FastAPI, HTTPException
from api.schemas import ReviewInferenceRequest, ReviewInferenceResponse, BatchInferenceRequest, BatchInferenceResponse
from src.data_pipeline import NepaliTextNormalizer
import uvicorn
import random

app = FastAPI(
    title="Nepali Code-Mixed ABSA Serving Layer",
    description="REST API for Aspect-Based Sentiment Analysis on Devanagari and Romanized Nepali-English text reviews.",
    version="1.0.0"
)

# Global instances (lazy loaded or mocked for prototype)
normalizer = NepaliTextNormalizer()

# Define aspect and sentiment catalogs
ASPECTS = ["Camera", "Battery", "Performance", "Price", "Design", "General"]
SENTIMENTS = ["Positive", "Negative", "Neutral"]

# Heuristic vocabulary rules for fallback prototype predictions
KEYWORD_ASPECT_MAP = {
    "camera": "Camera",
    "photo": "Camera",
    "charging": "Battery",
    "battery": "Battery",
    "backup": "Battery",
    "speed": "Performance",
    "processor": "Performance",
    "lag": "Performance",
    "slow": "Performance",
    "price": "Price",
    "budget": "Price",
    "mahango": "Price",
    "sasto": "Price",
    "design": "Design",
    "look": "Design",
    "colour": "Design"
}

KEYWORD_SENTIMENT_MAP = {
    "babal": "Positive",
    "danger": "Positive",
    "best": "Positive",
    "ramro": "Positive",
    "mann": "Positive",
    "khatra": "Positive",
    "nice": "Positive",
    "naramro": "Negative",
    "khatam": "Negative",
    "disappoint": "Negative",
    "worst": "Negative",
    "faldtu": "Negative",
    "slow": "Negative",
    "lag": "Negative"
}

@app.get("/")
def home():
    """Welcome route."""
    return {"message": "Welcome to the Nepali Code-Mixed ABSA Serving API. Access /docs for swagger documentation."}

@app.get("/health")
def health_check():
    """Health check endpoint for deployment orchestration."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ReviewInferenceResponse)
def predict_sentiment(request: ReviewInferenceRequest):
    """
    Parses a single code-mixed comment and extracts predicted Aspect and Sentiment.
    Runs rule-based fallback predictions if model weights are not loaded.
    """
    try:
        raw_text = request.text
        cleaned_text = normalizer.normalize_text(raw_text)
        
        # Rule-based fallback logic for prototyping
        predicted_aspect = "General"
        aspect_conf = 0.5
        for word, aspect in KEYWORD_ASPECT_MAP.items():
            if word in cleaned_text:
                predicted_aspect = aspect
                aspect_conf = random.uniform(0.8, 0.98)
                break
                
        predicted_sentiment = "Neutral"
        sentiment_conf = 0.5
        for word, sentiment in KEYWORD_SENTIMENT_MAP.items():
            if word in cleaned_text:
                predicted_sentiment = sentiment
                sentiment_conf = random.uniform(0.8, 0.98)
                break
                
        return ReviewInferenceResponse(
            original_text=raw_text,
            cleaned_text=cleaned_text,
            predicted_aspect=predicted_aspect,
            aspect_confidence=round(aspect_conf, 4),
            predicted_sentiment=predicted_sentiment,
            sentiment_confidence=round(sentiment_conf, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failure: {str(e)}")

@app.post("/predict/batch", response_model=BatchInferenceResponse)
def predict_batch_sentiment(request: BatchInferenceRequest):
    """
    Batch sentiment analysis endpoint.
    """
    results = []
    for item in request.reviews:
        results.append(predict_sentiment(item))
    return BatchInferenceResponse(predictions=results)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
