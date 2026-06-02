from fastapi import FastAPI

app = FastAPI(
    title="Nepali Code-Mixed ABSA Serving Layer",
    description="REST API for Aspect-Based Sentiment Analysis on Devanagari and Romanized Nepali-English text reviews.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to the Nepali Code-Mixed ABSA Serving API. Ready for step-by-step implementation."}
