from pydantic import BaseModel, Field
from typing import List

class ReviewInferenceRequest(BaseModel):
    """
    Schema representing incoming text review for sentiment analysis.
    """
    text: str = Field(..., min_length=3, description="The raw code-mixed text comment or review to analyze.")

class ReviewInferenceResponse(BaseModel):
    """
    Schema representing the joint model predictions for a single review.
    """
    original_text: str
    cleaned_text: str
    predicted_aspect: str = Field(..., description="Extracted aspect category (e.g. Camera, Battery).")
    aspect_confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for aspect extraction.")
    predicted_sentiment: str = Field(..., description="Assigned sentiment polarity (Positive, Negative, Neutral).")
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for sentiment polarity.")

class BatchInferenceRequest(BaseModel):
    """
    Schema representing batch request.
    """
    reviews: List[ReviewInferenceRequest]

class BatchInferenceResponse(BaseModel):
    """
    Schema representing batch prediction response.
    """
    predictions: List[ReviewInferenceResponse]
