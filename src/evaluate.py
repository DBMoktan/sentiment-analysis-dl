from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
import numpy as np

def calculate_absa_metrics(aspect_trues, aspect_preds, sentiment_trues, sentiment_preds, aspect_names=None, sentiment_names=None):
    """
    Computes precision, recall, macro-F1, and accuracy for both aspect and sentiment classification.
    Returns a structured dictionary containing all metric metrics.
    """
    if aspect_names is None:
        aspect_names = ["Camera", "Battery", "Performance", "Price", "Design", "General"]
    if sentiment_names is None:
        sentiment_names = ["Positive", "Negative", "Neutral"]
        
    print("=================== ASPECT EVALUATION REPORT ===================")
    aspect_report = classification_report(
        aspect_trues, 
        aspect_preds, 
        target_names=aspect_names[:len(set(aspect_trues).union(set(aspect_preds)))],
        output_dict=True
    )
    print(classification_report(
        aspect_trues, 
        aspect_preds, 
        target_names=aspect_names[:len(set(aspect_trues).union(set(aspect_preds)))]
    ))
    
    print("================== SENTIMENT EVALUATION REPORT ==================")
    sentiment_report = classification_report(
        sentiment_trues, 
        sentiment_preds, 
        target_names=sentiment_names[:len(set(sentiment_trues).union(set(sentiment_preds)))],
        output_dict=True
    )
    print(classification_report(
        sentiment_trues, 
        sentiment_preds, 
        target_names=sentiment_names[:len(set(sentiment_trues).union(set(sentiment_preds)))]
    ))
    
    # Calculate macro-f1 and accuracy metrics
    aspect_f1 = f1_score(aspect_trues, aspect_preds, average='macro')
    aspect_acc = accuracy_score(aspect_trues, aspect_preds)
    
    sentiment_f1 = f1_score(sentiment_trues, sentiment_preds, average='macro')
    sentiment_acc = accuracy_score(sentiment_trues, sentiment_preds)
    
    return {
        "aspect": {
            "f1_macro": aspect_f1,
            "accuracy": aspect_acc,
            "report": aspect_report,
            "confusion_matrix": confusion_matrix(aspect_trues, aspect_preds).tolist()
        },
        "sentiment": {
            "f1_macro": sentiment_f1,
            "accuracy": sentiment_acc,
            "report": sentiment_report,
            "confusion_matrix": confusion_matrix(sentiment_trues, sentiment_preds).tolist()
        }
    }

if __name__ == "__main__":
    # Test metrics module with mock predictions
    mock_aspect_trues = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mock_aspect_preds = [0, 1, 2, 0, 2, 2, 0, 1, 0] # Some errors
    
    mock_sentiment_trues = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mock_sentiment_preds = [0, 1, 2, 0, 1, 1, 0, 1, 2] # Some errors
    
    metrics = calculate_absa_metrics(
        mock_aspect_trues, 
        mock_aspect_preds, 
        mock_sentiment_trues, 
        mock_sentiment_preds
    )
    
    print("\nSummary Results:")
    print(f"Aspect Macro F1     : {metrics['aspect']['f1_macro']:.4f}")
    print(f"Sentiment Macro F1  : {metrics['sentiment']['f1_macro']:.4f}")
