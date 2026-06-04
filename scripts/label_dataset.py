import os
import pandas as pd

# Define Aspect labels and mapping to integers
ASPECT_MAP = {
    "Performance": 0,
    "Camera": 1,
    "Price": 2,
    "Battery": 3,
    "Design": 4,
    "Service": 5
}

# Define Sentiment mapping to integers
SENTIMENT_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

# Keyword definitions for Aspects
ASPECT_KEYWORDS = {
    "Performance": [
        "performance", "speed", "processor", "ram", "lag", "hang", "slow", "fast", 
        "gaming", "game", "chip", "smooth", "freeze", "running", "specs", "cpu", "gpu", 
        "snapdragon", "bionic", "dimensity", "hz", "120hz", "pubg", "ping", "latency",
        "ल्याग", "ह्याङ", "गेम", "स्लो", "फास्ट", "स्पिड", "स्पीड", "गेमिंग", "प्रोसिसर", 
        "चल्छ", "चलिरहेको", "चालु", "chalxa", "chaldaina", "chalne", "smooth", "load", "loading"
    ],
    "Camera": [
        "camera", "photo", "video", "picture", "pic", "focus", "lens", "zoom", "ultrawide", 
        "megapixel", "mp", "sensor", "selfie", "stabilization", "ois", "eis", "cinematic", 
        "record", "fps", "क्यामेरा", "भिडियो", "फोटो", "जुम", "लेन्स", "खिच्न", "सेन्सर",
        "khichda", "khichna", "selfie", "lens", "focus"
    ],
    "Price": [
        "price", "daam", "cost", "rate", "value", "expensive", "cheap", "budget", 
        "overpriced", "worth", "buy", "purchase", "sale", "discount", "cash", "tax", 
        "pay", "paid", "worth", "vfm", "मूल्य", "पैसा", "सस्तो", "महँगो", "किन्नु", "किने", 
        "किनेको", "दर", "भाउ", "तिर्न", "tirna", "sasto", "mahango", "kinne", "kineko", 
        "paisa", "kinna", "halera", "thapera", "lakh"
    ],
    "Battery": [
        "battery", "charge", "charging", "mah", "backup", "drain", "sot", "adapter", 
        "charger", "watt", "w", "ब्याट्री", "चार्ज", "चार्जर", "ब्याकअप", "drain", "backup"
    ],
    "Design": [
        "design", "look", "display", "screen", "color", "size", "heavy", "light", 
        "notch", "island", "oled", "amoled", "glass", "build", "weight", "thin", "thick", 
        "appearance", "डिजाईन", "रंग", "स्क्रिन", "डिस्प्ले", "डिजाइन", "लुक्स", "कालो", 
        "सेतो", "निलो", "halka", "garo", "thulo", "sano", "color", "screen", "display"
    ],
    "Service": [
        "service", "delivery", "shipping", "driver", "ride", "pathao", "indrive", "yango", 
        "net", "wifi", "internet", "router", "connection", "customer", "support", "fiber", 
        "order", "parcel", "tracking", "shopping", "daraz", "isp", "worldlink", "cgnet",
        "सेवा", "ड्राइवर", "पठाओ", "डेलिभरी", "नेट", "वाईफाई", "कनेक्सन", "अर्डर", 
        "सामान", "wifi", "internet", "router", "daraz", "pathao", "indrive", "yango", 
        "delivery", "order"
    ]
}

# Keyword definitions for Sentiments
SENTIMENT_KEYWORDS = {
    "Positive": [
        "good", "best", "nice", "love", "great", "excellent", "perfect", "awesome", 
        "smooth", "friendly", "satisfied", "worthy", "wow", "lovely", "top", "legend",
        "राम्रो", "दामी", "उत्कृष्ट", "धन्यवाद", "बबाल", "मज्जा", "राम्ररी", "सुन्दर", 
        "खुसी", "बधाई", "सही", "ठिक", "ramro", "dami", "babal", "best", "nice", "khatra", 
        "sahi", "thik", "badhiya", "lovely", "satisfy", "satisfied"
    ],
    "Negative": [
        "bad", "worst", "waste", "garbage", "disappoint", "disappointing", "slow", 
        "lag", "freeze", "overpriced", "fail", "regret", "downfall", "trash", "useless", 
        "hate", "horrible", "poor", "scam", "error", "issue", "buffer", "naramro",
        "नराम्रो", "बेकार", "झुट", "हावा", "झुर", "ढिलो", "समस्या", "दुख", "खत्तम", 
        "घटिया", "रिस", "गाह्रो", "घाटा", "naramro", "bekar", "jhuto", "kharab", "khattam", 
        "useless", "scam"
    ]
}

def annotate_text(cleaned_text: str, comment_text: str, category: str):
    """
    Determines the aspect category and sentiment polarity of a comment using keywords.
    Falls back to category domains if aspect is ambiguous.
    """
    cleaned_lower = str(cleaned_text).lower()
    raw_lower = str(comment_text).lower()
    
    # Combined text for checking matches
    search_text = f"{cleaned_lower} {raw_lower}"
    
    # 1. Aspect Category Classification
    aspect_scores = {aspect: 0 for aspect in ASPECT_KEYWORDS.keys()}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in search_text:
                aspect_scores[aspect] += 1
                
    max_aspect_score = max(aspect_scores.values())
    if max_aspect_score > 0:
        # Get aspects with the maximum score
        best_aspects = [a for a, s in aspect_scores.items() if s == max_aspect_score]
        # Resolve tie by picking first
        assigned_aspect = best_aspects[0]
    else:
        # Fallback to category metadata
        if category in ["ISP", "Ride-Sharing", "E-Commerce"]:
            assigned_aspect = "Service"
        elif category == "Tech":
            assigned_aspect = "Performance"  # default for tech
        else:
            assigned_aspect = "Performance"
            
    # 2. Sentiment Classification
    pos_score = 0
    neg_score = 0
    
    for kw in SENTIMENT_KEYWORDS["Positive"]:
        if kw in search_text:
            pos_score += 1
            
    for kw in SENTIMENT_KEYWORDS["Negative"]:
        if kw in search_text:
            neg_score += 1
            
    if pos_score > neg_score:
        assigned_sentiment = "Positive"
    elif neg_score > pos_score:
        assigned_sentiment = "Negative"
    else:
        assigned_sentiment = "Neutral"
        
    return ASPECT_MAP[assigned_aspect], SENTIMENT_MAP[assigned_sentiment]

def process_and_save_dataset(file_path: str):
    """Loads a dataset split, annotates each comment, and saves it."""
    print(f"[*] Annotating dataset: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset split not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    aspects = []
    sentiments = []
    
    for _, row in df.iterrows():
        cleaned = row.get("CleanedText", "")
        raw = row.get("CommentText", "")
        cat = row.get("Category", "")
        
        aspect_label, sentiment_label = annotate_text(cleaned, raw, cat)
        aspects.append(aspect_label)
        sentiments.append(sentiment_label)
        
    df["Aspect"] = aspects
    df["Sentiment"] = sentiments
    
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"[+] Saved annotated dataset to {file_path} | Count: {len(df)}")
    
    # Print label distribution
    print("\n--- Label Distributions ---")
    print("Aspect classes:")
    print(df["Aspect"].value_counts().to_dict())
    print("Sentiment classes:")
    print(df["Sentiment"].value_counts().to_dict())
    print("---------------------------\n")

def main():
    processed_dir = "data/processed"
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    processed_comments_path = os.path.join(processed_dir, "processed_comments.csv")
    
    # Process all splits
    for path in [train_path, val_path, processed_comments_path]:
        if os.path.exists(path):
            process_and_save_dataset(path)
            
if __name__ == "__main__":
    main()
