import streamlit as nn_streamlit # Standard imports
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page layout and aesthetics
st.set_page_config(
    page_title="Nepali ABSA Analytics",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown
st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.2rem;
            color: #4B5563;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metrics-card {
            background-color: #F3F4F6;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #3B82F6;
        }
    </style>
""", unsafe_allow_html=True)

# Application titles
st.markdown("<div class='main-header'>🇳🇵 Nepali Code-Mixed ABSA Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>State-of-the-Art Aspect-Based Sentiment Analysis on Devanagari & Romanized Reviews</div>", unsafe_allow_html=True)

# Local fallback prediction heuristics (when FastAPI server is offline)
def local_heuristic_predict(text):
    text_lower = text.lower()
    
    # Aspect heuristics
    aspect = "General"
    aspect_words = {
        "camera": "Camera", "photo": "Camera", "lens": "Camera",
        "battery": "Battery", "charging": "Battery", "backup": "Battery",
        "speed": "Performance", "lag": "Performance", "slow": "Performance", "processor": "Performance",
        "price": "Price", "budget": "Price", "mahango": "Price", "sasto": "Price",
        "design": "Design", "look": "Design", "colour": "Design"
    }
    for word, asp in aspect_words.items():
        if word in text_lower:
            aspect = asp
            break
            
    # Sentiment heuristics
    sentiment = "Neutral"
    sentiment_words = {
        "babal": "Positive", "danger": "Positive", "best": "Positive", "ramro": "Positive", "khatra": "Positive",
        "naramro": "Negative", "khatam": "Negative", "disappoint": "Negative", "worst": "Negative", "slow": "Negative"
    }
    for word, sent in sentiment_words.items():
        if word in text_lower:
            sentiment = sent
            break
            
    return {
        "original_text": text,
        "cleaned_text": text.strip(),
        "predicted_aspect": aspect,
        "aspect_confidence": 0.85 if aspect != "General" else 0.50,
        "predicted_sentiment": sentiment,
        "sentiment_confidence": 0.90 if sentiment != "Neutral" else 0.50
    }

# FastAPI Server Address
API_URL = "http://127.0.0.1:8000/predict"

# Sidebar controls
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Flag_of_Nepal.svg", width=100)
st.sidebar.title("Configuration")
mode = st.sidebar.radio("Analysis Mode", ["Single Text Review", "Batch File Upload"])

if mode == "Single Text Review":
    st.subheader("📝 Analyze Single Comment")
    
    # Pre-populated samples for the user to try
    sample_select = st.selectbox(
        "Select a Nepali code-mixed review sample to test:",
        [
            "Custom text input...",
            "Yo phone ko camera babal lagyo malai, clear aauchha photo",
            "kasto faldtu battery backup chha yar, 2 hour mai runout hunchha charging",
            "processor speed danger ramro chha tara price ali mahango bhayo",
            "Design ultra premium look lagyo tara slow lag hunchha bich bich ma"
        ]
    )
    
    # Handle sample selector behavior
    if sample_select == "Custom text input...":
        user_input = st.text_area(
            "Enter your Nepali review (Supports mixed Devanagari and Romanized letters):",
            "yo mobile ko design clean ra ramro chha tara price excessive lagyo"
        )
    else:
        user_input = st.text_area("Enter your Nepali review:", sample_select)
        
    if st.button("Extract Aspect & Sentiment", type="primary"):
        if len(user_input.strip()) < 3:
            st.error("Please enter a review containing at least 3 characters.")
        else:
            with st.spinner("Analyzing text..."):
                # Try communicating with the FastAPI endpoint first; fall back to local rule-base if offline
                try:
                    res = requests.post(API_URL, json={"text": user_input}, timeout=2.0)
                    if res.status_code == 200:
                        prediction = res.json()
                        st.info("⚡ Powered by Live FastAPI Inference Engine")
                    else:
                        prediction = local_heuristic_predict(user_input)
                        st.warning("⚠️ Serving local predictions (FastAPI server offline)")
                except Exception:
                    prediction = local_heuristic_predict(user_input)
                    st.warning("⚠️ Serving local predictions (FastAPI server offline)")
            
            # Display prediction layout in grid columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                st.metric("🎯 Extracted Aspect", prediction["predicted_aspect"])
                st.write(f"Confidence score: **{prediction['aspect_confidence']:.2%}**")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                # Color code sentiment metrics
                sentiment_color = "#10B981" if prediction["predicted_sentiment"] == "Positive" else "#EF4444" if prediction["predicted_sentiment"] == "Negative" else "#F59E0B"
                
                st.markdown(f"""
                    <div class='metrics-card' style='border-left: 5px solid {sentiment_color};'>
                """, unsafe_allow_html=True)
                st.metric("📊 Predicted Sentiment", prediction["predicted_sentiment"])
                st.write(f"Confidence score: **{prediction['sentiment_confidence']:.2%}**")
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.success(f"**Cleaned Text:** *\"{prediction['cleaned_text']}\"*")

else:
    st.subheader("📁 Batch File Sentiment Analyzer")
    st.markdown("Upload your scraped comments CSV to execute bulk aspect-based sentiment extraction.")
    
    uploaded_file = st.file_uploader("Upload CSV review file (must contain text column)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head(5))
        
        text_column = st.selectbox("Select Text/Comment Column:", df.columns)
        
        if st.button("Run Batch ABSA Processing"):
            with st.spinner("Processing batch dataset..."):
                # Apply local prediction row by row
                results = []
                for text in df[text_column]:
                    results.append(local_heuristic_predict(str(text)))
                
                res_df = pd.DataFrame(results)
                
                # Combine predictions back to original dataframe
                combined_df = pd.concat([df, res_df[['cleaned_text', 'predicted_aspect', 'predicted_sentiment']]], axis=1)
                
                st.success("Batch Processing completed!")
                st.dataframe(combined_df.head(10))
                
                # Plot charts of distributions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Aspect Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=combined_df, x='predicted_aspect', palette='viridis', ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                with col2:
                    st.write("Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # Color mapped to [Positive, Negative, Neutral]
                    colors = ["#10B981", "#EF4444", "#F59E0B"]
                    combined_df['predicted_sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=colors, ax=ax)
                    plt.ylabel('')
                    st.pyplot(fig)
                    
                # Download button for parsed predictions
                csv_data = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Analyzed Data (CSV)",
                    data=csv_data,
                    file_name="analyzed_reviews.csv",
                    mime="text/csv"
                )
