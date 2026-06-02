import re
import pandas as pd
from sklearn.model_selection import train_test_split

class NepaliTextNormalizer:
    """
    Handles cleaning and standardization of Devanagari and Romanized Nepali-English text.
    """
    def __init__(self):
        # Base characters and punctuation removal patterns
        self.punctuation_pattern = re.compile(r'[^\w\s\u0900-\u097F]') # Retain word characters, spaces, and Devanagari Unicode
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        
    def clean_urls_and_mentions(self, text):
        """Removes social media mentions, handles, and web URLs."""
        if not isinstance(text, str):
            return ""
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        return text

    def normalize_devanagari(self, text):
        """
        Normalizes Devanagari script variations, half-letters, and spacing.
        """
        # Add Devanagari Unicode character standardization if needed
        # (e.g., standardizing ZWNJ, halant normalization, etc.)
        return text

    def clean_emoji_and_special(self, text):
        """Removes emojis and non-alphanumeric/non-Devanagari characters."""
        return self.punctuation_pattern.sub('', text)

    def normalize_text(self, text):
        """
        Full text cleaning pipeline for both scripts.
        """
        text = self.clean_urls_and_mentions(text)
        text = self.normalize_devanagari(text)
        text = self.clean_emoji_and_special(text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to lower-case for Romanized text
        return text.lower()


def load_and_preprocess_dataset(filepath, normalizer=None):
    """
    Loads raw CSV/Excel file, cleans text using normalizer, and structures dataset.
    """
    if normalizer is None:
        normalizer = NepaliTextNormalizer()
        
    print(f"[*] Loading raw dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # We expect 'CommentText' column from scrape_youtube.py
    text_col = 'CommentText' if 'CommentText' in df.columns else df.columns[0]
    
    print("[*] Normalizing reviews/comments...")
    df['CleanedText'] = df[text_col].apply(normalizer.normalize_text)
    
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """Splits processed data into train and validation sets."""
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"[+] Dataset split completed. Train size: {len(train_df)}, Validation size: {len(val_df)}")
    return train_df, val_df


if __name__ == "__main__":
    # Test pipeline with a dummy code-mixed sample
    normalizer = NepaliTextNormalizer()
    sample = "GadgetByte team ko video danger babal lagchha malai! 😍 Best camera quality specs ever! Visit https://gadgetbyte.com @gadgetbytenepal"
    print("Original Text:", sample)
    print("Cleaned Text :", normalizer.normalize_text(sample))
