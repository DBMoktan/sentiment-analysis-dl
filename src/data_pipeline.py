import os
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

class NepaliTextNormalizer:
    """
    Handles cleaning and standardization of Devanagari and Romanized Nepali-English text.
    Removes web links, social handles, emojis, and special punctuation, while standardizing
    unicode representation and whitespaces.
    """
    def __init__(self):
        # Regular expressions for URL, handles, and basic noise
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        # Character whitelist: English alphabet, standard numbers, Devanagari block, Devanagari numbers, and whitespace
        self.clean_char_pattern = re.compile(r'[^a-zA-Z0-9\s\u0900-\u097F\u0966-\u096F]')
        # Spaces collapse pattern
        self.spaces_pattern = re.compile(r'\s+')
        # ZWJ/ZWNJ unicode characters pattern
        self.zwnj_zwj_pattern = re.compile(r'[\u200c\u200d]')

    def clean_urls_and_mentions(self, text: str) -> str:
        """Removes social media mentions, handles, and web URLs."""
        if not isinstance(text, str):
            return ""
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        return text

    def normalize_unicode(self, text: str) -> str:
        """Standardizes Devanagari unicode sequences and strips non-printable control chars."""
        # Normalize characters (canonical composition)
        text = unicodedata.normalize('NFKC', text)
        # Strip Zero Width Non-Joiner (ZWNJ) and Zero Width Joiner (ZWJ)
        text = self.zwnj_zwj_pattern.sub('', text)
        return text

    def clean_noise_and_punctuation(self, text: str) -> str:
        """Strips emojis, special characters, and non-whitelisted punctuation."""
        return self.clean_char_pattern.sub(' ', text)

    def normalize_text(self, text: str) -> str:
        """
        Executes the full text cleaning pipeline.
        Suitable for both Devanagari and Romanized text.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Clear links and handles
        text = self.clean_urls_and_mentions(text)
        
        # 2. Normalize Devanagari and clean control characters
        text = self.normalize_unicode(text)
        
        # 3. Strip emojis, special characters, and standard punctuation
        text = self.clean_noise_and_punctuation(text)
        
        # 4. Collapse spaces and convert to lowercase (lowercases Romanized characters)
        text = self.spaces_pattern.sub(' ', text).strip()
        text = text.lower()
        
        return text

def load_and_preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Loads raw CSV, cleans comments using NepaliTextNormalizer, and outputs a processed CSV.
    """
    print(f"[*] Loading raw dataset from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # Identify comment text column
    text_col = 'CommentText' if 'CommentText' in df.columns else df.columns[2]
    print(f"[*] Preprocessing text column: '{text_col}'")
    
    normalizer = NepaliTextNormalizer()
    print("[*] Running cleaning and normalization pipeline...")
    df['CleanedText'] = df[text_col].apply(normalizer.normalize_text)
    
    # Filter out empty comments after cleaning
    initial_count = len(df)
    df = df[df['CleanedText'].str.strip() != '']
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"[!] Removed {removed_count} comments that became empty after cleaning.")
        
    # Save the processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[+] Cleaned dataset saved to: {output_path} | Count: {len(df)}")
    
    return df
