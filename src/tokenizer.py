from transformers import AutoTokenizer

class NepaliCodeMixedTokenizer:
    """
    Wrapper around pre-trained transformer tokenizers (e.g., XLM-RoBERTa, DeBERTa, mBERT)
    tailored to parse Devanagari and Romanized Nepali-English text.
    """
    def __init__(self, model_name="xlm-roberta-base"):
        print(f"[*] Initializing pre-trained tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize(self, text):
        """Converts raw string to list of subword tokens."""
        return self.tokenizer.tokenize(text)
        
    def encode(self, text, max_length=128, padding="max_length", truncation=True):
        """Encodes text into inputs suitable for transformer model training."""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
    def decode(self, token_ids):
        """Decodes list of token IDs back into string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_vocab_size(self):
        """Returns tokenizer vocabulary size."""
        return len(self.tokenizer)

if __name__ == "__main__":
    # Test tokenizer on code-mixed sample
    # Note: Requires internet access to download tokenizer configs
    try:
        tokenizer = NepaliCodeMixedTokenizer("xlm-roberta-base")
        sample_text = "yo phone babal chha tara price mahango bhayo"
        tokens = tokenizer.tokenize(sample_text)
        print("Sample Text:", sample_text)
        print("Subword Tokens:", tokens)
    except Exception as e:
        print("[!] Tokenizer initialization failed (could be due to offline environment/missing config):", e)
