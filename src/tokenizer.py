from transformers import AutoTokenizer
import torch

class NepaliCodeMixedTokenizer:
    """
    Subword tokenizer wrapper around pre-trained transformer 
    tokenizers (e.g., xlm-roberta-base) for Romanized and Devanagari script.
    """
    def __init__(self, model_name: str = "xlm-roberta-base"):
        print(f"[*] Loading pre-trained tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str):
        """Converts raw string to list of subword tokens."""
        if not isinstance(text, str):
            return []
        return self.tokenizer.tokenize(text)
        
    def encode(self, text: str, max_length: int = 128):
        """Encodes text into PyTorch tensors suitable for model training."""
        if not isinstance(text, str):
            text = ""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

