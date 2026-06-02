import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskNepaliABSA(nn.Module):
    """
    Multi-Task Deep Learning model for Aspect-Based Sentiment Analysis.
    Shares a pre-trained multilingual transformer encoder backbone, and features 
    separate dense output classification heads for:
      1. Aspect Category Classification (Multi-class / Multi-label)
      2. Sentiment Polarity Classification (Multi-class: Positive, Negative, Neutral)
    """
    def __init__(self, model_name="xlm-roberta-base", num_aspects=6, num_sentiments=3, dropout_rate=0.2):
        super(MultiTaskNepaliABSA, self).__init__()
        print(f"[*] Building Multi-Task model with backbone: {model_name}")
        
        # Shared Transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification Head 1: Aspect Classifier
        self.aspect_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_aspects)
        )
        
        # Classification Head 2: Sentiment Classifier
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_sentiments)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass. Returns dict containing raw logits for both aspects and sentiment outputs.
        """
        # Encode inputs using shared backbone
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract pooler output or mean pool of sequence tokens
        # For XLM-RoBERTa, we can use the CLS token representation (first token in sequence)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        
        cls_representation = self.dropout(cls_representation)
        
        # Generate prediction logits for both heads
        aspect_logits = self.aspect_classifier(cls_representation)
        sentiment_logits = self.sentiment_classifier(cls_representation)
        
        return {
            "aspect_logits": aspect_logits,
            "sentiment_logits": sentiment_logits
        }

if __name__ == "__main__":
    # Test model structure with random input tensors
    model = MultiTaskNepaliABSA(num_aspects=6, num_sentiments=3)
    
    # Dummy input representing batch of 2 sentences, 10 tokens each
    dummy_input_ids = torch.randint(low=1, high=1000, size=(2, 10))
    dummy_attention_mask = torch.ones(size=(2, 10), dtype=torch.long)
    
    print("[*] Running dummy forward pass...")
    predictions = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    
    print("Aspect Logits Shape    :", predictions["aspect_logits"].shape) # Expected: [batch_size, num_aspects]
    print("Sentiment Logits Shape :", predictions["sentiment_logits"].shape) # Expected: [batch_size, num_sentiments]
    print("[+] Model architecture initialized and verified successfully!")
