import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.model import MultiTaskNepaliABSA
from src.tokenizer import NepaliCodeMixedTokenizer
import numpy as np

class NepaliABSADataset(Dataset):
    """
    Custom Dataset class for loading, tokenizing, and wrapping Nepali ABSA texts
    along with their aspect and sentiment labels.
    """
    def __init__(self, texts, aspect_labels, sentiment_labels, tokenizer, max_length=128):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        inputs = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        # Squeeze out batch dimension from tokenizer returns
        item = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }
        
        # Cast labels to appropriate tensor types
        item['aspect_label'] = torch.tensor(self.aspect_labels[idx], dtype=torch.long)
        item['sentiment_label'] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        
        return item


def train_epoch(model, data_loader, optimizer, scheduler, device, aspect_loss_fn, sentiment_loss_fn):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        aspect_labels = batch['aspect_label'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Compute joint losses
        loss_aspect = aspect_loss_fn(outputs['aspect_logits'], aspect_labels)
        loss_sentiment = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)
        
        # Joint loss optimization (summing losses)
        joint_loss = loss_aspect + loss_sentiment
        
        # Backward pass & step
        joint_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += joint_loss.item()
        
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, device, aspect_loss_fn, sentiment_loss_fn):
    """Evaluates the model on validation dataset."""
    model.eval()
    total_loss = 0
    aspect_preds = []
    aspect_trues = []
    sentiment_preds = []
    sentiment_trues = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            loss_aspect = aspect_loss_fn(outputs['aspect_logits'], aspect_labels)
            loss_sentiment = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)
            joint_loss = loss_aspect + loss_sentiment
            total_loss += joint_loss.item()
            
            # Record predictions for metric scoring
            aspect_preds.extend(torch.argmax(outputs['aspect_logits'], dim=1).cpu().numpy())
            aspect_trues.extend(aspect_labels.cpu().numpy())
            
            sentiment_preds.extend(torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy())
            sentiment_trues.extend(sentiment_labels.cpu().numpy())
            
    val_loss = total_loss / len(data_loader)
    
    # Calculate basic accuracy
    aspect_acc = np.mean(np.array(aspect_preds) == np.array(aspect_trues))
    sentiment_acc = np.mean(np.array(sentiment_preds) == np.array(sentiment_trues))
    
    return val_loss, aspect_acc, sentiment_acc


def run_training_pipeline():
    """
    Main training execution function.
    Coordinates tokenizer initialization, dataset splitting, loader configuration, 
    model creation, and training loop.
    """
    print("[*] Training pipeline initiated...")
    # This is a template configuration. Actual data will be parsed in notebooks or main runs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    
    # Placeholder for actual data structures
    dummy_texts = ["yo device ekdamai babal chha", "kasto faldtu quality ko battery backup"]
    dummy_aspects = [1, 2] # 1: Performance, 2: Battery
    dummy_sentiments = [0, 1] # 0: Positive, 1: Negative
    
    tokenizer = NepaliCodeMixedTokenizer()
    
    dataset = NepaliABSADataset(dummy_texts, dummy_aspects, dummy_sentiments, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = MultiTaskNepaliABSA(num_aspects=6, num_sentiments=3)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=10)
    
    aspect_loss = nn.CrossEntropyLoss()
    sentiment_loss = nn.CrossEntropyLoss()
    
    print("[*] Running prototype training epoch...")
    loss = train_epoch(model, loader, optimizer, scheduler, device, aspect_loss, sentiment_loss)
    print(f"[+] Training epoch loss: {loss:.4f}")
    
    val_loss, aspect_acc, sentiment_acc = evaluate_model(model, loader, device, aspect_loss, sentiment_loss)
    print(f"[+] Validation metrics: Loss: {val_loss:.4f}, Aspect Acc: {aspect_acc:.2%}, Sentiment Acc: {sentiment_acc:.2%}")

if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as e:
        print("[!] Pipeline prototype run failed (HuggingFace models offline or device issue):", e)
