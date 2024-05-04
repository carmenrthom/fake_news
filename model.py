# classifier.py
"""
BERT Binary Classifier
Author: Carmen Thom
Description: This script defines and trains a BERT-based binary classifier using the PyTorch library,
             training on the WELFake dataset for fake news detection, and saving the fine-tuned model.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def train_model(model_name, epochs=5, batch_size=32, max_length=64, learning_rate=5e-5, dropout_rate=0.01, workers=0):
    """Train a BERT model with specified parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        attention_probs_dropout_prob=dropout_rate,
        hidden_dropout_prob=dropout_rate,
        output_attentions=True
    ).to(device)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Data preparation
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/val.csv')
    train_texts = list(train_df["text"])
    train_labels = torch.tensor(list(train_df["label"]), dtype=torch.long)
    val_texts = list(val_df["text"])
    val_labels = torch.tensor(list(val_df["label"]), dtype=torch.long)

    tokenized_train_texts = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    tokenized_val_texts = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

    train_dataset = TensorDataset(tokenized_train_texts['input_ids'], tokenized_train_texts['attention_mask'], train_labels)
    val_dataset = TensorDataset(tokenized_val_texts['input_ids'], tokenized_val_texts['attention_mask'], val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)

    # Training loop
    print("***BEGINNING TRAINING***")
    train_losses, val_losses, val_accuracies, train_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train_preds = 0.0, 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_train_preds += torch.sum(preds == labels).item()

        scheduler.step()
        train_accuracy = correct_train_preds / len(train_df)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss, correct_val_preds = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = outputs.loss
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct_val_preds += torch.sum(preds == labels).item()

        val_accuracy = correct_val_preds / len(val_df)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "fine_tuned.pt")

def main():
    """Configure and run the training routine."""
    MODEL = 'google/bert_uncased_L-4_H-512_A-8'
    EPOCHS = 5
    BATCH_SIZE = 64
    MAX_LENGTH = 64
    LEARNING_RATE = 5e-5
    DROPOUT = 0.01
    WORKERS = 8

    transformers.logging.set_verbosity_error()
    train_model(MODEL, EPOCHS, BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, DROPOUT, WORKERS)

if __name__ == '__main__':
    main()
