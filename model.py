# model.py
"""
BERT Binary Classifier
Author: Carmen Thom
Description: This script defines and trains a BERT-based binary classifier using the PyTorch library,
             training on the WELFake dataset for fake news detection, and saving the fine-tuned model.
"""

import os
import re
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def remove_url(text):
    """Remove URLs from a text string."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', str(text))

def remove_html(text):
    """Remove HTML tags from a text string."""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', str(text))

def remove_emoji(text):
    """Remove emojis from a text string."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF" 
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text)) 



def preproccess(data):
    """Main function to preprocess the data."""

    print("*** Preprocessing Started ***")
    data = data["train"].to_pandas()
    data = data.fillna('No info', axis=1)
    data['text'] = 'TITLE: ' + data['title'] + '; TEXT: ' + data['text']
    data = data[["text", "label"]]
    data['label'] = data['label'].astype(int)
    data['text'] = data['text'].apply(remove_url)
    data['text'] = data['text'].apply(remove_html)
    data['text'] = data['text'].apply(remove_emoji)
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)
    print("*** Preprocessing Finished ***")
    return train_df, val_df, test_df

def train_model(model_name, epochs=5, batch_size=32, max_length=64, learning_rate=5e-5, dropout_rate=0.01, workers=0):
    """Train a BERT model with specified parameters."""
    data = load_dataset("davanstrien/WELFake")
    train_df, val_df, test_df = preproccess(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        attention_probs_dropout_prob=dropout_rate,
        hidden_dropout_prob=dropout_rate,
        output_attentions=True
    ).to(device)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate}
    ], lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Data preparation
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
        train_loss, train_correct_preds = 0.0,  0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            tr_loss = outputs.loss
            train_loss += tr_loss.item()
            tr_loss.backward()

            tr_logits = outputs.logits
            tr_preds = torch.argmax(tr_logits, dim=1)
            train_correct_preds += torch.sum(tr_preds == labels).item()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = train_correct_preds / len(train_df)

        # Validation phase
        model.eval()
        val_loss, correct_val_preds = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct_val_preds += torch.sum(preds == labels).item()

        # Validation calculations
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)  

        val_accuracy = correct_val_preds / len(val_df)
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model, "fine_tuned.pt")

def main():
    """Configure and run the training routine."""
    MODEL = 'google/bert_uncased_L-4_H-512_A-8'
    EPOCHS = 5
    BATCH_SIZE = 64
    MAX_LENGTH = 64
    LEARNING_RATE = 5e-5
    DROPOUT = 0.1
    WORKERS = 8

    transformers.logging.set_verbosity_error()
    train_model(MODEL, EPOCHS, BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, DROPOUT, WORKERS)

if __name__ == '__main__':
    main()
