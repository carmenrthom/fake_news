# BERT Binary Classifier 
# Carmen Thom

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW

def train_model(model_name, epochs=5, batch_size=32, max_length = 64, learning_rate= 5e-5, dropout_rate=0.01, workers = 0):
    # Model hyperparameterization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, attention_probs_dropout_prob=dropout_rate, hidden_dropout_prob=dropout_rate, output_attentions=True).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate}
    ], lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Importing / Processing Texts
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/val.csv')
    train_texts = list(train_df["text"])
    train_labels = torch.tensor(list(train_df["label"]), dtype=torch.long)
    val_texts = list(val_df["text"])
    val_labels = torch.tensor(list(val_df["label"]), dtype=torch.long)

    tokenized_train_texts = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    tokenized_val_texts = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

    # Creating dataset for batch loading
    train_dataset = TensorDataset(tokenized_train_texts['input_ids'], tokenized_train_texts['attention_mask'], train_labels)
    val_dataset = TensorDataset(tokenized_val_texts['input_ids'], tokenized_val_texts['attention_mask'], val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers,  shuffle=False)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Model Training
    print("***BEGINNING TRAINING***")
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        tr_correct_preds = 0
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
            tr_correct_preds += torch.sum(tr_preds == labels).item()
            
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        train_accuracy = tr_correct_preds / len(train_df)
        train_accuracies.append(train_accuracy)

        #Validation
        model.eval()
        val_loss = 0.0
        correct_preds = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct_preds += torch.sum(preds == labels).item()
                
        # Validation calculations
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)        

        val_accuracy = correct_preds / len(val_df)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    torch.save(model, "fine_tuned.pt")

def main():
    MODEL = 'google/bert_uncased_L-4_H-512_A-8' # More models available at https://github.com/google-research/bert
    EPOCHS = 5 # Default 5
    BATCH_SIZE = 64 # Default 32
    MAX_LENGTH = 64 # Default 64
    LEARNING_RATE = 5e-5 # Default 5e-5
    DROPOUT = 0.01 # Default 0.01
    WORKERS = 8 # Default 0

    train_model(MODEL, EPOCHS, BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, DROPOUT, WORKERS)

if __name__ == '__main__':
    main()
