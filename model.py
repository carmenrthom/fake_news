import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report

def tokenize_data(texts, labels, model_name, max_length=128):
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(labels) 
    return inputs, labels

def prepare_data(train_df, test_df, val_df, model_name):
    train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()

    train_inputs, train_labels = tokenize_data(train_texts, train_labels, model_name)
    test_inputs, test_labels = tokenize_data(test_texts, test_labels, model_name)
    val_inputs, val_labels = tokenize_data(val_texts, val_labels, model_name)


    train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    test_data = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
    val_data = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

    return train_data, test_data, val_data

def train_model(train_data, test_data, val_data, model_name, epochs=5, batch_size=32, learning_rate= 5e-5, num_workers=0, dropout_rate=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1, attention_probs_dropout_prob=dropout_rate, hidden_dropout_prob=dropout_rate).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()  
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("***BEGINNING TRAINING***")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = torch.sigmoid(outputs.logits.squeeze(dim=1)) 
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = torch.sigmoid(outputs.logits.squeeze(dim=1))
                val_predictions.extend(torch.round(torch.sigmoid(logits)).cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        val_classif = classification_report(val_true_labels, val_predictions)
        print(f'Epoch {epoch + 1}/{epochs}, Classification Report:\n{val_classif}')

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Classification Report (Validation)': val_classif,
        }
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        torch.save(checkpoint, os.path.join("saved_models", f'checkpoint_epoch_{epoch + 1}.pt'))

    # Save final model
    torch.save(model.state_dict(), os.path.join('saved_models', 'final_model.pt'))

def main():
    MODEL = 'google-bert/bert-base-uncased'
    EPOCHS = 5 # Default 5
    BATCH_SIZE = 32 # Default 32
    LEARNING_RATE = 5e-5 # Default 5e-5
    NUM_WORKERS = 0 # Default 0
    DROPOUT = 0.01 # Default 0.01

    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    val_df = pd.read_csv('dataset/val.csv')
    print("***DATASETS LOADED***")
    train_data, test_data, val_data = prepare_data(train_df, test_df, val_df, MODEL)
    print("***DATA PREPAIRED***")
    train_model(train_data, test_data, val_data, MODEL, EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_WORKERS, DROPOUT)

if __name__ == '__main__':
    main()
