import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(test_df, model):
    max_length, batch_size = 64, 16  # Fixed the syntax error here and set a batch size.

    texts_test = list(test_df["text"])
    test_labels = torch.tensor(list(test_df["label"]), dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
    tokenized_texts_test = tokenizer(texts_test, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

    test_dataset = TensorDataset(tokenized_texts_test['input_ids'], tokenized_texts_test['attention_mask'], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    test_loss = 0.0
    correct_preds = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct_preds += torch.sum(preds == labels).item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def main():
    try:
        model = torch.load("saved_models/saved_model.pt")
        test_df = pd.read_csv("dataset/test.csv")
        evaluate_model(test_df, model)
    except Exception as error:
        print(f"ERROR: {error}\nModel not loaded. Ensure model.py is run first and the model is correctly saved.")

if __name__ == "__main__":
    main()
