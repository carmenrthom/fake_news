# evaluation.py
"""
BERT Model Evaluation Script
Author: Carmen Thom
Description: This script loads the fine-tuned BERT model trained on the WELFake dataset and evaluates its performance on a test partition,
             generating a classification report and visualizing confusion matrices.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(test_df, model):
    """Evaluate the model's performance on the test dataset."""
    max_length, batch_size = 64, 16
    texts_test = list(test_df["text"])
    test_labels = torch.tensor(list(test_df["label"]), dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
    tokenized_texts_test = tokenizer(texts_test, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

    test_dataset = TensorDataset(tokenized_texts_test['input_ids'], tokenized_texts_test['attention_mask'], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    test_loss, correct_preds = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct_preds += torch.sum(preds == labels).item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:\n", classification_report(all_labels, all_preds))

    # Displaying the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalized confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cmn)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.show()

    # Identifying type I and type II errors
    fp_indices = np.where((np.array(all_preds) == 1) & (np.array(all_labels) == 0))[0]
    fn_indices = np.where((np.array(all_preds) == 0) & (np.array(all_labels) == 1))[0]

    with open('type_1_errors.txt', 'w') as f:
        for index in fp_indices:
            f.write(texts_test[index] + '\n')
    
    with open('type_2_errors.txt', 'w') as f:
        for index in fn_indices:
            f.write(texts_test[index] + '\n')

def main():
    """Main function to load the model and test data, and perform evaluation."""
    try:
        model = torch.load("fine_tuned.pt")
        test_df = pd.read_csv("dataset/test.csv")
        evaluate_model(test_df, model)
    except Exception as error:
        print(f"ERROR: {error}\nModel not loaded. Ensure model.py is run first and the model is correctly saved.")

if __name__ == "__main__":
    main()
