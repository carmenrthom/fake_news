# preprocessing.py
"""
Fake News Dataset Preprocessing
Author: Carmen Thom
Description: This script preprocesses the WELFake Dataset by cleaning text data and splitting it into train, test, and validation sets.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

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

def preprocess_data(df):
    """Add combined text columns and handle missing values."""
    df = df.fillna('No info', axis=1)
    df['text'] = 'TITLE: ' + df['title'] + '; TEXT: ' + df['text']
    return df

def main():
    """Main function to preprocess the data."""
    try:
        data = pd.read_csv('dataset/original/WELFake_Dataset.csv')
    except Exception as error:
        print(f"ERROR: Original dataset unable to be loaded ({error})")
        return

    print("*** Preprocessing Started ***")
    data = data[['title', 'text', 'label']]

    data.dropna(subset=['label'], inplace=True)
    data['label'] = data['label'].replace({'Real': 1, 'Fake': 0})
    data['label'] = data['label'].astype(int)
    data['text'] = data['text'].apply(remove_url)
    data['text'] = data['text'].apply(remove_html)
    data['text'] = data['text'].apply(remove_emoji)
    data = preprocess_data(data)

    train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    train_df.to_csv("dataset/train.csv", index=False)
    test_df.to_csv("dataset/test.csv", index=False)
    val_df.to_csv("dataset/val.csv", index=False)
    print("*** Preprocessing Finished ***")
       
if __name__ == "__main__":
    main()