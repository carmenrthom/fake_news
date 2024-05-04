# Preprocessing function for fake news dataset
# Carmen Thom

import re
import pandas as pd
from sklearn.model_selection import train_test_split

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',str(text))

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',str(text))

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text)) 

def processing(df):
    df = df.fillna('No info', axis=1)
    df['text'] = 'TITLE: ' + df.title + '; TEXT: ' + df.text
    return df

def main():
    try :
        data = pd.read_csv('dataset/original/WELFake_Dataset.csv')
    except Exception as error:
        print("ERROR: Original unable to be loaded")
        return
    
    print("***Preprocessing Started***")
    data = data[['title','text','label']]

    data.dropna(subset=['label'], inplace=True)
    data['label'] = data['label'].replace({'Real': 1, 'Fake': 0})
    data['label'] = data['label'].astype(int)
    data['text'] = data['text'].apply(remove_url)
    data['text'] = data['text'].apply(remove_html)
    data['text'] = data['text'].apply(remove_emoji)
    data = processing(data)

    train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    train_df.to_csv("dataset/train.csv")
    test_df.to_csv("dataset/test.csv")
    val_df.to_csv("dataset/val.csv")
    print("***Preprocessing Finished***")
       
if __name__ == "__main__":
    main()
