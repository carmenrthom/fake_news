import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Fake:  17455
# True:  21192

def main():
    fake_df = pd.read_csv("dataset/original/fake.csv")
    fake_df["label"] = 0
    true_df = pd.read_csv("dataset/original/true.csv")
    true_df["label"] = 1

    merge_df = pd.concat([fake_df, true_df])[["text", "label"]]
    merge_df = merge_df.drop_duplicates()
    merge_df = merge_df.sample(frac = 1)

    train_df, test_df = train_test_split(merge_df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    train_df.to_csv("dataset/train.csv")
    test_df.to_csv("dataset/test.csv")
    val_df.to_csv("dataset/val.csv")
       
if __name__ == "__main__":
    main()