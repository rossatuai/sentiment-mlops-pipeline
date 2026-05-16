import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


# Load raw data
df = pd.read_csv("data/raw/imdb_dataset.csv")


# Clean text
df["cleaned_review"] = df["review"].apply(clean_text)


# Create output folders
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data", exist_ok=True)


# Save full cleaned dataset
df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)


# Create train / validation split
train_df, validation_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# Save datasets for training and monitoring
train_df.to_csv(
    "data/train.csv",
    index=False
)

validation_df.to_csv(
    "data/validation.csv",
    index=False
)

print("Preprocessing complete.")
print("Train dataset saved.")
print("Validation dataset saved.")