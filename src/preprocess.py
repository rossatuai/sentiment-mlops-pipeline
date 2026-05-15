import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os


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

df = pd.read_csv("data/raw/imdb_dataset.csv")

df["cleaned_review"] = df["review"].apply(clean_text)


os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("Preprocessing complete.")
