# Import required libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
from sklearn.model_selection import train_test_split

# Download stopwords dataset
nltk.download('stopwords')

# Store English stopwords
STOP_WORDS = set(stopwords.words('english'))


# Function to clean review text
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split text into words
    words = text.split()

    # Remove common words
    filtered_words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


# Load raw data
df = pd.read_csv("data/raw/imdb_dataset.csv")


# Clean review text
df["cleaned_review"] = df["review"].apply(clean_text)


# Create output folders if missing
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data", exist_ok=True)


# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)


# Split dataset into training and validation data
train_df, validation_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# Save training dataset
train_df.to_csv(
    "data/train.csv",
    index=False
)

# Save validation dataset
validation_df.to_csv(
    "data/validation.csv",
    index=False
)

print("Preprocessing complete.")
print("Train dataset saved.")
print("Validation dataset saved.")