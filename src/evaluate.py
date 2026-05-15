import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

def evaluate_model():

    # Load validation dataset
    df = pd.read_csv("data/validation.csv")

    # Features and labels
    X = df["cleaned_review"]
    y_true = df["sentiment"]

    # Load trained TF-IDF vectorizer
    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    # Transform text into vectors
    X_vectorized = vectorizer.transform(X)

    # Load trained model
    model = joblib.load(
        "models/sentiment_model.pkl"
    )

    # Generate predictions
    y_pred = model.predict(X_vectorized)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return accuracy