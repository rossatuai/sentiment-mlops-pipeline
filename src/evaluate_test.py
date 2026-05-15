import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

def evaluate_model():

    # Load validation dataset
    df = pd.read_csv("data/validation.csv")

    # Use cleaned text column
    X = df["cleaned_review"]

    # Sentiment labels
    y_true = df["sentiment"]

    # Load trained model
    model = joblib.load("models/sentiment_model.pkl")

    # Generate predictions
    y_pred = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy