import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score



mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis")



df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

X = df["cleaned_review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="positive"
)

with mlflow.start_run():

    mlflow.log_param(
        "model_type",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_features",
        5000
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Training complete.")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")