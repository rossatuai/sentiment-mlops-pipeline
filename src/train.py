# Import required libraries
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# MLflow configuration
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis")


# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)


# Create validation dataset for monitoring
df.sample(
    100,
    random_state=42
).to_csv(
    "data/validation.csv",
    index=False
)


# Features and labels
X = df["cleaned_review"]
y = df["sentiment"]


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


# Create machine learning model
model = LogisticRegression()

# Train model
model.fit(
    X_train,
    y_train
)


# Generate predictions
predictions = model.predict(
    X_test
)


# Calculate performance metrics
accuracy = accuracy_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="positive"
)


# Log information to MLflow
with mlflow.start_run():

    mlflow.log_param(
        "model_type",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_features",
        5000
    )

    # Save accuracy metric
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Save F1 score metric
    mlflow.log_metric(
        "f1_score",
        f1
    )

    # Save model as MLflow artifact
    try:

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print("Model successfully logged to MLflow")

    except Exception as e:

        print(
            f"MLflow model logging failed: {e}"
        )


# Create models folder if missing
os.makedirs(
    "models",
    exist_ok=True
)

# Save trained model
joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# Save metrics locally
metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1)
}

with open(
    "models/metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f
    )


print("\nTraining complete.")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")