# Import required libraries
from flask import Flask, request, jsonify
import joblib
import json

# Create Flask application
app = Flask(__name__)

# Load trained model
model = joblib.load(
    "models/sentiment_model.pkl"
)

# Load text vectorizer
vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# Home page route
@app.route("/")
def home():

    return "Sentiment API Running - Final Test Deployment"


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


# Returns model metrics
@app.route("/metrics", methods=["GET"])
def metrics():

    # Open metrics file
    with open("models/metrics.json", "r") as f:
        metrics_data = json.load(f)

    return jsonify(metrics_data)


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data from request
    data = request.get_json()

    # Get input text
    text = data["text"]

    # Convert text into numerical format
    text_vectorized = vectorizer.transform([text])

    # Generate prediction
    prediction = model.predict(text_vectorized)

    return jsonify({
        "prediction": prediction[0]
    })


# Start Flask application
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",  # Makes app available externally
        port=5000,
        debug=True
    )