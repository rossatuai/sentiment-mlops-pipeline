from flask import Flask, request, jsonify
import joblib
import json

app = Flask(__name__)

# Load trained model
model = joblib.load(
    "models/sentiment_model.pkl"
)

# Load vectorizer
vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


@app.route("/")
def home():
    return "Sentiment API Running - Final Test Deployment"



@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


@app.route("/metrics", methods=["GET"])
def metrics():

    with open("models/metrics.json", "r") as f:
        metrics_data = json.load(f)

    return jsonify(metrics_data)


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)

    return jsonify({
        "prediction": prediction[0]
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )