import joblib

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

def predict_sentiment(text):

    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)

    return prediction[0]

example = "This game is amazing"

result = predict_sentiment(example)

print(f"Prediction: {result}")