# Import library for loading saved files
import joblib

# Load trained sentiment model
model = joblib.load(
    "models/sentiment_model.pkl"
)

# Load TF-IDF vectorizer
vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# Function to predict sentiment
def predict_sentiment(text):

    # Convert text into numerical format
    text_vectorized = vectorizer.transform([text])

    # Generate prediction
    prediction = model.predict(text_vectorized)

    return prediction[0]


# Example input text
example = "This movie is amazing"

# Store prediction result
result = predict_sentiment(example)

# Display result
print(f"Prediction: {result}")