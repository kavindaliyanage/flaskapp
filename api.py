from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the saved model and vectorizer
model = joblib.load('./rubber_category_model.joblib')
vectorizer = joblib.load('./tfidf_vectorizer.joblib')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data['description']

    # Preprocess the description
    processed_description = preprocess_text(description)

    # Vectorize the preprocessed description
    vectorized_description = vectorizer.transform([processed_description])

    # Make prediction
    prediction = model.predict(vectorized_description)[0]

    return jsonify({'category': prediction})


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)