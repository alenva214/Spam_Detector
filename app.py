from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib
import os

app = Flask(__name__)

# Global variables
model = None
vectorizer = None

def load_model_and_vectorizer():
    global model, vectorizer
    if os.path.exists('model/spam_model.h5') and os.path.exists('vectorizer.pkl'):
        model = load_model('model/spam_model.h5')
        vectorizer = joblib.load('vectorizer.pkl')
    else:
        print("Model or vectorizer not found. Please train the model first.")

# Load model and vectorizer on startup
load_model_and_vectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return render_template('result.html', error="Model or vectorizer not loaded. Please train the model first.")

    email_content = request.form['email_content']

    try:
        # Vectorize the input
        email_vectorized = vectorizer.transform([email_content]).toarray()

        # Make prediction
        prediction = model.predict(email_vectorized)
        probability = prediction[0][0]

        # Classify based on probability
        result = 'spam' if probability > 0.5 else 'ham'

        return render_template('result.html', prediction=result, probability=f"{probability:.2%}")

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('result.html', error=f"An error occurred during prediction: {str(e)}")

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        email_content = request.form['email_content'].splitlines()
        labels = request.form['labels'].splitlines()

        if len(email_content) != len(labels):
            return render_template('train.html', error="Number of emails and labels must match.")

        try:
            # Create DataFrame
            data = pd.DataFrame({'message': email_content, 'label': labels})

            # Preprocess labels
            data['label'] = data['label'].map({'spam': 1, 'ham': 0})

            # Split the data
            X = data['message']
            y = data['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vectorize the text
            global vectorizer
            vectorizer = CountVectorizer()
            X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
            X_test_vectorized = vectorizer.transform(X_test).toarray()

            # Get input shape
            input_shape = X_train_vectorized.shape[1]

            # Build and compile the model
            global model
            model = Sequential([
                Input(shape=(input_shape,)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            # Train the model
            model.fit(X_train_vectorized, y_train, epochs=20, batch_size=32, validation_split=0.2)

            # Evaluate the model
            y_pred = model.predict(X_test_vectorized)
            y_pred_binary = (y_pred > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)

            # Save the model and vectorizer
            model.save('model/spam_model.h5')
            joblib.dump(vectorizer, 'vectorizer.pkl')

            return render_template('train.html', message="Model trained successfully!",
                                   accuracy=f"{accuracy:.4f}", precision=f"{precision:.4f}",
                                   recall=f"{recall:.4f}", f1=f"{f1:.4f}")

        except Exception as e:
            return render_template('train.html', error=f"An error occurred during training: {str(e)}")

    return render_template('train.html')

if __name__ == '__main__':
    app.run(debug=True)