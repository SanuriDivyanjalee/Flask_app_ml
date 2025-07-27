# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "🎉 Iris ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)
    return jsonify({'predicted_class': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)