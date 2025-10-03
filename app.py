from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from exoplanet_classifier import ExoplanetClassifier
import os

app = Flask(__name__)

# Initialize the classifier
classifier = ExoplanetClassifier()

# Load the trained model (if it exists)
model_path = 'exoplanet_model.joblib'
if os.path.exists(model_path):
    try:
        classifier.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        classifier = None
else:
    print("No trained model found. Please train the model first.")
    classifier = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if classifier is None or classifier.best_model is None:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'})
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        required_fields = [
            'Orbital Period', 'Transit Duration', 'Transit Depth', 
            'Planet Radius', 'Star Eff. Temp', 'Stellar log', 'Star Radius'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'})
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([data])
        
        # Make prediction
        predicted_labels, probabilities = classifier.predict(input_df)
        
        # Get the prediction and confidence
        prediction = predicted_labels[0]
        max_prob_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][max_prob_idx]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(classifier.label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[0][i])
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/model_info')
def model_info():
    """Return information about the trained model"""
    try:
        if classifier is None or classifier.best_model is None:
            return jsonify({'error': 'Model not trained yet.'})
        
        return jsonify({
            'model_type': type(classifier.best_model).__name__,
            'features': classifier.feature_names,
            'classes': classifier.label_encoder.classes_.tolist(),
            'n_features': len(classifier.feature_names)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)