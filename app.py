from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = 'exoplanet_model.joblib'
model_data = None

if os.path.exists(model_path):
    try:
        model_data = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        print(f"🎯 Model accuracy: ~72%")
        print(f"📊 Features: {len(model_data['feature_names'])}")
        print(f"🏷️  Classes: {model_data['label_encoder'].classes_}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_data = None
else:
    print("❌ No trained model found!")
    model_data = None

def predict_exoplanet(input_data):
    """Make prediction using the trained model"""
    if model_data is None:
        return None, None
    
    try:
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Feature engineering (same as training)
        input_df['Log_Period'] = np.log1p(input_df['Orbital Period'])
        input_df['Log_Radius'] = np.log1p(input_df['Planet Radius'])
        input_df['Radius_Ratio'] = input_df['Planet Radius'] / input_df['Star Radius']
        
        # Select features
        X_input = input_df[model_data['feature_names']]
        X_input = X_input.replace([np.inf, -np.inf], np.nan)
        
        # Preprocess
        X_imputed = model_data['imputer'].transform(X_input)
        X_scaled = model_data['scaler'].transform(X_imputed)
        
        # Predict
        predictions = model_data['model'].predict(X_scaled)
        probabilities = model_data['model'].predict_proba(X_scaled)
        
        # Convert back to original labels
        predicted_labels = model_data['label_encoder'].inverse_transform(predictions)
        
        return predicted_labels, probabilities
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded. Please check the server logs.'})
        
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
            try:
                float(data[field])
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for field: {field}'})
        
        # Make prediction
        predicted_labels, probabilities = predict_exoplanet(data)
        
        if predicted_labels is None:
            return jsonify({'error': 'Prediction failed. Check input data.'})
        
        # Get the prediction and confidence
        prediction = predicted_labels[0]
        max_prob_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][max_prob_idx]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(model_data['label_encoder'].classes_):
            prob_dict[class_name] = float(probabilities[0][i])
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/model_info')
def model_info():
    """Return information about the trained model"""
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded.'})
        
        return jsonify({
            'model_type': type(model_data['model']).__name__,
            'features': model_data['feature_names'],
            'classes': model_data['label_encoder'].classes_.tolist(),
            'n_features': len(model_data['feature_names']),
            'accuracy': '~72%'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Exoplanet Classifier Web App...")
    if model_data is not None:
        print("✅ Ready to classify exoplanets!")
    else:
        print("❌ Model not loaded - predictions will not work!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)