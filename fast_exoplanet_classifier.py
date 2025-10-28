import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class FastExoplanetClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.best_model = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """Optimized preprocessing for faster training"""
        print("Starting data preprocessing...")
        
        # Make a copy
        data = df.copy()
        
        # Handle Transit Depth - convert to numeric
        data['Transit Depth'] = pd.to_numeric(data['Transit Depth'], errors='coerce')
        
        # Feature Engineering - only the most important features
        print("Engineering key features...")
        
        # 1. Log transformations for skewed features
        data['Log_Orbital_Period'] = np.log1p(data['Orbital Period'])
        data['Log_Transit_Duration'] = np.log1p(data['Transit Duration'])
        data['Log_Transit_Depth'] = np.log1p(data['Transit Depth'])
        data['Log_Planet_Radius'] = np.log1p(data['Planet Radius'])
        
        # 2. Key ratios
        data['Planet_Star_Radius_Ratio'] = data['Planet Radius'] / data['Star Radius']
        data['Transit_Signal_Strength'] = data['Transit Depth'] / data['Planet Radius']
        
        # 3. Stellar classification
        data['Is_Hot_Star'] = (data['Star Eff. Temp'] > 7000).astype(int)
        data['Is_Cool_Star'] = (data['Star Eff. Temp'] < 4000).astype(int)
        
        # Select features for modeling
        feature_columns = [
            'Orbital Period', 'Transit Duration', 'Transit Depth', 'Planet Radius',
            'Star Eff. Temp', 'Stellar log', 'Star Radius',
            'Log_Orbital_Period', 'Log_Transit_Duration', 'Log_Transit_Depth', 'Log_Planet_Radius',
            'Planet_Star_Radius_Ratio', 'Transit_Signal_Strength', 'Is_Hot_Star', 'Is_Cool_Star'
        ]
        
        # Extract features and target
        X = data[feature_columns].copy()
        y = data['Disposition'].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Impute missing values
        print("Imputing missing values...")
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Final feature set: {X.shape[1]} features")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        return X, y_encoded, y
    
    def train_models(self, X, y):
        """Train optimized models for speed and accuracy"""
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define optimized models
        rf = RandomForestClassifier(
            n_estimators=200,  # Reduced from 300
            max_depth=15,      # Reduced from 20
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,  # Reduced from 200
            learning_rate=0.1,
            max_depth=6,       # Reduced from 8
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        print("Training Random Forest...")
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        print(f"Random Forest accuracy: {rf_score:.4f}")
        
        print("Training Gradient Boosting...")
        gb.fit(X_train, y_train)
        gb_score = gb.score(X_test, y_test)
        print(f"Gradient Boosting accuracy: {gb_score:.4f}")
        
        # Create ensemble
        print("Creating ensemble...")
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        ensemble_score = ensemble.score(X_test, y_test)
        print(f"Ensemble accuracy: {ensemble_score:.4f}")
        
        # Select best model
        scores = {'RandomForest': rf_score, 'GradientBoosting': gb_score, 'Ensemble': ensemble_score}
        best_name = max(scores.items(), key=lambda x: x[1])[0]
        
        if best_name == 'RandomForest':
            self.best_model = rf
        elif best_name == 'GradientBoosting':
            self.best_model = gb
        else:
            self.best_model = ensemble
            
        print(f"\\nBest model: {best_name} with accuracy: {scores[best_name]:.4f}")
        
        # Final evaluation
        y_pred = self.best_model.predict(X_test)
        
        print(f"\\nFinal Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        
        print(f"\\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return X_test, y_test, y_pred
    
    def save_model(self, filepath='exoplanet_model.joblib'):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='exoplanet_model.joblib'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.imputer = model_data['imputer']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Apply same preprocessing
        input_df['Log_Orbital_Period'] = np.log1p(input_df['Orbital Period'])
        input_df['Log_Transit_Duration'] = np.log1p(input_df['Transit Duration'])
        input_df['Log_Transit_Depth'] = np.log1p(input_df['Transit Depth'])
        input_df['Log_Planet_Radius'] = np.log1p(input_df['Planet Radius'])
        input_df['Planet_Star_Radius_Ratio'] = input_df['Planet Radius'] / input_df['Star Radius']
        input_df['Transit_Signal_Strength'] = input_df['Transit Depth'] / input_df['Planet Radius']
        input_df['Is_Hot_Star'] = (input_df['Star Eff. Temp'] > 7000).astype(int)
        input_df['Is_Cool_Star'] = (input_df['Star Eff. Temp'] < 4000).astype(int)
        
        # Select features and handle missing values
        X_input = input_df[self.feature_names]
        X_input = X_input.replace([np.inf, -np.inf], np.nan)
        
        # Impute and scale
        X_imputed = self.imputer.transform(X_input)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities

# Main training script
if __name__ == "__main__":
    # Load data
    print("Loading TOI dataset...")
    df = pd.read_csv('Space Apps Data - TOI Data (1).csv')
    
    # Initialize classifier
    classifier = FastExoplanetClassifier()
    
    # Preprocess data
    X, y_encoded, y_original = classifier.preprocess_data(df)
    
    # Train models
    X_test, y_test, y_pred = classifier.train_models(X, y_encoded)
    
    # Save model
    classifier.save_model('exoplanet_model.joblib')
    
    print("\\n🎉 Model training complete!")
    print(f"🎯 Final accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("✅ Model saved successfully!")
    print("\\n🚀 Ready to launch web interface!")