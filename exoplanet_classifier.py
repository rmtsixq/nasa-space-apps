import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExoplanetClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = KNNImputer(n_neighbors=5)
        self.best_model = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """Comprehensive data preprocessing for maximum accuracy"""
        print("Starting data preprocessing...")
        
        # Make a copy
        data = df.copy()
        
        # Handle Transit Depth - convert to numeric
        data['Transit Depth'] = pd.to_numeric(data['Transit Depth'], errors='coerce')
        
        # Feature Engineering
        print("Engineering new features...")
        
        # 1. Create density proxy (Planet Radius / Star Radius ratio)
        data['Planet_Star_Radius_Ratio'] = data['Planet Radius'] / data['Star Radius']
        
        # 2. Transit signal strength (Transit Depth / Planet Radius)
        data['Transit_Signal_Strength'] = data['Transit Depth'] / data['Planet Radius']
        
        # 3. Orbital velocity proxy (2π * Star Radius / Orbital Period)
        data['Orbital_Velocity_Proxy'] = 2 * np.pi * data['Star Radius'] / data['Orbital Period']
        
        # 4. Stellar habitable zone factor
        # Rough estimate of habitable zone distance based on stellar temperature
        data['Habitable_Zone_Distance'] = np.sqrt(data['Star Eff. Temp'] / 5778)
        data['In_Habitable_Zone'] = (data['Orbital Period'] >= 0.5 * data['Habitable_Zone_Distance']) & \
                                    (data['Orbital Period'] <= 2.0 * data['Habitable_Zone_Distance'])
        
        # 5. Log transformations for skewed features
        data['Log_Orbital_Period'] = np.log1p(data['Orbital Period'])
        data['Log_Transit_Duration'] = np.log1p(data['Transit Duration'])
        data['Log_Transit_Depth'] = np.log1p(data['Transit Depth'])
        data['Log_Planet_Radius'] = np.log1p(data['Planet Radius'])
        
        # 6. Polynomial features for key relationships
        data['Period_Squared'] = data['Orbital Period'] ** 2
        data['Radius_Squared'] = data['Planet Radius'] ** 2
        
        # 7. Stellar classification features
        data['Is_Hot_Star'] = data['Star Eff. Temp'] > 7000
        data['Is_Cool_Star'] = data['Star Eff. Temp'] < 4000
        data['Is_Giant_Star'] = data['Star Radius'] > 2.0
        
        # Select features for modeling
        feature_columns = [
            'Orbital Period', 'Transit Duration', 'Transit Depth', 'Planet Radius',
            'Star Eff. Temp', 'Stellar log', 'Star Radius',
            'Planet_Star_Radius_Ratio', 'Transit_Signal_Strength', 'Orbital_Velocity_Proxy',
            'Log_Orbital_Period', 'Log_Transit_Duration', 'Log_Transit_Depth', 'Log_Planet_Radius',
            'Period_Squared', 'Radius_Squared', 'In_Habitable_Zone', 'Is_Hot_Star', 'Is_Cool_Star', 'Is_Giant_Star'
        ]
        
        # Extract features and target
        X = data[feature_columns].copy()
        y = data['Disposition'].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Impute missing values using KNN imputer (more sophisticated than mean/median)
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
        print(f"Feature names: {self.feature_names}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        return X, y_encoded, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best ensemble"""
        print("Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models with optimized hyperparameters
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        # Train and evaluate individual models
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            model_scores[name] = cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Test score
            test_score = model.score(X_test, y_test)
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}), Test Score: {test_score:.4f}")
        
        # Create ensemble with best models
        print("Creating ensemble model...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', trained_models['RandomForest']),
                ('gb', trained_models['GradientBoosting']),
                ('svm', trained_models['SVM']),
                ('mlp', trained_models['MLP'])
            ],
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        ensemble_score = ensemble.score(X_test, y_test)
        
        print(f"Ensemble Score: {ensemble_score:.4f}")
        
        # Select best model (ensemble if it's better than individual models)
        best_individual_score = max(model_scores.values())
        if ensemble_score > best_individual_score:
            self.best_model = ensemble
            best_score = ensemble_score
            best_name = "Ensemble"
        else:
            best_name = max(model_scores.items(), key=lambda x: x[1])[0]
            self.best_model = trained_models[best_name]
            best_score = model_scores[best_name]
        
        print(f"\\nBest model: {best_name} with score: {best_score:.4f}")
        
        # Final evaluation
        y_pred = self.best_model.predict(X_test)
        
        print(f"\\nFinal Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        
        print(f"\\nClassification Report:")
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
            # Convert single prediction to DataFrame
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Apply same preprocessing steps
        # Feature Engineering (same as in preprocess_data)
        input_df['Planet_Star_Radius_Ratio'] = input_df['Planet Radius'] / input_df['Star Radius']
        input_df['Transit_Signal_Strength'] = input_df['Transit Depth'] / input_df['Planet Radius']
        input_df['Orbital_Velocity_Proxy'] = 2 * np.pi * input_df['Star Radius'] / input_df['Orbital Period']
        input_df['Habitable_Zone_Distance'] = np.sqrt(input_df['Star Eff. Temp'] / 5778)
        input_df['In_Habitable_Zone'] = (input_df['Orbital Period'] >= 0.5 * input_df['Habitable_Zone_Distance']) & \
                                        (input_df['Orbital Period'] <= 2.0 * input_df['Habitable_Zone_Distance'])
        input_df['Log_Orbital_Period'] = np.log1p(input_df['Orbital Period'])
        input_df['Log_Transit_Duration'] = np.log1p(input_df['Transit Duration'])
        input_df['Log_Transit_Depth'] = np.log1p(input_df['Transit Depth'])
        input_df['Log_Planet_Radius'] = np.log1p(input_df['Planet Radius'])
        input_df['Period_Squared'] = input_df['Orbital Period'] ** 2
        input_df['Radius_Squared'] = input_df['Planet Radius'] ** 2
        input_df['Is_Hot_Star'] = input_df['Star Eff. Temp'] > 7000
        input_df['Is_Cool_Star'] = input_df['Star Eff. Temp'] < 4000
        input_df['Is_Giant_Star'] = input_df['Star Radius'] > 2.0
        
        # Select same features
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
    classifier = ExoplanetClassifier()
    
    # Preprocess data
    X, y_encoded, y_original = classifier.preprocess_data(df)
    
    # Train models
    X_test, y_test, y_pred = classifier.train_models(X, y_encoded)
    
    # Save model
    classifier.save_model('exoplanet_model.joblib')
    
    print("\\nModel training complete!")
    print(f"Final accuracy: {accuracy_score(y_test, y_pred):.4f}")