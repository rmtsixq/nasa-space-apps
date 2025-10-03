import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

print('🚀 Building MAXIMUM ACCURACY ensemble model...')
df = pd.read_csv('Space Apps Data - TOI Data (1).csv')
data = df.copy()
data['Transit Depth'] = pd.to_numeric(data['Transit Depth'], errors='coerce')

# ADVANCED feature engineering
print('⚡ Advanced feature engineering...')
data['Log_Period'] = np.log1p(data['Orbital Period'])
data['Log_Duration'] = np.log1p(data['Transit Duration'])
data['Log_Depth'] = np.log1p(data['Transit Depth'])
data['Log_Radius'] = np.log1p(data['Planet Radius'])

# Ratios and interactions
data['Radius_Ratio'] = data['Planet Radius'] / data['Star Radius']
data['Period_Duration_Ratio'] = data['Orbital Period'] / data['Transit Duration']
data['Depth_Radius_Ratio'] = data['Transit Depth'] / data['Planet Radius']

# Binned features
data['Temp_Hot'] = (data['Star Eff. Temp'] > 6000).astype(int)
data['Large_Planet'] = (data['Planet Radius'] > 5.0).astype(int)
data['Short_Period'] = (data['Orbital Period'] < 10).astype(int)

features = [
    'Orbital Period', 'Transit Duration', 'Transit Depth', 'Planet Radius', 
    'Star Eff. Temp', 'Stellar log', 'Star Radius',
    'Log_Period', 'Log_Duration', 'Log_Depth', 'Log_Radius',
    'Radius_Ratio', 'Period_Duration_Ratio', 'Depth_Radius_Ratio',
    'Temp_Hot', 'Large_Planet', 'Short_Period'
]

X = data[features].copy()
y = data['Disposition'].copy()
X = X.replace([np.inf, -np.inf], np.nan)

print(f'📊 Using {len(features)} features')

# Preprocessing
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
label_encoder = LabelEncoder()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print('🤖 Training ensemble models...')

# High-performance Random Forest
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=20, 
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42, 
    n_jobs=-1
)

# High-performance Gradient Boosting  
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42
)

print('  🌲 Training Random Forest...')
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

print('  🚀 Training Gradient Boosting...')
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)

print('  🎭 Creating ensemble...')
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'🏆 RESULTS:')
print(f'  Random Forest: {rf_score:.4f}')
print(f'  Gradient Boost: {gb_score:.4f}')
print(f'  🎯 ENSEMBLE: {accuracy:.4f}')

# Use best model
if accuracy >= max(rf_score, gb_score):
    best_model = ensemble
    final_accuracy = accuracy
    model_name = "Ensemble"
else:
    if rf_score > gb_score:
        best_model = rf
        final_accuracy = rf_score
        model_name = "Random Forest"
    else:
        best_model = gb
        final_accuracy = gb_score
        model_name = "Gradient Boosting"

print(f'✅ Using {model_name}: {final_accuracy:.4f}')

print('\n📋 Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model
model_data = {
    'model': best_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'imputer': imputer,
    'feature_names': features
}
joblib.dump(model_data, 'exoplanet_model.joblib')
print(f'💾 Model saved! Final accuracy: {final_accuracy:.4f}')