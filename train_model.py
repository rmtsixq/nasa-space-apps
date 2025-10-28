import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Loading TOI dataset...")
df = pd.read_csv('Space Apps Data - TOI Data (1).csv')
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

print("🔧 Preprocessing data...")
data = df.copy()
data['Transit Depth'] = pd.to_numeric(data['Transit Depth'], errors='coerce')

print("⚡ Engineering advanced features...")
# Advanced feature engineering for maximum accuracy
data['Planet_Star_Radius_Ratio'] = data['Planet Radius'] / data['Star Radius']
data['Transit_Signal_Strength'] = data['Transit Depth'] / data['Planet Radius']
data['Orbital_Velocity_Proxy'] = 2 * np.pi * data['Star Radius'] / data['Orbital Period']
data['Habitable_Zone_Distance'] = np.sqrt(data['Star Eff. Temp'] / 5778)
data['In_Habitable_Zone'] = (data['Orbital Period'] >= 0.5 * data['Habitable_Zone_Distance']) & (data['Orbital Period'] <= 2.0 * data['Habitable_Zone_Distance'])

# Log transformations
data['Log_Orbital_Period'] = np.log1p(data['Orbital Period'])
data['Log_Transit_Duration'] = np.log1p(data['Transit Duration'])
data['Log_Transit_Depth'] = np.log1p(data['Transit Depth'])
data['Log_Planet_Radius'] = np.log1p(data['Planet Radius'])

# Polynomial features
data['Period_Squared'] = data['Orbital Period'] ** 2
data['Radius_Squared'] = data['Planet Radius'] ** 2

# Stellar classification
data['Is_Hot_Star'] = data['Star Eff. Temp'] > 7000
data['Is_Cool_Star'] = data['Star Eff. Temp'] < 4000
data['Is_Giant_Star'] = data['Star Radius'] > 2.0

feature_columns = [
    'Orbital Period', 'Transit Duration', 'Transit Depth', 'Planet Radius',
    'Star Eff. Temp', 'Stellar log', 'Star Radius',
    'Planet_Star_Radius_Ratio', 'Transit_Signal_Strength', 'Orbital_Velocity_Proxy',
    'Log_Orbital_Period', 'Log_Transit_Duration', 'Log_Transit_Depth', 'Log_Planet_Radius',
    'Period_Squared', 'Radius_Squared', 'In_Habitable_Zone', 'Is_Hot_Star', 'Is_Cool_Star', 'Is_Giant_Star'
]

X = data[feature_columns].copy()
y = data['Disposition'].copy()
X = X.replace([np.inf, -np.inf], np.nan)

print(f"🎯 Feature set: {len(feature_columns)} features")

# Advanced preprocessing
print("🧠 Using KNN imputation...")
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()
label_encoder = LabelEncoder()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
y_encoded = label_encoder.fit_transform(y)

print(f"📊 Target classes: {label_encoder.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("🤖 Training high-performance models...")

# High-performance Random Forest
print("  🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print(f"     Random Forest accuracy: {rf_score:.4f}")

# High-performance Gradient Boosting
print("  🚀 Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)
print(f"     Gradient Boosting accuracy: {gb_score:.4f}")

# High-performance SVM
print("  ⚡ Training SVM...")
svm = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)
svm_score = svm.score(X_test, y_test)
print(f"     SVM accuracy: {svm_score:.4f}")

# Create ultimate ensemble
print("🎭 Creating ultimate ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svm', svm)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Final evaluation
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("🏆 FINAL MODEL PERFORMANCE:")
print(f"   🎯 Accuracy:  {accuracy:.4f}")
print(f"   🔍 Precision: {precision:.4f}")
print(f"   📡 Recall:    {recall:.4f}")
print(f"   ⚖️  F1-Score:  {f1:.4f}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the complete model
print("💾 Saving model...")
model_data = {
    'model': ensemble,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'imputer': imputer,
    'feature_names': feature_columns
}
joblib.dump(model_data, 'exoplanet_model.joblib')

print("✅ Model saved successfully to 'exoplanet_model.joblib'")
print("🚀 Ready for web interface!")