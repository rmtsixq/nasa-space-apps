import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading TOI dataset...")
df = pd.read_csv('Space Apps Data - TOI Data (1).csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Check data types and missing values
print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nMissing values percentage:")
print((df.isnull().sum() / len(df)) * 100)

# Check class distribution
print(f"\nClass distribution:")
print(df['Disposition'].value_counts())
print(f"\nClass distribution (percentages):")
print(df['Disposition'].value_counts(normalize=True) * 100)

# Basic statistics for numerical columns
print(f"\nBasic statistics:")
print(df.describe())

# Visualization of class distribution
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df['Disposition'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Disposition')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Check for outliers in numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical columns: {numerical_cols}")

plt.subplot(2, 2, 2)
for i, col in enumerate(numerical_cols[:3]):  # First 3 numerical columns
    plt.boxplot(df[col].dropna(), positions=[i], widths=0.6)
plt.title('Box plots for numerical features')
plt.xticks(range(len(numerical_cols[:3])), numerical_cols[:3], rotation=45)

# Correlation matrix
plt.subplot(2, 2, 3)
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("Data analysis complete!")