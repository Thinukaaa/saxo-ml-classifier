import sys
import os
import ast
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from utils.db_connection import get_connection
import joblib
# Load data
conn = get_connection()
query = """
SELECT af.*, s.sax_type AS saxophone_type, s.style AS performance_style
FROM dbo.AudioFeatures af
JOIN dbo.AudioSamples s ON af.audio_id = s.audio_id
"""
df = pd.read_sql(query, conn)
print("✅ Data loaded. Columns:", df.columns.tolist())

# Parse MFCCs
def parse_feature(feature_str):
    try:
        return ast.literal_eval(feature_str)
    except:
        return np.nan
df['mfcc_values'] = df['mfcc_values'].apply(parse_feature)
df = df.dropna(subset=['mfcc_values'])
mfcc_df = pd.DataFrame(df['mfcc_values'].tolist(), index=df.index)
df = pd.concat([df, mfcc_df], axis=1)

# Feature matrix and targets
X = df.drop(columns=[
    'audio_id', 'feature_id', 'mfcc_values', 'pitch', 'timbre', 'spectrogram',
    'created_at', 'saxophone_type', 'performance_style'
])
y_sax = df['saxophone_type']
y_style = df['performance_style']

# Helper to get dynamic k_neighbors
def get_k_neighbors(y):
    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    return max(1, min(min_samples - 1, 5))  # ensure at least 1
# Resample saxophone type
k_sax = get_k_neighbors(y_sax)
sm_sax = SMOTE(random_state=42, k_neighbors=k_sax)
X_sax, y_sax_bal = sm_sax.fit_resample(X, y_sax)
# Resample style
k_style = get_k_neighbors(y_style)
sm_style = SMOTE(random_state=42, k_neighbors=k_style)
X_style, y_style_bal = sm_style.fit_resample(X, y_style)
# Pipelines
pipeline_sax = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])
pipeline_style = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])
# Train-test split
X_train_sax, X_test_sax, y_train_sax, y_test_sax = train_test_split(
    X_sax, y_sax_bal, test_size=0.2, random_state=42)
X_train_style, X_test_style, y_train_style, y_test_style = train_test_split(
    X_style, y_style_bal, test_size=0.2, random_state=42)
# Train
pipeline_sax.fit(X_train_sax, y_train_sax)
pipeline_style.fit(X_train_style, y_train_style)
# Evaluate
print("\n--- SVM Evaluation (Saxophone Type) ---")
y_pred_sax = pipeline_sax.predict(X_test_sax)
print(classification_report(y_test_sax, y_pred_sax))
print("Accuracy:", accuracy_score(y_test_sax, y_pred_sax))
print("\n--- SVM Evaluation (Performance Style) ---")
y_pred_style = pipeline_style.predict(X_test_style)
print(classification_report(y_test_style, y_pred_style))
print("Accuracy:", accuracy_score(y_test_style, y_pred_style))
# Save models using joblib (better handling for large models)
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline_sax, 'models/svm_saxophone_model.pkl')
joblib.dump(pipeline_style, 'models/svm_style_model.pkl')
print("✅ Models saved successfully.")
