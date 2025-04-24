import os
import numpy as np
import pandas as pd
import joblib
import pyodbc
from ast import literal_eval
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1) Load your pipelines (trained on MFCC only)
sax_model   = joblib.load("models/svm_saxophone_model.pkl")
style_model = joblib.load("models/svm_style_model.pkl")

# 2) Fetch test set from DB
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=MSISWORDTHINUKA;"
    "Database=saxo_db;"
    "Trusted_Connection=yes;"
)
query = """
SELECT af.audio_id, af.mfcc_values,
       s.sax_type    AS saxophone_type,
       s.style       AS performance_style
FROM dbo.AudioFeatures af
JOIN dbo.AudioSamples s 
  ON af.audio_id = s.audio_id
WHERE s.is_test = 1
"""
df = pd.read_sql(query, conn)
conn.close()

if df.empty:
    print("⚠️ No test samples found (is_test = 1).")
    exit()

# 3) Parse MFCC and build feature matrix
X_list, y_sax, y_style = [], [], []
for _, row in df.iterrows():
    try:
        mfcc = np.array(literal_eval(row["mfcc_values"]), dtype=np.float32)
    except Exception:
        # skip malformed rows
        continue

    if mfcc.size == 0:
        continue

    # Only MFCCs (20 features) — no pitch/timbre
    X_list.append(mfcc)
    y_sax.append(row["saxophone_type"])
    y_style.append(row["performance_style"])

if not X_list:
    print("⚠️ No valid MFCC vectors to evaluate on.")
    exit()

X = np.stack(X_list)      # shape = (n_samples, 20)
y_sax   = np.array(y_sax)
y_style = np.array(y_style)

# 4) Perform predictions
sax_pred   = sax_model.predict(X)
style_pred = style_model.predict(X)

# 5) Evaluate Saxophone Type
print("\n--- Saxophone Type on Test Set ---")
print(f"Accuracy: {accuracy_score(y_sax, sax_pred):.3f}")
print(classification_report(y_sax, sax_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_sax, sax_pred))

# 6) Evaluate Performance Style
print("\n--- Performance Style on Test Set ---")
print(f"Accuracy: {accuracy_score(y_style, style_pred):.3f}")
print(classification_report(y_style, style_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_style, style_pred))
