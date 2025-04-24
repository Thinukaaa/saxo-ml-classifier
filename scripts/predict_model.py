import pandas as pd
import numpy as np
import joblib
import pyodbc
from ast import literal_eval
from datetime import datetime

# === Connect to DB ===
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=MSISWORDTHINUKA;"
    "Database=saxo_db;"
    "Trusted_Connection=yes;"
)

# === Load models ===
sax_model = joblib.load("models/svm_saxophone_model.pkl")
style_model = joblib.load("models/svm_style_model.pkl")

# === Fetch samples that are not yet predicted ===
query = """
SELECT af.audio_id, af.mfcc_values, af.pitch, af.timbre
FROM dbo.AudioFeatures af
LEFT JOIN dbo.Predictions p ON af.audio_id = p.audio_id
WHERE p.audio_id IS NULL
"""

df = pd.read_sql(query, conn)
if df.empty:
    print("✅ No new data to predict.")
    exit()

print(f"🎯 Predicting on {len(df)} new audio samples...")

# === Parse stringified lists into NumPy arrays ===
def parse_array(column_value):
    try:
        parsed = literal_eval(column_value)
        if isinstance(parsed, list):
            return np.array(parsed, dtype=np.float32)
        return np.array([])
    except Exception:
        return np.array([])

features = []
audio_ids = []

for idx, row in df.iterrows():
    mfcc = parse_array(row["mfcc_values"])
    pitch = parse_array(row["pitch"])
    timbre = parse_array(row["timbre"])

    # Defensive: skip rows with missing or badly parsed data
    if mfcc.size == 0 or pitch.size == 0 or timbre.size == 0:
        continue

    combined = np.concatenate((mfcc, pitch, timbre))
    features.append(combined)
    audio_ids.append(row["audio_id"])
# === Validate before prediction ===
if not features:
    print("⚠️ No valid feature vectors to predict.")
    exit()
X = np.array(features)

# Make sure it's 2D
if X.ndim == 1:
    X = X.reshape(1, -1)
# === Predict ===
sax_preds = sax_model.predict(X)
style_preds = style_model.predict(X)
# === Confidence ===
if hasattr(sax_model, "predict_proba") and hasattr(style_model, "predict_proba"):
    sax_probs = sax_model.predict_proba(X)
    style_probs = style_model.predict_proba(X)

    sax_conf = sax_probs.max(axis=1)
    style_conf = style_probs.max(axis=1)
    confidence = ((sax_conf + style_conf) / 2).round(3)
else:
    confidence = [1.0] * len(audio_ids)
# === Insert predictions into DB ===
cursor = conn.cursor()
for audio_id, sax, style, conf in zip(audio_ids, sax_preds, style_preds, confidence):
    cursor.execute("""
        INSERT INTO dbo.Predictions (audio_id, saxophone_type, performance_style, confidence_score, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """, audio_id, sax, style, conf, datetime.now())
conn.commit()
cursor.close()
print(f"✅ Predictions saved for {len(audio_ids)} samples.")
