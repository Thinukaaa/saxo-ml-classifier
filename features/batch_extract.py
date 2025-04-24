import os
import json
import pyodbc
from datetime import datetime
import librosa

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

        return {
            "mfcc": mfcc.mean(axis=1).tolist(),  # averaging across time
            "pitch": float(pitch),
            "timbre": float(spectral_centroid)
        }
    except Exception as e:
        print(f"❌ Feature extraction failed for {file_path}: {e}")
        return None

def connect_db():
    return pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=saxo_db;'
        'Trusted_Connection=yes;'
    )

def audio_already_processed(cursor, audio_id):
    cursor.execute("SELECT 1 FROM dbo.AudioFeatures WHERE audio_id = ?", audio_id)
    return cursor.fetchone() is not None

def get_all_audio_samples():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT audio_id, file_path FROM AudioSamples")
    data = cursor.fetchall()
    conn.close()
    return data

def insert_features(audio_id, features):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO dbo.AudioFeatures (audio_id, mfcc_values, pitch, timbre, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', audio_id, json.dumps(features["mfcc"]), features["pitch"], features["timbre"], datetime.now())

        conn.commit()
        print(f"✅ Features inserted for audio_id {audio_id}")
    except Exception as e:
        print(f"❌ Insert failed for audio_id {audio_id}: {e}")
    finally:
        conn.close()

def main():
    audio_samples = get_all_audio_samples()
    print(f"🔍 Found {len(audio_samples)} audio samples.")

    for audio_id, file_path in audio_samples:
        conn = connect_db()
        cursor = conn.cursor()

        if audio_already_processed(cursor, audio_id):
            print(f"⏭️ Skipping audio_id {audio_id} — already processed.")
            conn.close()
            continue

        conn.close()
        features = extract_audio_features(file_path)
        if features:
            insert_features(audio_id, features)

if __name__ == "__main__":
    main()
