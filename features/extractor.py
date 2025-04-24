import os
import librosa
import numpy as np
import soundfile as sf
import pyodbc

# Configuration
AUDIO_FILE_PATH = r'C:\audio\sax1.wav'   # Update this path as needed
AUDIO_ID = 1                              # Match this to your AudioSamples table

# Database Connection String
CONNECTION_STRING = r'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=saxo_db;Trusted_Connection=yes;'

def extract_features(file_path):
    """
    Extracts MFCC, pitch, and timbre from the given audio file.
    """
    try:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_mean = float(np.mean(pitch))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        timbre = float(np.mean(spectral_centroid))

        return mfcc_mean, pitch_mean, timbre

    except Exception as e:
        raise RuntimeError(f"Failed to extract features: {e}")

def insert_features(audio_id, mfcc_mean, pitch_mean, timbre):
    """
    Inserts extracted features into SQL Server.
    """
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO AudioFeatures (audio_id, mfcc_values, pitch, timbre)
            VALUES (?, ?, ?, ?)
        ''', audio_id, str(mfcc_mean), pitch_mean, timbre)

        conn.commit()
        conn.close()
        print(f"✅ Features for audio_id={audio_id} inserted into database.")

    except Exception as e:
        raise RuntimeError(f"Database insertion failed: {e}")

def main():
    print("🎵 Starting feature extraction...")
    
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"❌ Audio file not found at: {AUDIO_FILE_PATH}")
        return

    try:
        mfcc_mean, pitch_mean, timbre = extract_features(AUDIO_FILE_PATH)

        print("🔍 Extracted Features:")
        print("MFCC Mean:", mfcc_mean)
        print("Pitch Mean:", pitch_mean)
        print("Timbre:", timbre)

        insert_features(AUDIO_ID, mfcc_mean, pitch_mean, timbre)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
