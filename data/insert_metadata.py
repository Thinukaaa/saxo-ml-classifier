import os
import pyodbc
import soundfile as sf
from datetime import datetime

# Database connection settings
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=saxo_db;"
    "Trusted_Connection=yes;"
)

# Base directory of your dataset
BASE_DIR = r'C:\xampp\htdocs\saxo-ml-classifier\data'

# Function to calculate audio duration
def get_audio_duration(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return round(len(f) / f.samplerate, 2)  # duration in seconds
    except Exception as e:
        print(f"⚠️ Error getting duration for {file_path}: {e}")
        return None

# Insert metadata for a single audio file
def insert_metadata(cursor, file_path, sax_type, style):
    try:
        file_path_clean = file_path.replace('\\', '/')
        file_name = os.path.basename(file_path)
        duration = get_audio_duration(file_path)

        if duration is None:
            return  # Skip if duration couldn't be calculated

        cursor.execute('''
            INSERT INTO AudioSamples (user_id, file_path, file_name, duration, uploaded_at, sax_type, style)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', 1, file_path_clean, file_name, duration, datetime.now(), sax_type.capitalize(), style.capitalize())

        print(f"✅ Inserted: {file_name}")
    except Exception as e:
        print(f"❌ Error inserting {file_path}: {e}")

# Main runner
def main():
    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()

        for sax_type in os.listdir(BASE_DIR):
            sax_dir = os.path.join(BASE_DIR, sax_type)
            if not os.path.isdir(sax_dir): continue

            for style in os.listdir(sax_dir):
                style_dir = os.path.join(sax_dir, style)
                if not os.path.isdir(style_dir): continue

                for filename in os.listdir(style_dir):
                    if filename.endswith('.wav'):
                        full_path = os.path.join(style_dir, filename)
                        insert_metadata(cursor, full_path, sax_type, style)

        conn.commit()
    except Exception as db_err:
        print(f"❌ Database error: {db_err}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
