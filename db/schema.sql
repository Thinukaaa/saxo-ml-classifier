-- Create the main database
CREATE DATABASE saxofy_db;
GO

USE saxofy_db;
GO

-- Users table
CREATE TABLE Users (
    user_id INT IDENTITY(1,1) PRIMARY KEY,
    name NVARCHAR(100),
    email NVARCHAR(100) UNIQUE,
    password NVARCHAR(255),
    role VARCHAR(20) CHECK (role IN ('admin', 'teacher', 'student')),
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE()
);

-- AudioSamples table
CREATE TABLE AudioSamples (
    audio_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT,
    file_path NVARCHAR(255),
    file_name NVARCHAR(100),
    duration FLOAT,
    uploaded_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- AudioFeatures table
CREATE TABLE AudioFeatures (
    feature_id INT IDENTITY(1,1) PRIMARY KEY,
    audio_id INT,
    mfcc_values NVARCHAR(MAX),     -- JSON or comma-separated values
    pitch FLOAT,
    timbre FLOAT,
    spectrogram NVARCHAR(MAX),     -- Store as compressed string or path to image
    created_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (audio_id) REFERENCES AudioSamples(audio_id)
);

-- Predictions table
CREATE TABLE Predictions (
    prediction_id INT IDENTITY(1,1) PRIMARY KEY,
    audio_id INT,
    saxophone_type VARCHAR(50),
    performance_style VARCHAR(50),
    confidence_score FLOAT,
    predicted_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (audio_id) REFERENCES AudioSamples(audio_id)
);

-- System Logs table
CREATE TABLE SystemLogs (
    log_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT,
    activity NVARCHAR(255),
    timestamp DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);
