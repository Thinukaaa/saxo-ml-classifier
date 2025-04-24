import numpy as np
import pandas as pd
import pyodbc
import ast
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Define your connection globally (or pass to the function if needed)
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=MSISWORDTHINUKA;DATABASE=saxo_db;Trusted_Connection=yes;')
# Load and process the dataset from the DB
def load_data():
    query = """
        SELECT 
            p.audio_id,
            p.saxophone_type,
            p.performance_style,
            p.confidence_score,
            af.mfcc_values,
            af.pitch,
            af.timbre
        FROM dbo.Predictions p
        JOIN dbo.AudioFeatures af ON p.audio_id = af.audio_id
    """
    df = pd.read_sql(query, conn)
    print(f"Number of rows retrieved: {len(df)}")
    if df.empty:
        print("No data available. Exiting.")
        exit()
    # Parse and flatten the MFCC values
    def parse_features(row):
        mfcc = ast.literal_eval(row['mfcc_values']) if isinstance(row['mfcc_values'], str) else row['mfcc_values']
        pitch = [row['pitch']] if not isinstance(row['pitch'], list) else row['pitch']
        timbre = [row['timbre']] if not isinstance(row['timbre'], list) else row['timbre']
        return np.array(mfcc + pitch + timbre)
    X = np.array(df.apply(parse_features, axis=1).tolist())

    # Encode targets
    sax_le = LabelEncoder()
    style_le = LabelEncoder()
    y_sax = sax_le.fit_transform(df['saxophone_type'])
    y_style = style_le.fit_transform(df['performance_style'])

    return X, y_sax, y_style

def preprocess_data(X, y_sax, y_style):
    print(f"Shape of X: {X.shape}, y_sax: {len(y_sax)}, y_style: {len(y_style)}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE for saxophone type
    smote_sax = SMOTE(random_state=42)
    X_sax, y_sax = smote_sax.fit_resample(X_scaled, y_sax)

    # SMOTE for style
    smote_style = SMOTE(random_state=42)
    X_style, y_style = smote_style.fit_resample(X_scaled, y_style)

    print(f"After SMOTE - Saxophone: {np.bincount(y_sax)}, Style: {np.bincount(y_style)}")

    # Stratified train-test splits
    X_sax_train, X_sax_test, y_sax_train, y_sax_test = train_test_split(
        X_sax, y_sax, test_size=0.2, random_state=42, stratify=y_sax
    )
    X_style_train, X_style_test, y_style_train, y_style_test = train_test_split(
        X_style, y_style, test_size=0.2, random_state=42, stratify=y_style
    )

    return X_sax_train, X_sax_test, y_sax_train, y_sax_test, \
           X_style_train, X_style_test, y_style_train, y_style_test, \
           X_sax, y_sax, X_style, y_style

def train_model(X_train, X_test, y_train, y_test, label):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- Random Forest Results ({label}) ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def grid_search_tuning(X, y, label):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1]  }
    print(f"\n--- Grid Search Hyperparameter Tuning ({label}) ---")
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    print(f"Best params ({label}): {grid.best_params_}")
    print(classification_report(y, y_pred))
    return best_model
def cross_validation(model, X, y, label):
    print(f"\n--- Cross-Validation Results ({label}) ---")
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{label} CV Accuracy: {scores.mean()}")
def svm_classifier(X, y, label):
    print(f"\n--- SVM Classifier ({label}) ---")
    model = SVC(kernel='linear', class_weight='balanced')
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    # Confusion matrix
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap='Blues')
    plt.title(f'Confusion Matrix ({label})')
    plt.show()
def main():
    X, y_sax, y_style = load_data()
    data = preprocess_data(X, y_sax, y_style)
    X_sax_train, X_sax_test, y_sax_train, y_sax_test, \
    X_style_train, X_style_test, y_style_train, y_style_test, \
    X_sax_resampled, y_sax_resampled, X_style_resampled, y_style_resampled = data
    sax_rf = train_model(X_sax_train, X_sax_test, y_sax_train, y_sax_test, "Saxophone Type")
    style_rf = train_model(X_style_train, X_style_test, y_style_train, y_style_test, "Performance Style")
    best_rf_sax = grid_search_tuning(X_sax_resampled, y_sax_resampled, "Saxophone Type")
    best_rf_style = grid_search_tuning(X_style_resampled, y_style_resampled, "Performance Style")
    cross_validation(best_rf_sax, X_sax_resampled, y_sax_resampled, "Saxophone Type")
    cross_validation(best_rf_style, X_style_resampled, y_style_resampled, "Performance Style")
    svm_classifier(X_sax_resampled, y_sax_resampled, "Saxophone Type (SVM)")

if __name__ == "__main__":
    main()
