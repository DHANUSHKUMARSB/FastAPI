import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    preds = model.predict(X)
    return preds

# ✅ FIXED PATH LOGIC BELOW
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # starter/starter/ml

def save_model(model, encoder):
    model_path = os.path.join(BASE_DIR, "model.joblib")
    encoder_path = os.path.join(BASE_DIR, "encoder.joblib")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

def load_model():
    model_path = os.path.join(BASE_DIR, "model.joblib")
    encoder_path = os.path.join(BASE_DIR, "encoder.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}\n"
                                f"Run 'train_model.py' first to train and save the model.")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at: {encoder_path}\n"
                                f"Run 'train_model.py' first to train and save the encoder.")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder
