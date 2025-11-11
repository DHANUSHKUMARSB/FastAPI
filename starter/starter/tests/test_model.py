# test_model.py
import sys, os
# Add project root (starter) to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.model import train_model, compute_model_metrics, inference
import numpy as np

def test_train_model():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model = train_model(X, y)
    assert hasattr(model, "predict"), "Trained model must have a predict method"

def test_compute_model_metrics():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference():
    X = np.random.rand(5, 5)
    y = np.array([0, 1, 1, 0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)
