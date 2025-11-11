import sys, os
import pytest
from fastapi.testclient import TestClient

# Fix path imports so FastAPI finds main and ml modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from starter.main import app

client = TestClient(app)

def test_get_root():
    """Test GET endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API!"}

def test_post_predict_low_income():
    """Test POST endpoint for <=50K prediction."""
    sample = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] in [">50K", "<=50K"]

def test_post_predict_high_income():
    """Test POST endpoint for >50K prediction."""
    sample = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] in [">50K", "<=50K"]
