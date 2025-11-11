# main.py
import sys, os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from starter.starter.ml.model import load_model, inference


# Load model and encoder
model, encoder = load_model()

app = FastAPI(title="Census Income Prediction API",
              description="Predict whether a person earns >50K or <=50K based on census data.",
              version="1.0")

# Define Pydantic model for POST input
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        populate_by_name = True

@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/predict")
def predict(data: CensusData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict(by_alias=True)])

    # Reorder and encode categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    encoded = encoder.transform(input_df[cat_features])
    encoded_df = np.concatenate(
        [encoded, input_df.drop(columns=cat_features)], axis=1
    )

    # Run inference
    prediction = inference(model, encoded_df)
    result = ">50K" if prediction[0] == 1 else "<=50K"

    return {"prediction": result}
