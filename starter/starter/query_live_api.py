import requests
import sys

URL = "https://census-fastapi-dhanush.onrender.com/predict"

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

def main():
    try:
        print(f"POSTing to: {URL}\n")
        r = requests.post(URL, json=sample, timeout=10)
    except Exception as e:
        print("Error making request:", e)
        sys.exit(1)

    print("HTTP status code:", r.status_code)
    try:
        print("Response JSON:", r.json())
    except ValueError:
        print("Response text:", r.text)

if __name__ == "__main__":
    main()
