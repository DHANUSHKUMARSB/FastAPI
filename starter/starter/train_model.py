# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import numpy as np
from starter.starter.ml.model import train_model, compute_model_metrics, inference, save_model

# Load the dataset
data = pd.read_csv("../data/census.csv")

# Clean column names (remove spaces)
data.columns = data.columns.str.strip()

# Identify categorical features
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

# Split data
X = data.drop("salary", axis=1)
y = data["salary"]

# Binarize the label
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[cat_features])
X_test_encoded = encoder.transform(X_test[cat_features])

# Combine encoded categorical features with numeric ones
X_train_proc = np.concatenate([X_train_encoded, X_train.drop(columns=cat_features)], axis=1)
X_test_proc = np.concatenate([X_test_encoded, X_test.drop(columns=cat_features)], axis=1)

# Train model
model = train_model(X_train_proc, y_train)

# Evaluate model
preds = inference(model, X_test_proc)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Model performance:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {fbeta:.3f}")

# Save model and encoder
save_model(model, encoder)
print("Model and encoder saved successfully.")

def performance_on_slices(data, model, encoder, lb, cat_feature, output_file="slice_output.txt"):
    """
    Evaluate model performance on slices of data for one categorical feature.
    Saves results to a text file.
    """
    import numpy as np
    from ml.model import inference, compute_model_metrics

    slices = []
    unique_values = data[cat_feature].unique()

    with open(output_file, "w") as f:
        for val in unique_values:
            subset = data[data[cat_feature] == val]
            X_slice = subset.drop("salary", axis=1)
            y_slice = lb.transform(subset["salary"]).ravel()

            # Encode and process data
            X_slice_encoded = encoder.transform(X_slice[encoder.feature_names_in_])
            X_slice_proc = np.concatenate(
                [X_slice_encoded, X_slice.drop(columns=encoder.feature_names_in_)],
                axis=1
            )

            preds = inference(model, X_slice_proc)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            result = (
                f"{cat_feature} = {val} | "
                f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {fbeta:.3f}\n"
            )
            f.write(result)
            slices.append(result)
    print(f"\nSlice evaluation saved to {output_file}\n")
    return slices

performance_on_slices(data, model, encoder, lb, cat_feature="education")
