import pandas as pd
import numpy as np
import joblib
from onnxruntime import InferenceSession
from pathlib import Path

model = joblib.load(Path(__file__).resolve().parent / "model.pkl")
sess = InferenceSession(Path(__file__).resolve().parent / "model.onnx")

test_data = pd.read_csv(Path(__file__).resolve().parents[1] / "uber.csv")

features = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "passenger_count"]
X_test = test_data[features].head(10).copy()

original_pred = model.predict(X_test)

ort_inputs = {}
for i, col in enumerate(features):
    if col == "passenger_count":
        ort_inputs[col] = X_test[col].values.reshape(-1, 1).astype(np.int64)
    else:
        ort_inputs[col] = X_test[col].values.reshape(-1, 1).astype(np.float32)

onnx_pred = sess.run(None, ort_inputs)[0]

print("Original predictions:", original_pred[:5])
print("ONNX predictions:", onnx_pred[:5].ravel())
print("Max difference:", np.abs(original_pred - onnx_pred.ravel()).max())