import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from pathlib import Path

model = joblib.load(Path(__file__).resolve().parent / "model.pkl")
print(f"Загружена модель: {type(model)}")

initial_types = [
    ("pickup_latitude", FloatTensorType([None, 1])),
    ("pickup_longitude", FloatTensorType([None, 1])),
    ("dropoff_latitude", FloatTensorType([None, 1])),
    ("dropoff_longitude", FloatTensorType([None, 1])),
    ("passenger_count", Int64TensorType([None, 1])),
]

onnx_model = convert_sklearn(
    model, 
    initial_types=initial_types, 
    name="taxi_fare_model"
)

output_path = Path(__file__).resolve().parent / "model.onnx"
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Модель сохранена в {output_path}")