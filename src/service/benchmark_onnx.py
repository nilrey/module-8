import time
import json
import numpy as np
from onnxruntime import InferenceSession
from pathlib import Path

sess = InferenceSession(Path(__file__).resolve().parent / "model.onnx")

test_input = {
    "pickup_latitude": np.array([[40.738]], dtype=np.float32),
    "pickup_longitude": np.array([[-73.999]], dtype=np.float32),
    "dropoff_latitude": np.array([[40.723]], dtype=np.float32),
    "dropoff_longitude": np.array([[-73.999]], dtype=np.float32),
    "passenger_count": np.array([[1]], dtype=np.int64),
}

latencies = []
runs = 1000

for _ in range(runs):
    start = time.perf_counter()
    sess.run(None, test_input)
    end = time.perf_counter()
    latencies.append(end - start)

latencies_sorted = sorted(latencies)

results = {
    "runs": runs,
    "min_s": min(latencies),
    "mean_s": np.mean(latencies),
    "p50_s": np.percentile(latencies, 50),
    "p95_s": np.percentile(latencies, 95),
    "p99_s": np.percentile(latencies, 99),
    "max_s": max(latencies),
}

output_path = Path(__file__).resolve().parent / "profile_onnx_NilSharafutdinov.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Результаты сохранены в {output_path}")
print(f"p99 latency: {results['p99_s']*1000:.3f} ms")

if results['p99_s'] > 0.02:
    print(f"Предупреждение: p99 latency превышает 20 мс ({results['p99_s']*1000:.3f} ms)")
else:
    print("Успех: p99 latency не превышает 20 мс")