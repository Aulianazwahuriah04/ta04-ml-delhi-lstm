"""Memastikan prediksi NumPy sama dengan prediksi Keras asli.

Prediksi Keras pada data uji tersimpan di model/metrics.json (dibulatkan 2 desimal), jadi selisih
maksimum yang wajar adalah sekitar 0,005 °C. Jika TensorFlow terpasang, dibandingkan juga langsung dengan Keras.
    python verify_numpy.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import MinMax, NumpyForecaster

BASE = Path(__file__).parent
cfg = json.loads((BASE / "model" / "config.json").read_text())
T = int(cfg["timesteps"])
m = json.loads((BASE / "model" / "metrics.json").read_text())
s = pd.read_csv(BASE / "data" / "delhi_climate_clean.csv")[cfg["target"]].values.astype("float32")
n = len(s)
start = int(n * 0.70) + int(n * 0.15)

sc, net = MinMax(BASE / "model" / "scaler_params.json"), NumpyForecaster(BASE / "model" / "weights.npz")
scaled = sc.transform(s)
X = np.stack([scaled[i - T:i] for i in range(start, n)])[:, :, None].astype("float32")
pred = sc.inverse_transform(net.predict(X)).ravel()
ref = np.array(m["test_pred"])
diff = np.abs(pred - ref)
rmse = float(np.sqrt(((pred - np.array(m["test_actual"])) ** 2).mean()))
print(f"Data uji: {len(pred)} hari | selisih maks NumPy vs Keras (metrics.json): {diff.max():.4f} °C | rata-rata {diff.mean():.4f} °C")
print(f"RMSE NumPy: {rmse:.4f} | RMSE di metrics.json: {m['rmse']}")
ok = diff.max() <= 0.0051 and abs(rmse - m["rmse"]) < 0.002
try:
    from tensorflow.keras.models import load_model
    k = load_model(BASE / "model" / "lstm_model.h5", compile=False)
    d2 = np.abs(sc.inverse_transform(k.predict(X, verbose=0)).ravel() - pred).max()
    print(f"Langsung vs Keras: selisih maks {d2:.6f} °C")
    ok = ok and d2 < 1e-3
except Exception as e:
    print("(TensorFlow tidak tersedia atau gagal memuat model; perbandingan langsung dilewati:", type(e).__name__, ")")
print("HASIL:", "LULUS" if ok else "GAGAL, periksa export_weights.py")
