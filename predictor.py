"""Inferensi LSTM murni NumPy (tanpa TensorFlow) untuk aplikasi yang di-deploy.

Arsitektur sama dengan notebook: LSTM(64, return_sequences) -> LSTM(32) -> Dense(16, ReLU) -> Dense(1).
Dropout tidak dipakai saat inferensi. Urutan gerbang LSTM Keras: input, forget, cell (kandidat), output.
Bobot dibuat oleh export_weights.py dari model/lstm_model.h5 (file model Keras tetap disimpan sebagai arsip).
"""
import json
from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x, kernel, recurrent, bias, return_sequences):
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    outs = []
    for t in range(steps):
        z = x[:, t, :] @ kernel + h @ recurrent + bias
        i, f, g, o = (z[:, k * units:(k + 1) * units] for k in range(4))
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        outs.append(h)
    return np.stack(outs, axis=1) if return_sequences else h


class NumpyForecaster:
    """Antarmuka mirip Keras: predict(X) dengan X berbentuk (batch, timesteps, 1) pada skala 0-1."""

    def __init__(self, weights_path):
        w = np.load(weights_path)
        self.w = {k: w[k].astype(np.float32) for k in w.files}

    def predict(self, x, verbose=0):
        w = self.w
        x = np.asarray(x, dtype=np.float32)
        h = _lstm(x, w["lstm1_kernel"], w["lstm1_recurrent"], w["lstm1_bias"], True)
        h = _lstm(h, w["lstm2_kernel"], w["lstm2_recurrent"], w["lstm2_bias"], False)
        h = np.maximum(h @ w["dense1_kernel"] + w["dense1_bias"], 0.0)
        return h @ w["dense2_kernel"] + w["dense2_bias"]


class MinMax:
    """Pengganti MinMaxScaler(feature_range=(0, 1)) tanpa scikit-learn."""

    def __init__(self, params_path):
        p = json.loads(Path(params_path).read_text())
        self.lo, self.hi = float(p["data_min"]), float(p["data_max"])

    def transform(self, x):
        return (np.asarray(x, dtype=np.float32) - self.lo) / (self.hi - self.lo)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * (self.hi - self.lo) + self.lo
