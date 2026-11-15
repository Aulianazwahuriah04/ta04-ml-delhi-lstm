"""Aplikasi Flask: Peramalan Suhu Rata-Rata Harian Kota Delhi dengan LSTM.

Inferensi memakai NumPy (predictor.py), sehingga TensorFlow tidak diperlukan saat deploy."""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from predictor import MinMax, NumpyForecaster

BASE = Path(__file__).parent
WEIGHTS_PATH = BASE / "model" / "weights.npz"          # dibuat export_weights.py dari lstm_model.h5
SCALER_PATH = BASE / "model" / "scaler_params.json"
CONFIG_PATH = BASE / "model" / "config.json"
METRICS_PATH = BASE / "model" / "metrics.json"
DATA_PATH = BASE / "data" / "delhi_climate_clean.csv"

MAX_DAYS = 30

for p in (WEIGHTS_PATH, SCALER_PATH, CONFIG_PATH, METRICS_PATH, DATA_PATH):
    if not p.exists():
        raise FileNotFoundError(
            f"File {p.relative_to(BASE)} tidak ditemukan. "
            "Jalankan notebook training lalu export_weights.py (lihat README.md)."
        )

app = Flask(__name__)

# ---- Artefak statis (dimuat sekali saat start) ----
config = json.loads(CONFIG_PATH.read_text())
metrics = json.loads(METRICS_PATH.read_text())
scaler = MinMax(SCALER_PATH)
TIMESTEPS = int(config["timesteps"])
TARGET = config["target"]

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()

# ---- Model (NumPy, tanpa TensorFlow) dimuat saat pertama kali dibutuhkan ----
_model = None


def get_model():
    global _model
    if _model is None:
        _model = NumpyForecaster(WEIGHTS_PATH)
    return _model


@lru_cache(maxsize=MAX_DAYS)
def forecast(days: int):
    """Peramalan rekursif: prediksi 1 hari, lalu dipakai sebagai input hari berikutnya."""
    model = get_model()
    values = df[TARGET].values.reshape(-1, 1).astype("float32")
    window = scaler.transform(values[-TIMESTEPS:]).reshape(1, TIMESTEPS, 1).astype("float32")

    preds = []
    for _ in range(days):
        p = float(model.predict(window, verbose=0)[0, 0])
        preds.append(p)
        nxt = np.array(p, dtype="float32").reshape(1, 1, 1)
        window = np.concatenate([window[:, 1:, :], nxt], axis=1)

    inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    return [d.strftime("%Y-%m-%d") for d in dates], [round(float(v), 2) for v in inv]


class BadParam(Exception):
    pass


def int_arg(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except ValueError:
        raise BadParam(f"Parameter '{name}' harus berupa angka.")
    if not lo <= v <= hi:
        raise BadParam(f"Parameter '{name}' harus antara {lo} dan {hi}.")
    return v


@app.errorhandler(BadParam)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(Exception)
def server_error(e):
    from werkzeug.exceptions import HTTPException

    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    app.logger.exception("Kesalahan tidak terduga")
    return jsonify(error=f"{type(e).__name__}: {str(e)[:300]}"), 500


@app.route("/")
def index():
    return render_template("index.html", max_days=MAX_DAYS, timesteps=TIMESTEPS)


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/api/history")
def api_history():
    last = min(int_arg("last", 180, 7, 100000), len(df))
    s = df[TARGET].iloc[-last:]
    return jsonify(
        dates=[d.strftime("%Y-%m-%d") for d in s.index],
        values=[round(float(v), 2) for v in s.values],
        last_date=df.index[-1].strftime("%Y-%m-%d"),
        total_rows=len(df),
    )


@app.route("/api/forecast")
def api_forecast():
    days = int_arg("days", 7, 1, MAX_DAYS)
    dates, values = forecast(days)
    return jsonify(dates=dates, values=values, days=days,
                   last_date=df.index[-1].strftime("%Y-%m-%d"))



@app.route("/api/monthly")
def api_monthly():
    m = df[TARGET].groupby(df.index.month).mean()
    return jsonify(months=[int(i) for i in m.index], values=[round(float(v), 2) for v in m.values])

@app.route("/api/info")
def api_info():
    s_ = df[TARGET]
    return jsonify(
        first_date=df.index[0].strftime("%Y-%m-%d"), last_date=df.index[-1].strftime("%Y-%m-%d"),
        rows=len(df), timesteps=TIMESTEPS, target=TARGET,
        n_train=metrics.get("n_train"), n_val=metrics.get("n_val"), n_test=metrics.get("n_test"),
        epochs=metrics.get("epochs_trained"),
        t_min=round(float(s_.min()), 1), t_max=round(float(s_.max()), 1), t_mean=round(float(s_.mean()), 1),
    )


@app.route("/api/evaluation")
def api_evaluation():
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
