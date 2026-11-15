"""Ekspor bobot model Keras (.h5) ke NumPy (.npz) dan parameter scaler ke JSON.

Jalankan sekali setiap kali model dilatih ulang (di komputer yang punya h5py dan scikit-learn):
    python export_weights.py
Keluaran: model/weights.npz dan model/scaler_params.json (dipakai aplikasi yang di-deploy).
"""
import json
from pathlib import Path

import h5py
import joblib
import numpy as np

BASE = Path(__file__).parent
H5 = BASE / "model" / "lstm_model.h5"

with h5py.File(H5, "r") as f:
    g = f["model_weights"]
    layers = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
    found = []
    for name in layers:
        arrs = {}
        def visit(_, obj, arrs=arrs):
            if isinstance(obj, h5py.Dataset):
                arrs[obj.name.split("/")[-1]] = obj[()]
        g[name].visititems(visit)
        if arrs:
            found.append((name, arrs))

lstm = [a for n, a in found if "recurrent_kernel" in a]
dense = [a for n, a in found if "recurrent_kernel" not in a]
assert len(lstm) == 2 and len(dense) == 2, f"Arsitektur tidak sesuai: {[n for n, _ in found]}"
out = {}
for k, a in enumerate(lstm, 1):
    out[f"lstm{k}_kernel"], out[f"lstm{k}_recurrent"], out[f"lstm{k}_bias"] = a["kernel"], a["recurrent_kernel"], a["bias"]
for k, a in enumerate(dense, 1):
    out[f"dense{k}_kernel"], out[f"dense{k}_bias"] = a["kernel"], a["bias"]
expected = {"lstm1_kernel": (1, 256), "lstm2_kernel": (64, 128), "dense1_kernel": (32, 16), "dense2_kernel": (16, 1)}
for k, s in expected.items():
    assert out[k].shape == s, f"{k}: bentuk {out[k].shape}, seharusnya {s}"
np.savez(BASE / "model" / "weights.npz", **{k: v.astype(np.float32) for k, v in out.items()})

sc = joblib.load(BASE / "model" / "scaler.pkl")
(BASE / "model" / "scaler_params.json").write_text(
    json.dumps({"data_min": float(sc.data_min_[0]), "data_max": float(sc.data_max_[0])}, indent=2))
print("OK: model/weights.npz dan model/scaler_params.json dibuat. Jalankan verify_numpy.py untuk memeriksa.")
