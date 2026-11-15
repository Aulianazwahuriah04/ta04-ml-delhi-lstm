# SuhuDelhi - Peramalan Suhu Rata-Rata Harian Kota Delhi

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn)

Aplikasi web berbasis Flask untuk meramalkan **suhu rata-rata harian di Kota Delhi** 1 sampai 30 hari ke depan, menggunakan model **LSTM** (Long Short-Term Memory). Model membaca 30 hari suhu terakhir untuk menebak suhu hari berikutnya, lalu mengulanginya secara rekursif sampai jumlah hari yang dipilih. Proyek ini adalah Tugas Akhir 4 (TA-04) mata kuliah Pembelajaran Mesin.

---

## Fitur Aplikasi

- **Prakiraan 1-30 Hari** - pilih jumlah hari lewat slider atau tombol cepat, hasilnya tampil sebagai kartu berwarna sesuai suhu
- **Grafik Interaktif** - riwayat suhu dan prakiraan dalam bentuk garis atau batang, rentang riwayat bisa diatur, klik kartu untuk menyorot titiknya di grafik
- **Simulasi LSTM** - animasi jendela 30 hari yang bergeser setiap kali model menebak satu hari
- **Pola Musiman** - rata-rata suhu per bulan dari seluruh data historis
- **Performa Model** - RMSE, MAE, perbandingan dengan tebakan "sama dengan kemarin", grafik aktual vs prediksi, dan sebaran selisih
- **Toleransi Galat** - geser batas selisih untuk melihat persentase prediksi yang masih di dalam batas itu
- **Satuan dan Tema** - ganti satuan °C/°F, mode terang atau gelap, serta unduh hasil prakiraan sebagai CSV

---

## Teknologi

| Komponen | Teknologi |
|---|---|
| Backend | Flask 3, Gunicorn |
| Machine Learning | TensorFlow/Keras (LSTM), scikit-learn 1.6.1 (MinMaxScaler) |
| Data Processing | Pandas, NumPy |
| Visualisasi | Chart.js (aplikasi), Matplotlib (notebook) |
| Penyimpanan Model | HDF5 (`.h5`) dan Joblib |
| Frontend | HTML (Jinja2), CSS, JavaScript |

---

## Dataset

- **Sumber:** [Kaggle - Daily Weather Time Series Data, Delhi, India](https://www.kaggle.com/datasets/yug201/daily-climate-time-series-data-delhi-india)
- **Rentang waktu:** April 2013 sampai November 2024 (data harian)
- **Variabel yang dipakai:** suhu rata-rata harian (model univariat). Kolom tanggal dan kolom suhu dideteksi otomatis oleh notebook; jika hanya ada suhu maksimum dan minimum, rata-ratanya dihitung dari keduanya
- **Pembersihan:** file CSV di `data/` digabung, tanggal ganda dibuang, tanggal yang hilang dilengkapi, dan nilai kosong diisi dengan interpolasi berbasis waktu
- **Pembagian data (urut waktu):** 70% latih, 15% validasi, 15% uji. `MinMaxScaler` hanya di-*fit* pada data latih agar tidak terjadi kebocoran data

---

## Arsitektur Model

| Bagian | Pengaturan |
|---|---|
| Input | 30 hari terakhir (`TIMESTEPS = 30`), 1 fitur |
| Lapisan | LSTM(64) → Dropout(0,2) → LSTM(32) → Dropout(0,2) → Dense(16, ReLU) → Dense(1) |
| Optimizer / Loss | Adam / Mean Squared Error |
| Pelatihan | Maksimal 100 epoch, batch 32, `EarlyStopping` (patience 10) |
| Peramalan | Rekursif: tebakan satu hari dipakai lagi sebagai input hari berikutnya |

---

## Performa Model

Evaluasi pada data uji (15% terakhir dari data). Isi tabel ini dengan hasil dari sel evaluasi di notebook setelah training.

| Metrik | Nilai |
|---|---|
| RMSE data uji | ... °C |
| MAE data uji | ... °C |
| RMSE tebakan "sama dengan kemarin" | ... °C |

Tebakan "sama dengan kemarin" dipakai sebagai pembanding sederhana (baseline). Suhu harian sangat mirip dengan hari sebelumnya, sehingga baseline ini sulit dikalahkan. Galat peramalan jangka panjang juga menumpuk karena setiap prediksi dipakai lagi sebagai input, jadi anggap hasil di ujung rentang sebagai gambaran tren, bukan angka pasti.

---

## Instalasi Lokal

### 1. Clone repository

```bash
git clone https://github.com/Alnazh/ta04-delhi-lstm-forecasting.git
cd ta04-delhi-lstm-forecasting
```

### 2. Buat virtual environment

```bash
python -m venv .venv

# Windows (PowerShell, jika muncul error execution policy:
# Set-ExecutionPolicy -Scope Process Bypass)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
python app.py
```

Buka browser di `http://127.0.0.1:5000`

Folder `model/` dan `data/delhi_climate_clean.csv` sudah ikut di repository, jadi tidak perlu training ulang. Jika ingin melatih ulang atau memakai data baru, lihat bagian **Training Model**.

---

## Struktur Project

```
ta04-delhi-lstm-forecasting/
├── app.py                        # Entry point Flask (halaman dan API)
├── notebook_training.ipynb       # EDA, pelatihan, evaluasi, penyimpanan model
├── predictor.py                  # Inferensi LSTM dengan NumPy (tanpa TensorFlow)
├── export_weights.py             # Ekspor bobot .h5 -> weights.npz (jalankan setelah training)
├── verify_numpy.py               # Cek prediksi NumPy sama dengan Keras
├── requirements.txt              # Dependensi aplikasi yang di-deploy (ringan)
├── requirements-train.txt        # Dependensi training/ekspor (TensorFlow, scikit-learn, h5py)
├── .python-version
├── .gitignore
│
├── data/
│   ├── (CSV dari Kaggle)         # Dataset mentah, tidak di-commit
│   └── delhi_climate_clean.csv   # Data bersih untuk aplikasi
│
├── model/
│   ├── lstm_model.h5             # Model LSTM Keras tersimpan (arsip)
│   ├── scaler.pkl                # MinMaxScaler tersimpan (arsip)
│   ├── weights.npz               # Bobot untuk inferensi NumPy (dipakai aplikasi)
│   ├── scaler_params.json        # Min/maks scaler (dipakai aplikasi)
│   ├── config.json               # Timesteps dan nama kolom target
│   └── metrics.json              # Metrik evaluasi dan hasil prediksi data uji
│
├── templates/
│   └── index.html                # Halaman web (HTML + Jinja2)
│
├── static/
│   ├── css/
│   │   └── style.css             # Gaya tampilan
│   └── js/
│       ├── theme.js              # Tema terang/gelap (dimuat paling awal)
│       └── app.js                # Grafik, interaksi, pemanggilan API
│
└── docs/
    └── screenshots/              # Tangkapan layar aplikasi
```

---

## Training Model

1. Unduh dataset dari Kaggle (tautan di bagian **Dataset**), ekstrak, lalu taruh file CSV-nya di folder `data/`.
2. Buka `notebook_training.ipynb` (Jupyter atau Google Colab). Di Colab, buat folder `data` dan upload file CSV ke sana.
3. Jalankan sel pertama untuk memasang `scikit-learn==1.6.1`, lalu **Run all**. Jika versi di sel terakhir bukan 1.6.1, restart runtime dan jalankan ulang.
4. Jika kolom tidak dikenali otomatis, isi `DATE_COL` atau `TEMP_COL` di sel pengaturan.
5. Unduh hasilnya dan taruh di proyek: folder `model/` dan file `data/delhi_climate_clean.csv`.
6. Di komputer lokal, pasang `pip install -r requirements-train.txt`, lalu jalankan `python export_weights.py` (membuat `model/weights.npz` dan `model/scaler_params.json`) dan `python verify_numpy.py` (harus menampilkan **LULUS**).

Aplikasi yang di-deploy tidak memakai TensorFlow: prediksi dihitung dengan NumPy dari bobot hasil ekspor. Prediksinya sama dengan model Keras (selisih maksimum 0,005 °C pada data uji, setara pembulatan 2 desimal di `metrics.json`). Gunakan versi library yang sama seperti di `requirements-train.txt` saat training.

---

## Endpoint API

| Endpoint | Fungsi |
|---|---|
| `/` | Halaman web |
| `/api/history?last=180` | Data historis (N hari terakhir) |
| `/api/forecast?days=7` | Prediksi 1-30 hari ke depan |
| `/api/monthly` | Rata-rata suhu per bulan |
| `/api/info` | Info dataset dan pembagian data |
| `/api/evaluation` | RMSE, MAE, baseline, dan data aktual vs prediksi pada data uji |
| `/health` | Cek status server |

---

## Tampilan Aplikasi

> Taruh tangkapan layar asli di folder `docs/screenshots/` (nama file tetap sama).

### Beranda
![Beranda](docs/screenshots/01-beranda.png)

### Prediksi
![Prediksi](docs/screenshots/02-prediksi.png)

### Performa Model
![Performa Model](docs/screenshots/03-performa-model.png)

---

## Cara Penggunaan

1. Buka halaman utama, lalu pergi ke bagian **Prediksi**
2. Pilih jumlah hari ke depan (1-30) dengan slider atau tombol cepat
3. Atur rentang riwayat dan bentuk grafik (garis atau batang) sesuai kebutuhan
4. Klik salah satu kartu hari untuk menyorot titiknya di grafik, atau gunakan tombol **Unduh CSV** untuk menyimpan hasil
5. Buka bagian **Performa Model** untuk melihat seberapa akurat model pada data uji

---

## Deployment

Karena inferensi memakai NumPy, aplikasi hanya butuh Flask, pandas, dan NumPy (sekitar 100-200 MB terpasang), sehingga muat di hosting kecil termasuk paket gratis PythonAnywhere. Server produksi:

```bash
gunicorn app:app --workers 1 --threads 2 --timeout 180
```

### PythonAnywhere (langkah ringkas)

1. Di tab **Consoles**, buka Bash lalu ambil kode: `git clone https://github.com/Alnazh/ta04-delhi-lstm-forecasting.git` (atau upload zip lewat tab **Files**). Pastikan folder `model/` (termasuk `weights.npz` dan `scaler_params.json`) dan `data/delhi_climate_clean.csv` ikut ada.
2. Buat virtualenv dan pasang dependensi (ganti `python3.10` dengan versi yang tersedia di akunmu):
   ```bash
   cd ~/ta04-delhi-lstm-forecasting
   python3.10 -m venv ~/ta04-venv
   source ~/ta04-venv/bin/activate
   pip install -r requirements.txt
   ```
3. Di tab **Web**, pilih *Add a new web app* > *Manual configuration* > versi Python yang sama. Isi **Virtualenv** dengan `/home/USERNAME/ta04-venv`.
4. Klik file WSGI di tab **Web**, hapus isinya, lalu isi:
   ```python
   import sys
   path = "/home/USERNAME/ta04-delhi-lstm-forecasting"
   if path not in sys.path:
       sys.path.insert(0, path)
   from app import app as application
   ```
5. Klik **Reload**, lalu buka `https://USERNAME.pythonanywhere.com/health`. Jika muncul `{"status":"ok"}`, aplikasi siap.

Catatan paket gratis: hanya 1 web app, kuota 100 detik CPU per hari, disk 512 MiB, dan web app perlu diperpanjang secara berkala. Jika muncul error, lihat **Error log** di tab **Web**.

### Hosting lain (Render, Railway)

Pakai `requirements.txt` yang sama dan perintah `gunicorn` di atas; setelah deploy cek `/health`.
