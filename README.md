# Dashboard Booking Hotel

Dashboard Streamlit 2 tab untuk client:

1. **Ringkasan** — kondisi booking 2015–2017 (KPI, pembatalan, musiman, revenue, asal tamu, rekomendasi).
2. **Prediksi 2018** — perkiraan angka bulanan 2018 + model risiko pembatalan per booking.

## Menjalankan

```bash
pip install -r requirements.txt

# 1. bangun artefak model & forecast (sekali, atau tiap data berubah)
python -m src.build_all

# 2. jalankan dashboard
streamlit run app.py
```

> Jika perintah `streamlit` tidak dikenali, pakai `python -m streamlit run app.py`.

`hotel_bookings.csv` harus ada di folder ini (dataset Hotel Booking Demand, Kaggle).

## Struktur

| Berkas | Isi |
|---|---|
| `app.py` | Dashboard Streamlit (2 tab) |
| `src/data.py` | Pemuatan, pembersihan, penyiapan fitur (dipakai bersama) |
| `src/train_cancellation.py` | Latih 3 model pembatalan → `models/` |
| `src/forecast_2018.py` | Forecast bulanan 2018 → `models/` |
| `src/build_all.py` | Jalankan kedua skrip di atas |
| `models/` | Artefak hasil (model `.joblib`, metrik `.json`, `.csv`) — dibuat oleh skrip |

## Bagian prediksi

### Model pembatalan (klasifikasi)

- **Target:** `is_canceled`. **Uji temporal:** latih 2015–2016, uji 2017 (bukan pengacakan biasa).
- **3 model:** `logistic` (baseline mudah dijelaskan), `tree` (aturan yang bisa dibaca),
  `gboost` = `HistGradientBoostingClassifier` (paling akurat, dipakai untuk skor).
- **Kolom bocor dibuang:** `reservation_status`, `reservation_status_date`, dan kolom
  yang baru final menjelang check-in (`assigned_room_type`, `booking_changes`,
  `days_in_waiting_list`). Lihat `src/data.py`.
- Performa realistis (tanpa kebocoran): ROC-AUC gboost ≈ **0,88**.
- **Peringatan:** `deposit_type = "Non Refund"` di data 99% berakhir batal — kemungkinan
  besar artefak pencatatan; validasi ke tim reservasi sebelum dipakai untuk kebijakan.

### Forecast 2018 (deret waktu)

- Data hanya **26 bulan** (Jul 2015 – Agu 2017) = 2 siklus musiman → **bukan** ML.
- **Holt-Winters** (tren + musiman) dengan pembanding **seasonal-naive**.
- Backtest (sembunyikan 6 bulan terakhir): MAPE ≈ **24–30%** → hasil dibaca sebagai
  **rentang**, bukan angka pasti.
- Angka revenue: EUR × kurs pada panel Filter (default 1 EUR = Rp 17.500).

## Catatan data

- Data 2015–2017, hotel di Portugal, **pra-COVID**. Forecast 2018 berasumsi tidak ada
  perubahan rezim pasar.
- Pembersihan: baris tanpa tamu & ADR janggal (<0 atau ≥5000) dibuang; `children`
  kosong → 0; `company` (94% kosong) dibuang.
# hotel-kaggle-
