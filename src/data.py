"""
Pemuatan data, pembersihan, dan penyiapan fitur — dipakai bersama oleh skrip
training, skrip forecast, dan dashboard.

Penjelasan Bahasa Indonesia; nama variabel & fungsi Bahasa Inggris.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATA_PATH = Path(__file__).resolve().parents[1] / "hotel_bookings.csv"

# --------------------------------------------------------------------------- #
# Kolom yang TIDAK boleh dipakai sebagai fitur
# --------------------------------------------------------------------------- #
# "Bocor total": hanya terisi setelah booking selesai/batal — sama saja
# memberi tahu model jawabannya.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]

# "Bocor ringan": nilainya masih berubah setelah booking dibuat, jadi belum
# tentu diketahui saat kita ingin menilai booking baru. Dibuang demi aman.
POST_BOOKING_COLS = ["assigned_room_type", "booking_changes", "days_in_waiting_list"]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

TARGET = "is_canceled"

# Fitur untuk model pembatalan
NUMERIC_FEATURES = [
    "lead_time", "adr", "total_nights", "total_guests",
    "adults", "children", "babies",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "previous_cancellations", "previous_bookings_not_canceled",
    "is_repeated_guest", "required_car_parking_spaces",
    "total_of_special_requests", "arrival_date_week_number",
]
CATEGORICAL_FEATURES = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "customer_type", "agent",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Kategori dengan banyak nilai unik — nilai langka digabung jadi "Other"
HIGH_CARD_COLS = ["country", "agent"]


# --------------------------------------------------------------------------- #
# Pemuatan & pembersihan
# --------------------------------------------------------------------------- #
def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan ringan yang aman untuk semua keperluan."""
    df = df.copy()

    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("UNK")
    df["agent"] = df["agent"].fillna(0).astype("int64").astype(str)  # "0" = tanpa agen
    df = df.drop(columns=["company"], errors="ignore")  # ~94% kosong

    # Buang baris tak masuk akal
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] < 5000)]

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-" + df["arrival_date_month"]
        + "-" + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    df["arrival_month_start"] = df["arrival_date"].values.astype("datetime64[M]")
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["booking_value"] = df["adr"] * df["total_nights"]  # estimasi nilai transaksi (EUR)

    return df.reset_index(drop=True)


def load_clean(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


# --------------------------------------------------------------------------- #
# Transformer: gabungkan kategori langka -> "Other"
# --------------------------------------------------------------------------- #
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Pelajari `top_k` kategori tersering per kolom saat fit; sisanya -> 'Other'.

    Menjaga jumlah kolom one-hot tetap terkendali dan membuat model tahan
    terhadap nilai kategori baru yang belum pernah dilihat.
    """

    def __init__(self, columns: list[str], top_k: int = 12):
        self.columns = columns
        self.top_k = top_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = {
            c: set(X[c].astype(str).value_counts().head(self.top_k).index)
            for c in self.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns:
            s = X[c].astype(str)
            X[c] = np.where(s.isin(self.keep_[c]), s, "Other")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


# --------------------------------------------------------------------------- #
# Fitur untuk model pembatalan
# --------------------------------------------------------------------------- #
def make_cancellation_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Ambil matriks fitur X dan target y dari data yang sudah dibersihkan."""
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def temporal_split(df: pd.DataFrame, train_years=(2015, 2016), test_years=(2017,)):
    """Pisah berdasarkan tahun kedatangan — meniru situasi nyata: latih dari
    masa lalu, uji pada periode berikutnya (bukan pengacakan biasa)."""
    train = df[df["arrival_date_year"].isin(train_years)]
    test = df[df["arrival_date_year"].isin(test_years)]
    return train, test


# --------------------------------------------------------------------------- #
# Agregasi bulanan untuk forecast
# --------------------------------------------------------------------------- #
def monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkas ke level bulan berdasarkan tanggal kedatangan."""
    g = df.groupby("arrival_month_start")
    out = pd.DataFrame({
        "bookings": g.size(),
        "stays": g.apply(lambda d: int((d["is_canceled"] == 0).sum()), include_groups=False),
        "room_nights": g.apply(lambda d: int(d.loc[d["is_canceled"] == 0, "total_nights"].sum()), include_groups=False),
        "revenue_eur": g.apply(lambda d: float(d.loc[d["is_canceled"] == 0, "booking_value"].sum()), include_groups=False),
    })
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out["cancel_rate"] = 1 - out["stays"] / out["bookings"]
    return out
