"""
Perkiraan angka bulanan tahun 2018 (booking, tamu menginap, room-night, revenue).

Data hanya 26 bulan (Jul 2015 - Agu 2017) = 2 siklus musiman. Terlalu pendek
untuk machine learning; dipakai metode deret waktu klasik:

  - holt_winters   : tren + pola musiman (statsmodels)
  - seasonal_naive : nilai bulan yang sama tahun lalu x pertumbuhan tahunan
                     -> pembanding sederhana yang transparan

Hasil disajikan sebagai rentang, bukan angka pasti. Backtest (menyembunyikan
6 bulan terakhir) dipakai untuk mengukur besar kesalahan.

Jalankan:  python -m src.forecast_2018
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.data import load_clean, monthly_series

warnings.simplefilter("ignore")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

METRICS = ["bookings", "stays", "room_nights", "revenue_eur"]
HORIZON_END = "2018-12-01"
SEED = 42


# --------------------------------------------------------------------------- #
# Metode forecast
# --------------------------------------------------------------------------- #
def _known_init(y: pd.Series, sp: int = 12) -> dict:
    """Hitung nilai awal level/tren/musiman sendiri — statsmodels butuh >2 siklus
    penuh untuk cara otomatis, sedangkan data kita hanya ~2 siklus."""
    first = y.iloc[:sp]
    level = float(first.mean())
    if len(y) >= 2 * sp:
        trend = float(y.iloc[sp:2 * sp].mean() - level) / sp
    else:
        slope = np.polyfit(np.arange(len(y)), y.values, 1)[0]
        trend = float(slope)
    seasonal = (first.values / level)
    seasonal = seasonal / seasonal.mean()  # rata-rata ~1 (multiplikatif)
    return {"initial_level": level, "initial_trend": trend,
            "initial_seasonal": seasonal}


def holt_winters(y: pd.Series, steps: int) -> pd.DataFrame:
    init = _known_init(y)
    fit = ExponentialSmoothing(
        y, trend="add", seasonal="mul", seasonal_periods=12,
        initialization_method="known", **init,
    ).fit()
    mean = fit.forecast(steps)

    sim = fit.simulate(steps, repetitions=3000, anchor="end", random_state=SEED)
    lo80 = sim.quantile(0.10, axis=1)
    hi80 = sim.quantile(0.90, axis=1)
    lo95 = sim.quantile(0.025, axis=1)
    hi95 = sim.quantile(0.975, axis=1)

    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.DataFrame(
        {"yhat": mean.values, "lo80": lo80.values, "hi80": hi80.values,
         "lo95": lo95.values, "hi95": hi95.values},
        index=idx,
    ).clip(lower=0)


def seasonal_naive(y: pd.Series, steps: int) -> pd.DataFrame:
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")

    by_month = {m: y[y.index.month == m] for m in range(1, 13)}
    # pertumbuhan tahunan rata-rata (rasio antar tahun untuk bulan yang sama)
    ratios = []
    for m, s in by_month.items():
        if len(s) >= 2:
            ratios.extend((s.values[1:] / s.values[:-1]).tolist())
    growth = float(np.mean(ratios)) if ratios else 1.0

    out = []
    for d in idx:
        s = by_month[d.month]
        base_year = s.index[-1].year
        base_val = s.values[-1]
        years_ahead = d.year - base_year
        out.append(base_val * (growth ** years_ahead))
    yhat = np.array(out)
    return pd.DataFrame(
        {"yhat": yhat, "lo80": yhat * 0.85, "hi80": yhat * 1.15,
         "lo95": yhat * 0.75, "hi95": yhat * 1.25},
        index=idx,
    ).clip(lower=0)


METHODS = {"holt_winters": holt_winters, "seasonal_naive": seasonal_naive}


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def backtest(y: pd.Series, holdout: int = 6) -> dict:
    train, test = y.iloc[:-holdout], y.iloc[-holdout:]
    scores = {}
    for name, fn in METHODS.items():
        fc = fn(train, holdout)
        scores[name] = round(mape(test.values, fc["yhat"].values), 1)
    return scores


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Memuat data…")
    df = load_clean()
    ms = monthly_series(df)
    steps = (pd.Timestamp(HORIZON_END).to_period("M") - ms.index[-1].to_period("M")).n

    hist_rows, fc_rows, bt = [], [], {}
    for metric in METRICS:
        y = ms[metric].astype(float)
        y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp(how="start")
        y = y.asfreq("MS")

        for d, v in y.items():
            hist_rows.append({"month": d.date().isoformat(), "metric": metric, "value": float(v)})

        bt[metric] = backtest(y)

        for name, fn in METHODS.items():
            fc = fn(y, steps)
            for d, r in fc.iterrows():
                fc_rows.append({
                    "month": d.date().isoformat(), "metric": metric, "model": name,
                    "yhat": round(r.yhat, 2), "lo80": round(r.lo80, 2), "hi80": round(r.hi80, 2),
                    "lo95": round(r.lo95, 2), "hi95": round(r.hi95, 2),
                })

    pd.DataFrame(hist_rows).to_csv(MODELS_DIR / "forecast_history.csv", index=False)
    fc_df = pd.DataFrame(fc_rows)
    fc_df.to_csv(MODELS_DIR / "forecast_2018.csv", index=False)

    # Ringkasan total 2018 (model holt_winters)
    y2018 = fc_df[(fc_df.model == "holt_winters") & fc_df.month.str.startswith("2018")]
    summary_2018 = y2018.groupby("metric")[["yhat", "lo80", "hi80"]].sum().round(0).to_dict("index")

    (MODELS_DIR / "forecast_metrics.json").write_text(json.dumps({
        "history_months": int(len(ms)),
        "history_range": [ms.index[0].date().isoformat(), ms.index[-1].date().isoformat()],
        "horizon_steps": int(steps),
        "backtest_mape_pct": bt,
        "backtest_holdout_months": 6,
        "total_2018_holt_winters": summary_2018,
        "note": (
            "Data pra-COVID, hotel di Portugal, hanya 2 siklus musiman. "
            "Angka 2018 adalah ekstrapolasi — perlakukan sebagai rentang, bukan target pasti."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBacktest MAPE (holdout 6 bulan terakhir):")
    print(pd.DataFrame(bt).T.to_string())
    print("\nPerkiraan total 2018 (Holt-Winters):")
    for m, v in summary_2018.items():
        print(f"  {m:12s} {v['yhat']:,.0f}   (80% PI {v['lo80']:,.0f} – {v['hi80']:,.0f})")
    print("\nArtefak tersimpan di:", MODELS_DIR)


if __name__ == "__main__":
    main()
