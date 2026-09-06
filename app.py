"""
Dashboard Booking Hotel — laporan untuk client.

Dua bagian (tab):
  1. Ringkasan       — kondisi 2015-2017
  2. Prediksi 2018   — forecast angka bulanan + model risiko pembatalan

Jalankan:  streamlit run app.py
Sebelum tab Prediksi bisa dipakai, jalankan dulu:
  python -m src.train_cancellation
  python -m src.forecast_2018
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import joblib  # noqa: E402
from src.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES  # noqa: E402,F401
from src.data import RareCategoryGrouper  # noqa: E402,F401  (dibutuhkan joblib saat unpickle)

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "hotel_bookings.csv"
MODELS_DIR = ROOT / "models"
KURS_DEFAULT = 17_500  # 1 EUR = Rp ... (ADR dataset dalam EUR; hotel di Portugal)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
C_CITY = "#2a78d6"       # slot 1 — biru
C_RESORT = "#eb6834"     # slot 2 — oranye
C_SEQ = "#2a78d6"        # sequential 1-hue
C_BATAL = "#d03b3b"      # status: critical
C_REALISASI = "#0ca30c"  # status: good

BULAN_ID = {
    "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
    "May": "Mei", "June": "Jun", "July": "Jul", "August": "Agu",
    "September": "Sep", "October": "Okt", "November": "Nov", "December": "Des",
}
BULAN_URUT = list(BULAN_ID.values())

METRIK_LABEL = {
    "bookings": "Jumlah booking",
    "stays": "Tamu menginap (setelah pembatalan)",
    "room_nights": "Room-night",
    "revenue_eur": "Estimasi revenue",
}

st.set_page_config(page_title="Dashboard Booking Hotel", page_icon="🏨", layout="wide")


# --------------------------------------------------------------------------- #
# Data & artefak
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Memuat data…")
def muat_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Tidak diketahui")
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] < 5000)]
    df["total_malam"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["tanggal"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"]
        + "-" + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    df["periode"] = df["tanggal"].dt.to_period("M").dt.to_timestamp()
    df["bulan_id"] = df["arrival_date_month"].map(BULAN_ID)
    df["nilai_booking_eur"] = df["adr"] * df["total_malam"]
    bins = [-1, 7, 30, 90, 180, 10_000]
    labels = ["0–7 hari", "8–30 hari", "31–90 hari", "91–180 hari", "180+ hari"]
    df["kelompok_lead_time"] = pd.cut(df["lead_time"], bins=bins, labels=labels)
    df["deposit_id"] = df["deposit_type"].map(
        {"No Deposit": "Tanpa Deposit", "Non Refund": "Non-Refund", "Refundable": "Refundable"}
    )
    return df


@st.cache_data
def load_json(name: str):
    p = MODELS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_csv(name: str):
    p = MODELS_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_text(name: str):
    p = MODELS_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_resource
def load_model(name: str):
    p = MODELS_DIR / name
    return joblib.load(p) if p.exists() else None


# --------------------------------------------------------------------------- #
# Format
# --------------------------------------------------------------------------- #
def fmt_rp(x: float, desimal: int | None = None) -> str:
    x = float(x)
    a = abs(x)
    if a >= 1e12:
        s = f"Rp {x / 1e12:,.{2 if desimal is None else desimal}f} T"
    elif a >= 1e9:
        s = f"Rp {x / 1e9:,.{2 if desimal is None else desimal}f} M"
    elif a >= 1e6:
        s = f"Rp {x / 1e6:,.{1 if desimal is None else desimal}f} jt"
    elif a >= 1e3:
        s = f"Rp {x / 1e3:,.0f} rb"
    else:
        s = f"Rp {x:,.0f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_ribu(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def styl(fig: go.Figure, tinggi: int = 360) -> go.Figure:
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=16), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID, color=MUTED, title_font_color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=MUTED, title_font_color=MUTED)
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
df_all = muat_data()

st.sidebar.header("Filter")
st.sidebar.caption("Berlaku untuk tab **Ringkasan**. Tab Prediksi memakai seluruh data.")
hotel_pil = st.sidebar.multiselect(
    "Hotel", ["City Hotel", "Resort Hotel"], default=["City Hotel", "Resort Hotel"]
)
tahun_pil = st.sidebar.multiselect(
    "Tahun kedatangan", [2015, 2016, 2017], default=[2015, 2016, 2017]
)
kurs = st.sidebar.number_input(
    "Kurs 1 EUR = Rp", min_value=1_000, max_value=100_000, value=KURS_DEFAULT, step=500,
    help="ADR pada dataset berdenominasi EUR (hotel di Portugal). Angka Rupiah adalah estimasi.",
)

df = df_all[df_all["hotel"].isin(hotel_pil) & df_all["arrival_date_year"].isin(tahun_pil)].copy()
df["nilai_booking_rp"] = df["nilai_booking_eur"] * kurs

st.title("Dashboard Booking Hotel")
tab_ringkasan, tab_prediksi = st.tabs(["📊 Ringkasan", "🔮 Prediksi 2018"])


# =========================================================================== #
# TAB 1 — RINGKASAN
# =========================================================================== #
def render_ringkasan() -> None:
    if df.empty:
        st.warning("Tidak ada data untuk filter ini.")
        return

    rentang = f"{df['tanggal'].min():%b %Y} – {df['tanggal'].max():%b %Y}"
    st.caption(
        f"Periode kedatangan {rentang} · {fmt_ribu(len(df))} booking · "
        f"sumber: dataset Hotel Booking Demand (Kaggle)"
    )

    # 1. KPI
    total_booking = len(df)
    rate_batal = df["is_canceled"].mean()
    adr_avg = df["adr"].mean() * kurs
    lead_avg = df["lead_time"].mean()
    rev_realisasi = df.loc[df["is_canceled"] == 0, "nilai_booking_rp"].sum()
    rev_bocor = df.loc[df["is_canceled"] == 1, "nilai_booking_rp"].sum()

    k = st.columns(6)
    k[0].metric("Total booking", fmt_ribu(total_booking))
    k[1].metric("Tingkat pembatalan", f"{rate_batal:.1%}")
    k[2].metric("ADR rata-rata", fmt_rp(adr_avg, 1))
    k[3].metric("Estimasi revenue terealisasi", fmt_rp(rev_realisasi))
    k[4].metric("Estimasi revenue hilang (batal)", fmt_rp(rev_bocor))
    k[5].metric("Rata-rata lead time", f"{lead_avg:.0f} hari")
    st.caption(
        "Estimasi revenue = ADR × jumlah malam menginap. "
        "“Revenue hilang” = nilai yang sama untuk booking yang akhirnya dibatalkan."
    )
    st.divider()

    # 2. Masalah utama
    st.subheader(f"Masalah utama: {rate_batal:.0%} booking berakhir dibatalkan")
    st.markdown(
        "Lebih dari sepertiga reservasi tidak menjadi tamu yang menginap. "
        "Ini titik kebocoran terbesar — perbaikan kecil di sini berdampak langsung ke okupansi dan pendapatan."
    )
    pv = df.groupby([df["periode"], "hotel"])["is_canceled"].mean().mul(100).reset_index()
    fig = go.Figure()
    for nm, col in [("City Hotel", C_CITY), ("Resort Hotel", C_RESORT)]:
        d = pv[pv["hotel"] == nm]
        if not d.empty:
            fig.add_trace(go.Scatter(
                x=d["periode"], y=d["is_canceled"], name=nm, mode="lines",
                line=dict(width=2, color=col),
                hovertemplate="%{x|%b %Y}<br>" + nm + ": %{y:.0f}%<extra></extra>",
            ))
    fig.add_hline(y=rate_batal * 100, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"rata-rata {rate_batal:.0%}", annotation_position="top left")
    fig.update_yaxes(title="% dibatalkan", ticksuffix="%")
    st.plotly_chart(styl(fig, 380).update_layout(title="Tingkat pembatalan per bulan"),
                    width="stretch")
    st.divider()

    # 3. Kapan tamu datang
    st.subheader("Kapan tamu datang: puncak di musim panas, City Hotel lebih fluktuatif")
    c1, c2 = st.columns(2)
    vol = df.groupby([df["periode"], "hotel"]).size().reset_index(name="n")
    fig = go.Figure()
    for nm, col in [("City Hotel", C_CITY), ("Resort Hotel", C_RESORT)]:
        d = vol[vol["hotel"] == nm]
        fig.add_trace(go.Bar(x=d["periode"], y=d["n"], name=nm, marker_color=col,
                             hovertemplate="%{x|%b %Y}<br>" + nm + ": %{y} booking<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="jumlah booking")
    c1.plotly_chart(styl(fig).update_layout(title="Volume booking per bulan"), width="stretch")

    musim = df.groupby(["bulan_id", "hotel"]).size().reset_index(name="n")
    musim["bulan_id"] = pd.Categorical(musim["bulan_id"], categories=BULAN_URUT, ordered=True)
    musim = musim.sort_values("bulan_id")
    fig = go.Figure()
    for nm, col in [("City Hotel", C_CITY), ("Resort Hotel", C_RESORT)]:
        d = musim[musim["hotel"] == nm]
        fig.add_trace(go.Bar(x=d["bulan_id"], y=d["n"], name=nm, marker_color=col,
                             hovertemplate=nm + " · %{x}: %{y} booking<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="total booking (semua tahun)")
    c2.plotly_chart(styl(fig).update_layout(title="Pola musiman per bulan kalender"), width="stretch")
    st.divider()

    # 4. Siapa yang membatalkan
    st.subheader("Siapa yang membatalkan — dan kenapa")
    c1, c2 = st.columns(2)
    seg = (df.groupby("market_segment")
           .agg(rate=("is_canceled", "mean"), n=("is_canceled", "size")).reset_index())
    seg = seg[seg["n"] >= 100].sort_values("rate")
    fig = go.Figure(go.Bar(
        x=seg["rate"] * 100, y=seg["market_segment"], orientation="h", marker_color=C_SEQ,
        text=[f"{v:.0f}%" for v in seg["rate"] * 100], textposition="outside",
        hovertemplate="%{y}: %{x:.0f}% dibatalkan<extra></extra>",
    ))
    fig.update_xaxes(title="% dibatalkan", ticksuffix="%", range=[0, 100])
    c1.plotly_chart(styl(fig).update_layout(title="Tingkat pembatalan per segmen pasar"), width="stretch")

    lt = df.groupby("kelompok_lead_time", observed=True)["is_canceled"].mean().mul(100).reset_index()
    fig = go.Figure(go.Bar(
        x=lt["kelompok_lead_time"].astype(str), y=lt["is_canceled"], marker_color=C_SEQ,
        text=[f"{v:.0f}%" for v in lt["is_canceled"]], textposition="outside",
        hovertemplate="%{x}: %{y:.0f}% dibatalkan<extra></extra>",
    ))
    fig.update_yaxes(title="% dibatalkan", ticksuffix="%", range=[0, 100])
    c2.plotly_chart(styl(fig).update_layout(title="Makin jauh hari pesan, makin sering batal"),
                    width="stretch")

    dep = (df.groupby("deposit_id")
           .agg(rate=("is_canceled", "mean"), n=("is_canceled", "size")).reset_index().sort_values("rate"))
    fig = go.Figure(go.Bar(
        x=dep["deposit_id"], y=dep["rate"] * 100,
        marker_color=[C_BATAL if r > 0.5 else C_SEQ for r in dep["rate"]],
        text=[f"{v:.0f}%  ({fmt_ribu(n)} booking)" for v, n in zip(dep["rate"] * 100, dep["n"])],
        textposition="outside", hovertemplate="%{x}: %{y:.0f}% dibatalkan<extra></extra>",
    ))
    fig.update_yaxes(title="% dibatalkan", ticksuffix="%", range=[0, 105])
    st.plotly_chart(styl(fig, 340).update_layout(
        title="Anomali: booking “Non-Refund” justru hampir selalu dibatalkan"), width="stretch")
    st.caption(
        "Indikasi kuat bahwa deposit non-refund dipakai untuk kanal/booking berisiko tinggi "
        "(mis. jaminan grup atau OTA tertentu) — perlu dicek ke tim reservasi, bukan diartikan mentah."
    )
    st.divider()

    # 5. Pendapatan
    st.subheader("Estimasi pendapatan dan kebocorannya")
    st.markdown(
        f"Dengan kurs 1 EUR = Rp {kurs:,.0f}".replace(",", ".")
        + f", estimasi **{fmt_rp(rev_bocor)}** nilai booking batal sepanjang periode — "
        f"setara **{rev_bocor / (rev_realisasi + rev_bocor):.0%}** dari total nilai reservasi."
    )
    c1, c2 = st.columns(2)
    adr_bulan = df.groupby([df["periode"], "hotel"])["adr"].mean().mul(kurs).reset_index()
    fig = go.Figure()
    for nm, col in [("City Hotel", C_CITY), ("Resort Hotel", C_RESORT)]:
        d = adr_bulan[adr_bulan["hotel"] == nm]
        fig.add_trace(go.Scatter(x=d["periode"], y=d["adr"], name=nm, mode="lines",
                                 line=dict(width=2, color=col),
                                 hovertemplate="%{x|%b %Y}<br>" + nm + ": %{y:,.0f}<extra></extra>"))
    fig.update_yaxes(title="ADR rata-rata (Rp)", tickprefix="Rp ")
    c1.plotly_chart(styl(fig).update_layout(title="ADR bergerak naik, puncak tiap musim panas"),
                    width="stretch")

    rev = df.assign(status=np.where(df["is_canceled"] == 1, "Batal (hilang)", "Terealisasi"))
    rev = rev.groupby([rev["periode"], "status"])["nilai_booking_rp"].sum().reset_index()
    fig = go.Figure()
    for nm, col in [("Terealisasi", C_REALISASI), ("Batal (hilang)", C_BATAL)]:
        d = rev[rev["status"] == nm]
        fig.add_trace(go.Bar(x=d["periode"], y=d["nilai_booking_rp"], name=nm, marker_color=col,
                             hovertemplate="%{x|%b %Y}<br>" + nm + ": %{y:,.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="estimasi nilai booking (Rp)", tickprefix="Rp ")
    c2.plotly_chart(styl(fig).update_layout(title="Nilai booking: terealisasi vs hilang per bulan"),
                    width="stretch")
    st.divider()

    # 6. Asal tamu
    st.subheader("Dari mana tamu berasal")
    neg = (df[df["country"] != "Tidak diketahui"].groupby("country")
           .agg(n=("is_canceled", "size"), rate=("is_canceled", "mean")).reset_index()
           .sort_values("n", ascending=False).head(10).sort_values("n"))
    fig = go.Figure(go.Bar(
        x=neg["n"], y=neg["country"], orientation="h", marker_color=C_SEQ,
        text=[f"{fmt_ribu(n)}  ·  {r:.0%} batal" for n, r in zip(neg["n"], neg["rate"])],
        textposition="outside", hovertemplate="%{y}: %{x} booking<extra></extra>",
    ))
    fig.update_xaxes(title="jumlah booking")
    st.plotly_chart(styl(fig, 420).update_layout(title="10 negara asal terbanyak (kode ISO)"),
                    width="stretch")
    st.caption("Portugal (PRT) mendominasi sekaligus menyumbang pembatalan terbanyak — pasar domestik perlu strategi retensi tersendiri.")
    st.divider()

    # 7. Rekomendasi
    st.subheader("Rekomendasi")
    st.markdown(
        """
1. **Kelola booking lead-time panjang.** Reservasi > 3 bulan sebelum kedatangan
   punya risiko batal tertinggi. Terapkan konfirmasi ulang berkala (H-60 / H-30).
2. **Tinjau kebijakan deposit.** Segmen dengan deposit “Non-Refund” hampir selalu
   berakhir batal — cek bersama tim reservasi.
3. **Fokus segmen berisiko.** Groups dan sebagian Offline TA/TO membatalkan jauh
   di atas rata-rata; ketatkan termin pembayaran.
4. **Overbooking terukur di puncak musim.** Volume dan ADR sama-sama memuncak di
   Jul–Agu; gunakan probabilitas batal per booking (lihat tab Prediksi).
5. **Program retensi pasar domestik (Portugal).**
"""
    )
    st.caption(
        "Semua angka Rupiah adalah estimasi hasil konversi dari ADR (EUR) memakai kurs pada panel Filter. "
        "Nilai booking = ADR × jumlah malam; belum termasuk pendapatan non-kamar."
    )


# =========================================================================== #
# TAB 2 — PREDIKSI 2018
# =========================================================================== #
def _artefak_hilang(nama_file: list[str]) -> list[str]:
    return [f for f in nama_file if not (MODELS_DIR / f).exists()]


def render_forecast() -> None:
    hist = load_csv("forecast_history.csv")
    fc = load_csv("forecast_2018.csv")
    meta = load_json("forecast_metrics.json")
    if hist is None or fc is None or meta is None:
        st.info("Belum ada hasil forecast. Jalankan dulu:  `python -m src.forecast_2018`")
        return

    st.subheader("Perkiraan angka bulanan 2018")
    mape_hw = {k: v["holt_winters"] for k, v in meta["backtest_mape_pct"].items()}
    lo_mape, hi_mape = min(mape_hw.values()), max(mape_hw.values())
    st.markdown(
        f"Data hanya **{meta['history_months']} bulan** "
        f"({meta['history_range'][0][:7]} – {meta['history_range'][1][:7]}) — 2 siklus musiman. "
        "Model deret waktu (Holt-Winters), **bukan** machine learning. "
        "Pada uji coba menyembunyikan 6 bulan terakhir, perkiraan meleset rata-rata "
        f"**{lo_mape:.0f}–{hi_mape:.0f}%** — jadi baca angka di bawah "
        "sebagai **rentang arah**, bukan target pasti."
    )

    metrik = st.radio("Metrik", list(METRIK_LABEL), format_func=METRIK_LABEL.get,
                      horizontal=True, key="fc_metrik")
    faktor = kurs if metrik == "revenue_eur" else 1.0
    is_rp = metrik == "revenue_eur"

    h = hist[hist["metric"] == metrik].copy()
    h["month"] = pd.to_datetime(h["month"])
    h["value"] *= faktor
    f_hw = fc[(fc["metric"] == metrik) & (fc["model"] == "holt_winters")].copy()
    f_sn = fc[(fc["metric"] == metrik) & (fc["model"] == "seasonal_naive")].copy()
    for d in (f_hw, f_sn):
        d["month"] = pd.to_datetime(d["month"])
        for c in ["yhat", "lo80", "hi80", "lo95", "hi95"]:
            d[c] *= faktor

    fig = go.Figure()
    # pita 80%
    fig.add_trace(go.Scatter(x=f_hw["month"], y=f_hw["hi80"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f_hw["month"], y=f_hw["lo80"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                             name="rentang 80%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h["month"], y=h["value"], mode="lines", name="aktual",
                             line=dict(color=INK_2, width=2)))
    fig.add_trace(go.Scatter(x=f_hw["month"], y=f_hw["yhat"], mode="lines", name="perkiraan (Holt-Winters)",
                             line=dict(color=C_SEQ, width=2)))
    fig.add_trace(go.Scatter(x=f_sn["month"], y=f_sn["yhat"], mode="lines",
                             name="pembanding (seasonal-naive)",
                             line=dict(color=C_RESORT, width=1.5, dash="dot")))
    fig.add_vrect(x0="2018-01-01", x1="2018-12-31", fillcolor=MUTED, opacity=0.06, line_width=0)
    if is_rp:
        fig.update_yaxes(tickprefix="Rp ")
    st.plotly_chart(styl(fig, 420).update_layout(title=f"{METRIK_LABEL[metrik]} — aktual & perkiraan 2018"),
                    width="stretch")

    # Ringkasan total 2018
    y2018 = f_hw[f_hw["month"].dt.year == 2018]
    tot, lo, hi = y2018["yhat"].sum(), y2018["lo80"].sum(), y2018["hi80"].sum()
    prev = h[h["month"].dt.year == 2016]["value"].sum()  # 2016 = tahun penuh terakhir
    c = st.columns(3)
    fnum = (lambda v: fmt_rp(v)) if is_rp else (lambda v: fmt_ribu(v))
    c[0].metric(f"Perkiraan total 2018 · {METRIK_LABEL[metrik]}", fnum(tot),
                delta=f"{(tot / prev - 1):+.0%} vs 2016" if prev else None)
    c[1].metric("Batas bawah (80%)", fnum(lo))
    c[2].metric("Batas atas (80%)", fnum(hi))

    with st.expander("Tabel bulanan 2018"):
        tampil = y2018[["month", "yhat", "lo80", "hi80"]].copy()
        tampil["month"] = tampil["month"].dt.strftime("%b 2018")
        tampil.columns = ["Bulan", "Perkiraan", "Bawah 80%", "Atas 80%"]
        for cc in ["Perkiraan", "Bawah 80%", "Atas 80%"]:
            tampil[cc] = tampil[cc].map(fnum)
        st.dataframe(tampil, hide_index=True, width="stretch")


def render_model_pembatalan() -> None:
    metrics = load_json("cancellation_metrics.json")
    imp = load_csv("cancellation_feature_importance.csv")
    rules = load_text("cancellation_tree_rules.txt")
    if metrics is None:
        st.info("Belum ada model. Jalankan dulu:  `python -m src.train_cancellation`")
        return

    st.subheader("Model risiko pembatalan")
    sp = metrics["split"]
    st.markdown(
        f"Dilatih pada **{sp['n_train']:,} booking 2015–2016**, diuji pada "
        f"**{sp['n_test']:,} booking 2017** yang tidak dipakai melatih. "
        "Kolom yang membocorkan jawaban (`reservation_status` dll) dibuang."
    )

    mdf = pd.DataFrame(metrics["models"]).set_index("model")
    g = mdf.loc["gboost"]
    base = mdf.loc["baseline (semua 'tidak batal')"]
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{g['roc_auc']:.2f}", help="1,0 = sempurna · 0,5 = tebak asal")
    c[1].metric("PR-AUC", f"{g['pr_auc']:.2f}", delta=f"{g['pr_auc'] - base['pr_auc']:+.2f} vs asal-tebak")
    c[2].metric("Akurasi", f"{g['accuracy']:.0%}")
    c[3].metric("Recall (batal terdeteksi)", f"{g['recall']:.0%}")
    c[4].metric("Brier score", f"{g['brier']:.3f}", help="makin kecil makin baik; kualitas probabilitas")

    st.caption("Perbandingan ketiga model (diuji pada data 2017):")
    show = mdf.reset_index()[["model", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "brier"]]
    show.columns = ["Model", "ROC-AUC", "PR-AUC", "Akurasi", "Presisi", "Recall", "Brier"]
    st.dataframe(show, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    if imp is not None:
        top = imp.head(10).sort_values("importance")
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                               marker_color=C_SEQ,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="penurunan ROC-AUC bila fitur diacak")
        c1.plotly_chart(styl(fig, 380).update_layout(title="Faktor paling menentukan (model gboost)"),
                        width="stretch")

    cal = metrics.get("calibration_gboost")
    if cal:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=MUTED, dash="dot"), name="ideal"))
        fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"],
                                 mode="lines+markers", line=dict(color=C_SEQ, width=2),
                                 name="model",
                                 hovertemplate="prediksi %{x:.0%} → aktual %{y:.0%}<extra></extra>"))
        fig.update_xaxes(title="probabilitas yang diprediksi", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="proporsi batal sebenarnya", tickformat=".0%", range=[0, 1])
        c2.plotly_chart(styl(fig, 380).update_layout(title="Kalibrasi: apakah “70%” benar-benar 70%?"),
                        width="stretch")

    st.markdown(
        "**Catatan:** kolom `deposit_type = Non Refund` sangat dominan karena di data "
        "99% booking jenis ini berakhir batal — kemungkinan besar artefak pencatatan. "
        "Model tetap akurat tanpa mengandalkan kolom ini (lihat faktor lain di grafik), "
        "tapi sebaiknya divalidasi ke tim reservasi sebelum dipakai untuk kebijakan."
    )
    if rules:
        with st.expander("Versi sederhana: aturan pohon keputusan (bisa dibaca manusia)"):
            st.code(rules, language="text")


def render_skor_booking() -> None:
    model = load_model("cancellation_gboost.joblib")
    schema = load_json("cancellation_input_schema.json")
    if model is None or schema is None:
        st.info("Belum ada model. Jalankan dulu:  `python -m src.train_cancellation`")
        return

    st.subheader("Coba: skor satu booking")
    st.caption("Isi detail sebuah pemesanan; model memperkirakan kemungkinannya dibatalkan.")

    def num(key, label, step=1.0):
        s = schema[key]
        return st.number_input(label, min_value=float(s["min"]), max_value=float(max(s["max"], s["min"] + 1)),
                               value=float(s["median"]), step=step, key=f"in_{key}")

    def cat(key, label):
        s = schema[key]
        return st.selectbox(label, s["choices"], key=f"in_{key}")

    c1, c2, c3 = st.columns(3)
    with c1:
        hotel = cat("hotel", "Hotel")
        market_segment = cat("market_segment", "Segmen pasar")
        distribution_channel = cat("distribution_channel", "Kanal distribusi")
        customer_type = cat("customer_type", "Tipe pelanggan")
        deposit_type = cat("deposit_type", "Jenis deposit")
    with c2:
        lead_time = num("lead_time", "Lead time (hari sebelum datang)")
        adr = num("adr", "ADR (EUR)")
        arrival_date_week_number = num("arrival_date_week_number", "Minggu ke- (1–53)")
        total_of_special_requests = num("total_of_special_requests", "Jumlah permintaan khusus")
        required_car_parking_spaces = num("required_car_parking_spaces", "Slot parkir diminta")
    with c3:
        adults = num("adults", "Dewasa")
        children = num("children", "Anak")
        stays_in_week_nights = num("stays_in_week_nights", "Malam hari kerja")
        stays_in_weekend_nights = num("stays_in_weekend_nights", "Malam akhir pekan")
        previous_cancellations = num("previous_cancellations", "Pembatalan sebelumnya")
        is_repeated_guest = st.selectbox("Tamu langganan?", [0, 1],
                                         format_func=lambda v: "Ya" if v else "Tidak", key="in_rep")

    # Baris fitur lengkap: mulai dari nilai default skema, timpa dengan input form
    row = {}
    for f in NUMERIC_FEATURES:
        row[f] = float(schema[f]["median"])
    for f in CATEGORICAL_FEATURES:
        row[f] = schema[f]["default"]
    row.update(dict(
        hotel=hotel, market_segment=market_segment, distribution_channel=distribution_channel,
        customer_type=customer_type, deposit_type=deposit_type,
        lead_time=lead_time, adr=adr, arrival_date_week_number=arrival_date_week_number,
        total_of_special_requests=total_of_special_requests,
        required_car_parking_spaces=required_car_parking_spaces,
        adults=adults, children=children, stays_in_week_nights=stays_in_week_nights,
        stays_in_weekend_nights=stays_in_weekend_nights,
        previous_cancellations=previous_cancellations, is_repeated_guest=is_repeated_guest,
    ))
    row["total_nights"] = stays_in_week_nights + stays_in_weekend_nights
    row["total_guests"] = adults + children + row.get("babies", 0)
    X_one = pd.DataFrame([row])[FEATURES]

    prob = float(model.predict_proba(X_one)[0, 1])
    warna = C_BATAL if prob >= 0.6 else (C_RESORT if prob >= 0.35 else C_REALISASI)
    label = "RISIKO TINGGI" if prob >= 0.6 else ("Risiko sedang" if prob >= 0.35 else "Risiko rendah")

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": warna},
            "steps": [
                {"range": [0, 35], "color": "rgba(12,163,12,0.12)"},
                {"range": [35, 60], "color": "rgba(235,104,52,0.12)"},
                {"range": [60, 100], "color": "rgba(208,59,59,0.12)"},
            ],
        },
        title={"text": f"Kemungkinan dibatalkan — <b>{label}</b>"},
    ))
    st.plotly_chart(styl(fig, 320), width="stretch")
    st.caption(
        "Rata-rata historis 37%. Model paling berguna untuk memberi peringkat prioritas "
        "(booking mana yang perlu dikonfirmasi ulang), bukan vonis pasti."
    )


def render_prediksi() -> None:
    render_forecast()
    st.divider()
    render_model_pembatalan()
    st.divider()
    render_skor_booking()


# --------------------------------------------------------------------------- #
with tab_ringkasan:
    render_ringkasan()
with tab_prediksi:
    render_prediksi()
