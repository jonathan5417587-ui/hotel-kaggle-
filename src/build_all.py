"""Bangun semua artefak prediksi sekaligus.

Jalankan:  python -m src.build_all
"""

from src import forecast_2018, train_cancellation

if __name__ == "__main__":
    print("=" * 60, "\n1/2 · Model pembatalan\n", "=" * 60, sep="")
    train_cancellation.main()
    print("\n", "=" * 60, "\n2/2 · Forecast 2018\n", "=" * 60, sep="")
    forecast_2018.main()
    print("\nSemua artefak siap. Jalankan:  streamlit run app.py")
