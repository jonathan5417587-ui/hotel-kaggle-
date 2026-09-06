"""
Latih model prediksi pembatalan booking.

Menghasilkan 3 model:
  - logistic : regresi logistik    -> baseline yang mudah dijelaskan
  - tree     : pohon keputusan     -> aturan yang bisa dibaca manusia
  - gboost   : gradient boosting   -> paling akurat, dipakai untuk skor

Aturan main:
  - Kolom "bocor" (reservation_status dll) TIDAK dipakai.
  - Data dipisah berdasarkan waktu: latih 2015-2016, uji 2017.

Jalankan:  python -m src.train_cancellation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import joblib

from src.data import (
    CATEGORICAL_FEATURES, FEATURES, HIGH_CARD_COLS, NUMERIC_FEATURES,
    RareCategoryGrouper, load_clean, make_cancellation_xy, temporal_split,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor(scale_numeric: bool) -> Pipeline:
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    column_tf = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), CATEGORICAL_FEATURES),
        ("num", numeric_step, NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("group_rare", RareCategoryGrouper(columns=HIGH_CARD_COLS, top_k=12)),
        ("columns", column_tf),
    ])


def build_model(kind: str) -> Pipeline:
    if kind == "logistic":
        pre = build_preprocessor(scale_numeric=True)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)
    elif kind == "tree":
        pre = build_preprocessor(scale_numeric=False)
        clf = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=200, class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif kind == "gboost":
        pre = build_preprocessor(scale_numeric=False)
        clf = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=400,
            l2_regularization=1.0, early_stopping=True, random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("prep", pre), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #
def evaluate(name: str, y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def grouped_permutation_importance(model, X, y, n_repeats=5) -> pd.DataFrame:
    """Permutation importance per kolom asli (sebelum one-hot)."""
    r = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )
    imp = (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                      "std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def input_schema(df: pd.DataFrame) -> dict:
    """Rentang & pilihan tiap fitur — dipakai form 'skor satu booking' di dashboard."""
    schema = {}
    for c in NUMERIC_FEATURES:
        s = df[c]
        schema[c] = {
            "type": "number",
            "min": float(s.min()), "max": float(s.quantile(0.995)),
            "median": float(s.median()),
        }
    for c in CATEGORICAL_FEATURES:
        vc = df[c].astype(str).value_counts()
        schema[c] = {
            "type": "category",
            "choices": vc.head(15).index.tolist(),
            "default": vc.index[0],
        }
    return schema


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Memuat data…")
    df = load_clean()
    train_df, test_df = temporal_split(df)
    X_train, y_train = make_cancellation_xy(train_df)
    X_test, y_test = make_cancellation_xy(test_df)
    print(f"  latih {len(X_train):,} (2015-2016) · uji {len(X_test):,} (2017)")

    results = {
        "split": {
            "train_years": [2015, 2016], "test_years": [2017],
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_cancel_rate": round(float(y_train.mean()), 4),
            "test_cancel_rate": round(float(y_test.mean()), 4),
        },
        "models": [],
    }

    # Baseline konyol: tebak semua "tidak batal"
    base_pred = np.zeros(len(y_test))
    base = evaluate("baseline (semua 'tidak batal')", y_test, base_pred)
    base["roc_auc"] = 0.5
    base["pr_auc"] = round(float(y_test.mean()), 4)
    results["models"].append(base)

    fitted = {}
    for kind in ["logistic", "tree", "gboost"]:
        print(f"Melatih: {kind}…")
        model = build_model(kind)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        m = evaluate(kind, y_test, proba)
        results["models"].append(m)
        fitted[kind] = model
        joblib.dump(model, MODELS_DIR / f"cancellation_{kind}.joblib")
        print(f"  ROC-AUC {m['roc_auc']}  PR-AUC {m['pr_auc']}  akurasi {m['accuracy']}")

    # Kurva kalibrasi (pakai model gboost)
    proba_g = fitted["gboost"].predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba_g, n_bins=10, strategy="quantile")
    results["calibration_gboost"] = {
        "mean_predicted": [round(float(x), 4) for x in mean_pred],
        "fraction_positive": [round(float(x), 4) for x in frac_pos],
    }

    # Feature importance (gboost, di-subsample biar cepat)
    print("Menghitung feature importance…")
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=min(8000, len(X_test)), replace=False)
    imp = grouped_permutation_importance(fitted["gboost"], X_test.iloc[idx], y_test.iloc[idx])
    imp.to_csv(MODELS_DIR / "cancellation_feature_importance.csv", index=False)

    # Aturan pohon (bisa dibaca manusia)
    tree_pipe = fitted["tree"]
    feat_names = tree_pipe.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    rules = export_text(tree_pipe.named_steps["clf"], feature_names=list(feat_names), max_depth=4)
    (MODELS_DIR / "cancellation_tree_rules.txt").write_text(rules, encoding="utf-8")

    # Skema input untuk form dashboard
    (MODELS_DIR / "cancellation_input_schema.json").write_text(
        json.dumps(input_schema(df), indent=2), encoding="utf-8"
    )

    (MODELS_DIR / "cancellation_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSelesai. Artefak tersimpan di:", MODELS_DIR)
    print(pd.DataFrame(results["models"])[["model", "roc_auc", "pr_auc", "accuracy", "recall", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
