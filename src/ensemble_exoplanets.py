#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensemble_tabular_exoplanets.py
Stacking: [LightGBM (GPU se disponível) + RandomForest + ExtraTrees] -> Meta: Regressão Logística
Saída = probabilidade 0..1. Thresholds: 0.95 / 0.50.

Execução:
  conda activate exoplanets
  python /home/rota_2024/nasa/src/ensemble_tabular_exoplanets.py --predict
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import joblib
import matplotlib.pyplot as plt

import lightgbm as lgb

try:
    from tsfresh import extract_features
    TSFRESH_OK = True
except Exception:
    TSFRESH_OK = False

ROOT = Path("/home/rota_2024/nasa").resolve()
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = [
    DATA_DIR / "kepler_objects_of_interest.csv",
    DATA_DIR / "k2.csv",
    DATA_DIR / "TESS.csv",
]

THRESH_PLANET = 0.95
THRESH_CANDIDATE = 0.50

PLANET_STR = {"confirmed", "cp", "kp", "planet", "yes"}
NONPLANET_STR = {"false positive", "fp", "not", "non-planet", "refuted", "afp", "ntp"}
LABEL_LIKE_COLS = {
    "y_bin", "label", "class", "target",
    "koi_disposition", "disposition", "tfopwg_disp",
    "is_planet", "is_exoplanet"
}

# ---------- Alinhador picklable ----------
class FeatureAligner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        X = X.copy()
        return X.reindex(columns=self.feature_names, fill_value=np.nan)
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names, dtype=object)

def _read_csv_any(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] arquivo ausente: {path}")
        return None
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#", on_bad_lines="skip")
        if df is not None and df.shape[1] >= 1:
            print(f"[OK] lido (autodetect) {path.name} -> {df.shape}")
            return df
    except Exception as e:
        print(f"[WARN] autodetect falhou em {path.name}: {e}")
    for sep in [";", "\t", ",", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", comment="#", on_bad_lines="skip")
            if df is not None and df.shape[1] >= 1:
                print(f"[OK] lido (sep='{sep}') {path.name} -> {df.shape}")
                return df
        except Exception as e:
            print(f"[WARN] falha com sep='{sep}' em {path.name}: {e}")
    print(f"[ERRO] não foi possível ler {path.name}")
    return None

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _merge_sources(csv_files: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_files:
        df = _read_csv_any(p)
        if df is None or df.empty:
            continue
        df = _std_cols(df)
        df["__source__"] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _extract_label_bin(df: pd.DataFrame):
    s = pd.Series(index=df.index, dtype="float64")
    raw = pd.Series(index=df.index, dtype="object")
    candidates = ["koi_disposition","disposition","tfopwg_disp","label","class"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return s, raw
    raw = df[col].astype(str).fillna("").str.strip()
    low = raw.str.lower()
    is_p = low.isin(PLANET_STR) | low.str.contains(r"\bconfirmed\b|\bcp\b|\bkp\b|\bplanet\b", regex=True)
    is_n = low.isin(NONPLANET_STR) | low.str.contains(r"false\s*positive|fp|afp|ntp|non[- ]?planet|refuted", regex=True)
    is_c = low.str.contains(r"\bpc\b|\bcandidate\b", regex=True)
    s.loc[is_p] = 1.0
    s.loc[is_n] = 0.0
    s.loc[is_c & ~is_p & ~is_n] = np.nan
    return s, raw

def _drop_label_like(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in LABEL_LIKE_COLS]
    return df[cols].copy()

def _collect_tabular(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_label_like(df)
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        return num
    nun = num.dropna(axis=1, how="all")
    constant = [c for c in nun.columns if nun[c].nunique(dropna=True) <= 1]
    nun = nun.drop(columns=constant, errors="ignore")
    return nun

def _maybe_tsfresh(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not TSFRESH_OK:
        return None
    cols = {c.lower(): c for c in df.columns}
    time_c = next((cols[c] for c in ["time","bjd","btjd","mjd","jd"] if c in cols), None)
    flux_c = next((cols[c] for c in ["pdcsap_flux","sap_flux","flux","flux_norm","relative_flux"] if c in cols), None)
    id_c   = next((cols[c] for c in ["id","kepid","epic","epic_id","tic","tic_id","object_id"] if c in cols), None)
    if not (time_c and flux_c and id_c):
        return None
    df2 = df[[id_c, time_c, flux_c]].dropna().copy()
    df2.columns = ["id","time","flux"]
    if df2.empty:
        return None
    try:
        feats = extract_features(df2, column_id="id", column_sort="time")
        feats = feats.replace([np.inf,-np.inf], np.nan).fillna(0.0)
        feats.reset_index(drop=True, inplace=True)
        return feats
    except Exception as e:
        print("[WARN] TSFRESH falhou:", e)
        return None

def _gpu_params_or_cpu() -> dict:
    return dict(
        objective="binary",
        learning_rate=0.05,
        n_estimators=600,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_gain_to_split=0.0,
        random_state=42,
        device_type="gpu",
        device="gpu",
    )

def _cpu_params() -> dict:
    return dict(
        objective="binary",
        learning_rate=0.05,
        n_estimators=600,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_gain_to_split=0.0,
        random_state=42,
    )

def train_and_predict(infile: Optional[Path]=None, predict_only: bool=False) -> Path:
    model_path = ART_DIR / "model_ensemble.pkl"
    schema_path = ART_DIR / "ensemble_feature_columns.json"
    preds_path = ART_DIR / "predictions_ensemble.csv"

    if predict_only:
        if infile is None or not model_path.exists() or not schema_path.exists():
            raise SystemExit("[ERRO] precisa de ensemble treinado e schema; use --predict antes.")
        with open(schema_path) as f:
            feature_names = json.load(f)["feature_names"]
        bundle, meta = joblib.load(model_path)

        df_new = _std_cols(_read_csv_any(infile) or pd.DataFrame())
        X_all = _collect_tabular(df_new).reindex(columns=feature_names, fill_value=np.nan)

        p_lgb = bundle["lgb"].predict_proba(X_all)[:,1]
        p_rf  = bundle["rf"].predict_proba(X_all)[:,1]
        p_et  = bundle["et"].predict_proba(X_all)[:,1]
        X_meta = np.vstack([p_lgb, p_rf, p_et]).T
        prob = meta.predict_proba(X_meta)[:,1]
        out = df_new.copy()
        out["prob_planet"] = prob
        out["class"] = np.where(out["prob_planet"]>=THRESH_PLANET, "planet",
                        np.where(out["prob_planet"]>=THRESH_CANDIDATE, "candidate", "non-planet"))
        out.to_csv(preds_path, index=False)
        print(f"[OK] predições em {preds_path}")
        return preds_path

    # ---------- Treino ----------
    if infile is not None:
        df = _std_cols(_read_csv_any(infile) or pd.DataFrame())
    else:
        df = _merge_sources(CSV_FILES)
    if df.empty:
        raise SystemExit("[ERRO] nenhum CSV lido.")

    y_bin = pd.Series(index=df.index, dtype="float64")
    raw = pd.Series(index=df.index, dtype="object")
    y_bin, raw = _extract_label_bin(df)
    df["y_bin"] = y_bin
    df_train = df[df["y_bin"].notna()].copy()
    if df_train.empty:
        raise SystemExit("[ERRO] sem rótulos binários confiáveis (confirmed x false positive).")
    y = df_train["y_bin"].astype(int).values

    X_tab = _collect_tabular(df_train)

    feats = _maybe_tsfresh(df_train)
    if feats is not None and not feats.empty:
        feats = feats.reset_index(drop=True)
        X_tab = X_tab.reset_index(drop=True)
        n = min(len(X_tab), len(feats))
        X_tab = pd.concat([X_tab.iloc[:n], feats.iloc[:n]], axis=1)
        y = y[:n]

    X_tr, X_va, y_tr, y_va = train_test_split(X_tab, y, test_size=0.2, stratify=y, random_state=42)

    feature_names = list(X_tab.columns)
    pre = Pipeline([("align", FeatureAligner(feature_names)), ("imp", SimpleImputer(strategy="median"))])

    lgbm = lgb.LGBMClassifier(**_gpu_params_or_cpu(), class_weight="balanced")
    rf   = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=-1,
                                  class_weight="balanced", random_state=42)
    et   = ExtraTreesClassifier(n_estimators=800, max_features="sqrt", n_jobs=-1,
                                class_weight="balanced", random_state=42)

    pipe_lgb = Pipeline([("pre", pre), ("clf", lgbm)])
    pipe_rf  = Pipeline([("pre", pre), ("clf", rf)])
    pipe_et  = Pipeline([("pre", pre), ("clf", et)])

    try:
        pipe_lgb.fit(X_tr, y_tr)
    except Exception as e:
        print("[WARN] LightGBM GPU falhou, usando CPU:", e)
        lgbm = lgb.LGBMClassifier(**_cpu_params(), class_weight="balanced")
        pipe_lgb = Pipeline([("pre", pre), ("clf", lgbm)])
        pipe_lgb.fit(X_tr, y_tr)

    pipe_rf.fit(X_tr, y_tr)
    pipe_et.fit(X_tr, y_tr)

    p_lgb_tr = pipe_lgb.predict_proba(X_tr)[:,1]
    p_rf_tr  = pipe_rf.predict_proba(X_tr)[:,1]
    p_et_tr  = pipe_et.predict_proba(X_tr)[:,1]
    X_meta_tr = np.vstack([p_lgb_tr, p_rf_tr, p_et_tr]).T

    p_lgb_va = pipe_lgb.predict_proba(X_va)[:,1]
    p_rf_va  = pipe_rf.predict_proba(X_va)[:,1]
    p_et_va  = pipe_et.predict_proba(X_va)[:,1]
    X_meta_va = np.vstack([p_lgb_va, p_rf_va, p_et_va]).T

    meta = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    meta.fit(X_meta_tr, y_tr)

    prob = meta.predict_proba(X_meta_va)[:,1]
    y_pred = (prob >= 0.5).astype(int)
    roc = roc_auc_score(y_va, prob) if len(np.unique(y_va)) > 1 else np.nan
    ap  = average_precision_score(y_va, prob)
    print(f"[METRICS] ROC-AUC={roc:.3f}  AP(PR-AUC)={ap:.3f}")
    print(classification_report(y_va, y_pred, digits=3))

    # PR curve
    prec, rec, _ = precision_recall_curve(y_va, prob)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.title("Precision-Recall (Stacking)")
    plt.savefig(ART_DIR / "precision_recall_ensemble.png", bbox_inches="tight")

    # salvar bundle e schema
    bundle = {"lgb": pipe_lgb, "rf": pipe_rf, "et": pipe_et}
    joblib.dump((bundle, meta), model_path)
    with open(ART_DIR / "ensemble_feature_columns.json", "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    print(f"[OK] ensemble salvo em {model_path}")
    print(f"[OK] schema salvo em {ART_DIR / 'ensemble_feature_columns.json'}")

    # predições no conjunto completo
    df_all = _merge_sources(CSV_FILES) if infile is None else _std_cols(_read_csv_any(infile) or pd.DataFrame())
    X_all = _collect_tabular(df_all).reindex(columns=feature_names, fill_value=np.nan)
    p_lgb = bundle["lgb"].predict_proba(X_all)[:,1]
    p_rf  = bundle["rf"].predict_proba(X_all)[:,1]
    p_et  = bundle["et"].predict_proba(X_all)[:,1]
    X_meta = np.vstack([p_lgb, p_rf, p_et]).T
    p_all = meta.predict_proba(X_meta)[:,1]

    out = df_all.copy()
    out["prob_planet"] = p_all
    out["class"] = np.where(out["prob_planet"]>=THRESH_PLANET, "planet",
                    np.where(out["prob_planet"]>=THRESH_CANDIDATE, "candidate", "non-planet"))
    preds_path = ART_DIR / "predictions_ensemble.csv"
    out.to_csv(preds_path, index=False)
    print(f"[OK] predições em {preds_path}")

    return preds_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true", help="Treinar e gerar predições nos CSVs padrão.")
    ap.add_argument("--predict-only", action="store_true", help="Só prever com ensemble salvo (requer --infile).")
    ap.add_argument("--infile", type=str, default=None, help="CSV único para treinar/prever.")
    args = ap.parse_args()

    if args.predict_only:
        train_and_predict(Path(args.infile) if args.infile else None, predict_only=True)
    elif args.predict:
        train_and_predict(Path(args.infile) if args.infile else None, predict_only=False)
    else:
        train_and_predict()
