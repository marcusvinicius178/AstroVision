#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exo_tabular_lgbm.py
Baseline tabular com saída probabilística 0..1 usando LightGBM (GPU se disponível, com fallback p/ CPU).
- Lê CSVs KOI/K2/TESS com parsing robusto
- Constrói rótulo binário (CONFIRMED=1; FALSE POSITIVE/negativos=0; CANDIDATE=ignorado no treino)
- NÃO inclui colunas de rótulo/disposition nas features (evita vazamento)
- (Opcional) TSFRESH se houver time/flux (senão segue só tabular)
- Alinha colunas via transformer de topo (classe picklable) + imputa faltantes
- Salva modelo e gera predições (prob, class) com thresholds:
    prob >= 0.95 => "planet"
    0.50 <= prob < 0.95 => "candidate"
    prob < 0.50 => "non-planet"

Execução:
  conda activate exoplanets
  python /home/rota_2024/nasa/src/exo_tabular_lgbm.py --predict
  # -> artifacts/model_lgbm.pkl  e  artifacts/predictions_lgbm.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# LightGBM
import lightgbm as lgb

# TSFRESH (opcional, se houver time/flux)
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

# limiares de saída
THRESH_PLANET = 0.95
THRESH_CANDIDATE = 0.50

# termos para mapeamento de rótulo textual
PLANET_STR = {"confirmed", "cp", "kp", "planet", "yes"}
NONPLANET_STR = {"false positive", "fp", "not", "non-planet", "refuted", "afp", "ntp"}

# colunas que NUNCA entram nas features (evita vazamento)
LABEL_LIKE_COLS = {
    "y_bin", "label", "class", "target",
    "koi_disposition", "disposition", "tfopwg_disp",
    # variantes que às vezes aparecem
    "is_planet", "is_exoplanet"
}

# -----------------------------
#  Transformer alinhador (picklable)
# -----------------------------
class FeatureAligner(BaseEstimator, TransformerMixin):
    """Garante que X tenha exatamente as colunas de feature_names, na ordem, criando ausentes com NaN."""
    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            # Para segurança: converte para DataFrame se vier array
            X = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        X = X.copy()
        return X.reindex(columns=self.feature_names, fill_value=np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names, dtype=object)

# -----------------------------
#  IO / utilitários
# -----------------------------

def _read_csv_any(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] arquivo ausente: {path}")
        return None
    # 1) autodetect (engine='python')
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#", on_bad_lines="skip")
        if df is not None and df.shape[1] >= 1:
            print(f"[OK] lido (autodetect) {path.name} -> {df.shape}")
            return df
    except Exception as e:
        print(f"[WARN] autodetect falhou em {path.name}: {e}")
    # 2) separadores comuns
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
    """
    Retorna (y_bin, y_raw_text) — y_bin com {0,1,NaN}, onde:
      1 = confirmado (cp/kp/confirmed/planet)
      0 = falso positivo/negativo
      NaN = candidato/indefinido (fica fora do treino)
    """
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
    """Mantém só numéricas úteis, removendo colunas label-like e constantes."""
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
    # parâmetros estáveis; force_col_wise é padrão automático, aqui deixamos o LightGBM decidir
    return dict(
        objective="binary",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_gain_to_split=0.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        device_type="gpu",
        device="gpu",
    )

def _cpu_params() -> dict:
    return dict(
        objective="binary",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_gain_to_split=0.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
    )

# -----------------------------
#  Treino / Predição
# -----------------------------

def train_and_predict(infile: Optional[Path]=None, predict_only: bool=False) -> Path:
    model_path = ART_DIR / "model_lgbm.pkl"
    schema_path = ART_DIR / "lgbm_feature_columns.json"
    preds_path = ART_DIR / "predictions_lgbm.csv"

    if predict_only:
        if infile is None or not model_path.exists() or not schema_path.exists():
            raise SystemExit("[ERRO] para --predict-only, garanta artifacts/model_lgbm.pkl e lgbm_feature_columns.json, e passe --infile.")
        df_new = _std_cols(_read_csv_any(infile) or pd.DataFrame())
        X_all = _collect_tabular(df_new)
        with open(schema_path) as f:
            feature_names = json.load(f)["feature_names"]
        X_all = X_all.reindex(columns=feature_names, fill_value=np.nan)

        pipe = joblib.load(model_path)
        proba = pipe.predict_proba(X_all)[:,1]
        out = df_new.copy()
        out["prob_planet"] = proba
        out["class"] = np.where(out["prob_planet"]>=THRESH_PLANET, "planet",
                        np.where(out["prob_planet"]>=THRESH_CANDIDATE, "candidate", "non-planet"))
        out.to_csv(preds_path, index=False)
        print(f"[OK] predições em {preds_path}")
        return preds_path

    # Treino
    if infile is not None:
        df = _std_cols(_read_csv_any(infile) or pd.DataFrame())
    else:
        df = _merge_sources(CSV_FILES)
    if df.empty:
        raise SystemExit("[ERRO] nenhum CSV lido.")

    y_bin, _ = _extract_label_bin(df)
    df["y_bin"] = y_bin
    df_train = df[df["y_bin"].notna()].copy()
    if df_train.empty:
        raise SystemExit("[ERRO] sem rótulos binários (confirmed x false positive).")

    # features tabulares
    X_tab = _collect_tabular(df_train)
    y = df_train["y_bin"].astype(int).values

    # (opcional) acrescenta TSFRESH se houver time/flux/id
    feats = _maybe_tsfresh(df_train)
    if feats is not None and not feats.empty:
        feats = feats.reset_index(drop=True)
        X_tab = X_tab.reset_index(drop=True)
        n = min(len(X_tab), len(feats))
        X_tab = pd.concat([X_tab.iloc[:n], feats.iloc[:n]], axis=1)
        y = y[:n]

    # split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tab, y, test_size=0.2, random_state=42, stratify=y
    )

    # lista final de colunas de treino (salvamos para alinhamento posterior)
    feature_names = list(X_tab.columns)

    # pré-processador: alinhar -> imputar (classe picklable, sem closures)
    pre = Pipeline(steps=[
        ("align", FeatureAligner(feature_names)),
        ("imp", SimpleImputer(strategy="median")),
    ])

    lgbm_gpu = lgb.LGBMClassifier(**_gpu_params_or_cpu(), class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", lgbm_gpu)])

    # tentativa com GPU
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        print("[WARN] LightGBM GPU falhou, trocando para CPU:", e)
        lgbm_cpu = lgb.LGBMClassifier(**_cpu_params(), class_weight="balanced")
        pipe = Pipeline(steps=[("pre", pre), ("clf", lgbm_cpu)])
        pipe.fit(X_train, y_train)

    # avaliação
    p_val = pipe.predict_proba(X_val)[:,1]
    roc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan
    ap  = average_precision_score(y_val, p_val)
    y_pred = (p_val >= 0.5).astype(int)
    print(f"[METRICS] ROC-AUC={roc:.3f}  AP(PR-AUC)={ap:.3f}")
    print(classification_report(y_val, y_pred, digits=3))

    # salva modelo e schema de colunas
    joblib.dump(pipe, model_path)
    with open(schema_path, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    print(f"[OK] modelo salvo em {model_path}")
    print(f"[OK] schema de colunas salvo em {schema_path}")

    # predições no conjunto completo (ou infile)
    df_all = _merge_sources(CSV_FILES) if infile is None else _std_cols(_read_csv_any(infile) or pd.DataFrame())
    X_all = _collect_tabular(df_all)
    X_all = X_all.reindex(columns=feature_names, fill_value=np.nan)

    proba = pipe.predict_proba(X_all)[:,1]
    out = df_all.copy()
    out["prob_planet"] = proba
    out["class"] = np.where(out["prob_planet"]>=THRESH_PLANET, "planet",
                    np.where(out["prob_planet"]>=THRESH_CANDIDATE, "candidate", "non-planet"))
    preds_path = ART_DIR / "predictions_lgbm.csv"
    out.to_csv(preds_path, index=False)
    print(f"[OK] predições em {preds_path}")

    return preds_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-train", action="store_true", help="Apenas treinar e salvar modelo.")
    ap.add_argument("--predict", action="store_true", help="Treinar e gerar predictions no(s) CSV(s) padrões.")
    ap.add_argument("--predict-only", action="store_true", help="Só prever usando modelo salvo (requer --infile).")
    ap.add_argument("--infile", type=str, default=None, help="CSV único para treinar/prever.")
    args = ap.parse_args()

    if args.predict_only:
        infile = Path(args.infile) if args.infile else None
        train_and_predict(infile=infile, predict_only=True)
    elif args.only_train:
        train_and_predict(infile=Path(args.infile) if args.infile else None, predict_only=False)
    elif args.predict:
        train_and_predict(infile=Path(args.infile) if args.infile else None, predict_only=False)
    else:
        train_and_predict()
