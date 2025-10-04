#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_exoplanets.py
Ensemble híbrido: TSFresh + LightGBM + RF + SVM (+ meta Stacking) + fusão com score do VGG.

Reuso do carregador UNIFICADO para Kepler/K2/TESS.
Gera:
  - artifacts/precision_recall_curve.png
  - artifacts/ensemble_report.txt
  - artifacts/ensemble_confusion.npy
  - artifacts/lgbm_best.txt (params)
  - artifacts/shap_values.npy  (opcional)

Requisitos:
  pip install tsfresh lightgbm shap scikit-learn pandas numpy astropy pillow
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.makedirs("artifacts", exist_ok=True)

import numpy as np
import pandas as pd

# FITS / imagens
try:
    from astropy.io import fits
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False

# TSFresh
from tsfresh import extract_features

# ML
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# LightGBM
import lightgbm as lgb

# Keras (para carregar VGG e obter score visual)
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# ========= Config =========
CSV_PATHS = [
    "data/kepler_objects_of_interest.csv",
    "data/k2.csv",
    "data/TESS.csv",
]
FITS_DIRS = {
    "kepler": "data/fits_kepler",
    "k2":     "data/fits_k2",
    "tess":   "data/fits_tess",
}
DOWNSAMPLE = 64
IMG_SIZE = 224
VAL_SIZE = 0.2
SEED = 42

# ========= Helpers compartilhados =========
def map_disposition_to_label(series):
    if series is None:
        return None
    s = series.astype(str).str.lower()
    positives = s.str.contains("confirmed") | s.str.contains("candidate") | s.str.contains("pc")
    negatives = s.str.contains("false") | s.str.contains("eb") | s.str.contains("fp")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[positives] = 1.0
    out[negatives] = 0.0
    return out

def try_read_csv(path):
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        df["__source_csv__"] = os.path.basename(path)
        return df
    except Exception as e:
        print(f"[WARN] Falha ao ler {path}: {e}")
        return None

def unify_catalogs(csv_paths):
    dfs = []
    for p in csv_paths:
        df = try_read_csv(p)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    cat = pd.concat(dfs, ignore_index=True)
    id_cols = [c for c in cat.columns if str(c).lower() in
               ["kepid","kepoi_name","epic","epic_id","ticid","tic_id","id","object_id","rowid","row_id"]]
    cat["_raw_id"] = cat[id_cols[0]] if id_cols else np.arange(len(cat))
    disp_cols = [c for c in cat.columns if str(c).lower() in
                 ["koi_disposition","disposition","disposition_name","label","class"]]
    label = map_disposition_to_label(cat[disp_cols[0]]) if disp_cols else None
    if label is None and "is_planet" in [c.lower() for c in cat.columns]:
        col = [c for c in cat.columns if c.lower()=="is_planet"][0]
        label = cat[col].astype(float)
    per_cols = [c for c in cat.columns if "period" in str(c).lower() or str(c).lower()=="koi_period"]
    t0_cols  = [c for c in cat.columns if "t0" in str(c).lower() or "epoch" in str(c).lower() or str(c).lower()=="koi_time0bk"]
    out = pd.DataFrame({
        "id": cat["_raw_id"],
        "label": label if label is not None else np.nan,
        "period": cat[per_cols[0]] if per_cols else np.nan,
        "t0": cat[t0_cols[0]] if t0_cols else np.nan,
        "source": cat["__source_csv__"] if "__source_csv__" in cat.columns else "unknown_csv"
    })
    return out

def load_time_flux_from_csv(df):
    cols = {c.lower():c for c in df.columns}
    time_col = cols.get("time", None)
    flux_col = next((cols.get(k) for k in ["pdcsap_flux","sap_flux","flux"]), None)
    id_col = next((cols.get(k) for k in ["id","kepid","epic","epic_id","ticid","tic_id","object_id"]), None)
    label_col = next((cols.get(k) for k in ["label","is_planet"]), None)
    if time_col is None or flux_col is None or id_col is None:
        return pd.DataFrame()
    tmp = df[[id_col, time_col, flux_col]].copy()
    tmp.columns = ["id","time","flux"]
    if label_col:
        tmp["label"] = df[label_col].astype(float)
    else:
        tmp["label"] = np.nan
    return tmp

def get_all_time_flux_from_csvs(csv_paths):
    dfs = []
    for p in csv_paths:
        df = try_read_csv(p)
        if df is None:
            continue
        tf = load_time_flux_from_csv(df)
        if not tf.empty:
            dfs.append(tf)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def read_fits_time_flux(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        time = data["TIME"]
        flux = data["PDCSAP_FLUX"] if "PDCSAP_FLUX" in data.columns.names else data["SAP_FLUX"]
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    if np.std(flux) > 0:
        flux = (flux - np.mean(flux)) / np.std(flux)
    return time, flux

def harvest_from_fits(dfcatalog, fits_dirs):
    if not ASTROPY_OK:
        print("[WARN] astropy não disponível; não é possível ler FITS.")
        return pd.DataFrame()
    curves = []
    for miss in ["kepler","k2","tess"]:
        base = fits_dirs.get(miss, None)
        if not base or not os.path.isdir(base):
            continue
        for fname in os.listdir(base):
            if not fname.endswith(".fits"):
                continue
            fpath = os.path.join(base, fname)
            try:
                time, flux = read_fits_time_flux(fpath)
                cid = os.path.splitext(fname)[0]
                label = np.nan
                if not dfcatalog.empty:
                    m = dfcatalog[dfcatalog["id"].astype(str) == cid]
                    if not m.empty and "label" in m.columns:
                        label = float(m["label"].iloc[0]) if pd.notna(m["label"].iloc[0]) else np.nan
                curves.append(pd.DataFrame({"id":[cid]*len(time), "time":time, "flux":flux, "label":[label]*len(time)}))
            except Exception as e:
                print(f"[WARN] Erro FITS {fpath}: {e}")
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

def preprocess_curve(time, flux, outlier_sigma=5.0, resample_len=DOWNSAMPLE):
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    mask = np.abs(flux - med) <= outlier_sigma * 1.4826 * mad
    time, flux = time[mask], flux[mask]
    idx = np.argsort(time)
    time, flux = time[idx], flux[idx]
    if len(time) < 4:
        return None, None
    t_lin = np.linspace(time.min(), time.max(), resample_len)
    f_lin = np.interp(t_lin, time, flux)
    if np.std(f_lin) > 0:
        f_lin = (f_lin - np.mean(f_lin)) / np.std(f_lin)
    return t_lin, f_lin

def lightcurve_to_image(flux, size=IMG_SIZE, downsample=DOWNSAMPLE):
    N = downsample
    if len(flux) != N:
        flux = flux[:N] if len(flux) >= N else np.pad(flux, (0,N-len(flux)), mode="edge")
    diff = np.abs(flux[:,None] - flux[None,:])
    eps = np.percentile(diff, 10)
    rp = (diff < eps).astype(np.float32)
    from PIL import Image
    img = Image.fromarray((rp*255).astype(np.uint8)).resize((size,size))
    img = np.array(img).astype(np.float32)/255.0
    img = np.stack([img,img,img], axis=-1)
    return img

def build_image_and_feature_sets(df_all):
    """
    Gera:
      - X_imgs (1 img por id)
      - y_bin  (1 rótulo por id)
      - X_feats_ts (features TSFresh por id)
    """
    # 1) Imagens
    Ximgs, ybin = [], []
    # 2) Features TSFresh: precisamos montar DataFrame longo
    df_long = []
    ids = []
    for cid, grp in df_all.groupby("id"):
        arr = grp.sort_values("time")
        time = arr["time"].values.astype(float)
        flux = arr["flux"].values.astype(float)
        t, f = preprocess_curve(time, flux)
        if t is None:
            continue
        # imagem
        img = lightcurve_to_image(f)
        Ximgs.append(img)
        lbl = arr["label"].dropna()
        if lbl.empty:
            continue
        ybin.append(1 if (lbl.values.mean() >= 0.5) else 0)
        # TSFresh long
        df_long.append(pd.DataFrame({"id":[len(ids)]*len(t), "time":t, "flux":f}))
        ids.append(cid)
    if not Ximgs:
        return np.empty((0,IMG_SIZE,IMG_SIZE,3)), np.array([]), pd.DataFrame(), []
    Ximgs = np.stack(Ximgs, axis=0)
    ybin = np.array(ybin).astype(int)
    df_long = pd.concat(df_long, ignore_index=True)
    return Ximgs, ybin, df_long, ids

def main():
    # 1) Catálogo e séries
    df_cat = unify_catalogs(CSV_PATHS)
    df_csv_series = get_all_time_flux_from_csvs(CSV_PATHS)
    if df_csv_series.empty:
        df_fits = harvest_from_fits(df_cat, FITS_DIRS)
        if df_fits.empty:
            print("[ERRO] Sem séries temporais em CSV ou FITS.")
            sys.exit(1)
        df_all = df_fits
    else:
        if not df_cat.empty:
            lab_map = df_cat[["id","label"]].dropna().drop_duplicates().set_index("id")["label"].to_dict()
            df_csv_series["label"] = df_csv_series.apply(
                lambda r: lab_map.get(str(r["id"]), r["label"]), axis=1
            )
        df_all = df_csv_series
    df_all = df_all.dropna(subset=["time","flux"])
    if "label" not in df_all.columns or df_all["label"].isna().all():
        print("[WARN] Sem labels; assumindo 0 (não recomendado).")
        df_all["label"] = 0.0

    # 2) Monta imagens + TSFresh (por id)
    Ximgs, ybin, df_long, id_list = build_image_and_feature_sets(df_all)
    if Ximgs.shape[0] == 0:
        print("[ERRO] Não foi possível gerar dataset.")
        sys.exit(1)

    # 3) TSFresh features
    X_feats = extract_features(df_long, column_id="id", column_sort="time")
    X_feats = X_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4) Split (mesmo índice para imagens e features)
    Xtr_i, Xval_i, ytr, yval = train_test_split(Ximgs, ybin, test_size=VAL_SIZE, stratify=ybin, random_state=SEED)
    # Precisamos dividir X_feats no mesmo corte — criaremos índices consistentes
    idx_all = np.arange(len(ybin))
    tr_idx, val_idx, _, _ = train_test_split(idx_all, ybin, test_size=VAL_SIZE, stratify=ybin, random_state=SEED)
    Xtr_f = X_feats.iloc[tr_idx].values
    Xval_f = X_feats.iloc[val_idx].values

    # 5) Score visual do VGG (se existir modelo treinado)
    vgg_path = "artifacts/vgg_model.h5"
    if os.path.isfile(vgg_path):
        vgg = load_model(vgg_path)
        proba_vgg_tr  = vgg.predict(Xtr_i).ravel()
        proba_vgg_val = vgg.predict(Xval_i).ravel()
    else:
        print("[WARN] artifacts/vgg_model.h5 não encontrado; score visual será zero.")
        proba_vgg_tr  = np.zeros(len(ytr))
        proba_vgg_val = np.zeros(len(yval))

    # 6) Modelos base em features TSFresh
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # LightGBM (GPU se disponível)
    lgbm = lgb.LGBMClassifier(device="gpu", random_state=SEED, class_weight="balanced")
    grid_lgb = GridSearchCV(
        lgbm,
        {"num_leaves":[31,63], "learning_rate":[0.01,0.1], "n_estimators":[200,400]},
        cv=cv, scoring="f1", n_jobs=2
    )
    grid_lgb.fit(Xtr_f, ytr)
    best_lgb = grid_lgb.best_estimator_
    with open("artifacts/lgbm_best.txt","w") as f:
        f.write(str(grid_lgb.best_params_))

    # Random Forest
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED)
    rf.fit(Xtr_f, ytr)

    # SVM
    svm = SVC(probability=True, class_weight="balanced", random_state=SEED)
    svm.fit(Xtr_f, ytr)

    # 7) Probabilidades (val)
    p_lgb_tr = best_lgb.predict_proba(Xtr_f)[:,1]
    p_rf_tr  = rf.predict_proba(Xtr_f)[:,1]
    p_svm_tr = svm.predict_proba(Xtr_f)[:,1]

    p_lgb = best_lgb.predict_proba(Xval_f)[:,1]
    p_rf  = rf.predict_proba(Xval_f)[:,1]
    p_svm = svm.predict_proba(Xval_f)[:,1]

    # 8) Meta-learner (Logistic Regression) com fusão dos 4 scores
    Xmeta_tr  = np.vstack([p_lgb_tr, p_rf_tr, p_svm_tr, proba_vgg_tr]).T
    Xmeta_val = np.vstack([p_lgb,    p_rf,    p_svm,    proba_vgg_val]).T

    meta = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=200)
    meta.fit(Xmeta_tr, ytr)

    yscore = meta.predict_proba(Xmeta_val)[:,1]
    ypred  = (yscore >= 0.5).astype(int)

    # 9) Métricas
    rep = classification_report(yval, ypred)
    cm  = confusion_matrix(yval, ypred)
    with open("artifacts/ensemble_report.txt","w") as f:
        f.write(rep)
    np.save("artifacts/ensemble_confusion.npy", cm)
    print(rep); print(cm)

    # PR-AUC
    prec, rec, _ = precision_recall_curve(yval, yscore)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.title("Precision-Recall (Ensemble)")
    plt.savefig("artifacts/precision_recall_curve.png")

    # 10) SHAP (opcional) para LightGBM
    try:
        import shap
        expl = shap.TreeExplainer(best_lgb)
        shap_values = expl.shap_values(Xval_f)
        np.save("artifacts/shap_values.npy", shap_values)
    except Exception as e:
        print("[WARN] SHAP não gerado:", e)

if __name__ == "__main__":
    main()
