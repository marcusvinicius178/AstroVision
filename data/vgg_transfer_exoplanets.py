#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vgg_transfer_exoplanets.py
Transfer learning (VGG16) + carregador UNIFICADO para Kepler, K2 e TESS.

Entrada esperada (ajuste caminhos conforme seu ambiente):
  CSVs (qualquer subseto disponível):
    - data/kepler_objects_of_interest.csv
    - data/k2.csv
    - data/TESS.csv
  FITS (se os CSVs não tiverem série temporal):
    - data/fits_kepler/*.fits
    - data/fits_k2/*.fits
    - data/fits_tess/*.fits

Saídas:
  - artifacts/vgg_model_best.h5
  - artifacts/vgg_model.h5
  - artifacts/training_curve.png
  - artifacts/classification_report.txt
  - artifacts/confusion_matrix.npy
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.makedirs("artifacts", exist_ok=True)

import numpy as np
import pandas as pd

# --------- TF / Keras (VGG16) ----------
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import set_random_seed

# --------- I/O FITS ----------
try:
    from astropy.io import fits
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False

# --------- Utils / Plot / Metrics ----------
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ========= GPU setup =========
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Para 1050Ti evitar OOM
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print("GPU mem growth not set:", e)

set_random_seed(42)
np.random.seed(42)

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
IMG_SIZE = 224
DOWNSAMPLE = 64  # pontos por curva antes de virar imagem
VAL_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# ========= Helpers: normalização de rótulos =========
def map_disposition_to_label(series):
    """
    Converte diferentes colunas de disposição (CONFIRMED/CANDIDATE/FALSE POSITIVE etc.)
    para binário: 1=planeta (confirmed/candidate/pc), 0=não-planeta (false positive/eb etc.)
    """
    if series is None:
        return None
    s = series.astype(str).str.lower()
    positives = s.str.contains("confirmed") | s.str.contains("candidate") | s.str.contains("pc")
    negatives = s.str.contains("false") | s.str.contains("eb") | s.str.contains("fp")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[positives] = 1.0
    out[negatives] = 0.0
    # se algo ficou NaN, deixa para tratar depois
    return out

# ========= Loaders unificados =========
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
    """
    Une Kepler/K2/TESS em um único DataFrame padronizado de METADADOS.
    Retorna df_catalog com colunas possíveis: id, label, period, t0, source
    (NÃO garante séries temporais aqui; para curvas usamos FITS ou colunas time/flux).
    """
    dfs = []
    for p in csv_paths:
        df = try_read_csv(p)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()

    cat = pd.concat(dfs, ignore_index=True)

    # Identificador: tentamos várias colunas comuns.
    id_cols = [c for c in cat.columns if str(c).lower() in
               ["kepid","kepoi_name","epic","epic_id","ticid","tic_id","id","object_id","rowid","row_id"]]
    cat["_raw_id"] = None
    if id_cols:
        # prioriza a 1a que existir
        first = id_cols[0]
        cat["_raw_id"] = cat[first]
    else:
        # fallback: cria id incremental
        cat["_raw_id"] = np.arange(len(cat))

    # Disposição / rótulo
    disp_cols = [c for c in cat.columns if str(c).lower() in
                 ["koi_disposition","disposition","disposition_name","label","class"]]
    label = None
    if disp_cols:
        label = map_disposition_to_label(cat[disp_cols[0]])
    # alguns datasets já binários
    if label is None and "is_planet" in [c.lower() for c in cat.columns]:
        col = [c for c in cat.columns if c.lower()=="is_planet"][0]
        label = cat[col].astype(float)

    # período e t0 (se houver)
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
    """
    Tenta detectar colunas 'time' e 'flux' dentro do mesmo CSV (formato já temporal).
    Retorna DataFrame com colunas: id, time, flux, label
    Se não encontrar, retorna DataFrame vazio -> usaremos FITS depois.
    """
    cols = {c.lower():c for c in df.columns}
    time_col = cols.get("time", None)
    # flux pode ser PDCSAP_FLUX, SAP_FLUX, FLUX
    flux_col = next((cols.get(k) for k in ["pdcsap_flux","sap_flux","flux"]), None)
    id_col = next((cols.get(k) for k in ["id","kepid","epic","epic_id","ticid","tic_id","object_id"]), None)
    label_col = next((cols.get(k) for k in ["label","is_planet"]), None)

    # Se não há time e flux, não é um CSV de séries temporais
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
    """
    Percorre os CSVs tentando extrair séries temporais diretas (mesmo arquivo contendo time/flux/id/label).
    Concatena no esquema padrão.
    """
    dfs = []
    for p in csv_paths:
        df = try_read_csv(p)
        if df is None:
            continue
        tf = load_time_flux_from_csv(df)
        if not tf.empty:
            dfs.append(tf)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# -------- FITS fallback ----------
def read_fits_time_flux(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        time = data["TIME"]
        flux = data["PDCSAP_FLUX"] if "PDCSAP_FLUX" in data.columns.names else data["SAP_FLUX"]
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    # normalização (z-score)
    if np.std(flux) > 0:
        flux = (flux - np.mean(flux)) / np.std(flux)
    return time, flux

def harvest_from_fits(dfcatalog, fits_dirs):
    """
    Para ids no catálogo, tenta ler FITS dos diretórios indicados e monta
    DataFrame: id, time, flux, label
    """
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
                # id do arquivo: usa o nome do fits (sem .fits)
                cid = os.path.splitext(fname)[0]
                # label do catálogo, se existir
                label = np.nan
                if not dfcatalog.empty:
                    # tentativas de match por string
                    m = dfcatalog[dfcatalog["id"].astype(str) == cid]
                    if not m.empty and "label" in m.columns:
                        label = float(m["label"].iloc[0]) if pd.notna(m["label"].iloc[0]) else np.nan
                # acumula
                curves.append(pd.DataFrame({"id":[cid]*len(time), "time":time, "flux":flux, "label":[label]*len(time)}))
            except Exception as e:
                print(f"[WARN] Erro FITS {fpath}: {e}")
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

# --------- Pré-proc série temporal ---------
def preprocess_curve(time, flux, outlier_sigma=5.0, resample_len= DOWNSAMPLE):
    # remove outliers (5-sigma)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    mask = np.abs(flux - med) <= outlier_sigma * 1.4826 * mad
    time, flux = time[mask], flux[mask]
    # ordena por tempo
    idx = np.argsort(time)
    time, flux = time[idx], flux[idx]
    # reamostra para comprimento fixo
    if len(time) < 4:
        return None, None
    t_lin = np.linspace(time.min(), time.max(), resample_len)
    f_lin = np.interp(t_lin, time, flux)
    # normaliza z-score
    if np.std(f_lin) > 0:
        f_lin = (f_lin - np.mean(f_lin)) / np.std(f_lin)
    return t_lin, f_lin

def lightcurve_to_image(flux, size=IMG_SIZE, downsample=DOWNSAMPLE):
    # recurrence plot simples
    N = downsample
    if len(flux) != N:
        # assume já downsampleado
        flux = flux[:N] if len(flux) >= N else np.pad(flux, (0, N-len(flux)), mode="edge")
    diff = np.abs(flux[:,None] - flux[None,:])
    eps = np.percentile(diff, 10)
    rp = (diff < eps).astype(np.float32)
    # redimensiona
    from PIL import Image
    img = Image.fromarray((rp*255).astype(np.uint8)).resize((size,size))
    img = np.array(img).astype(np.float32)/255.0
    img = np.stack([img,img,img], axis=-1)
    return img

def build_image_dataset(df_all):
    """
    df_all precisa conter múltiplas linhas por id (time series).
    Retorna X_imgs [N,224,224,3], y_bin [N]
    """
    X, y = [], []
    for cid, grp in df_all.groupby("id"):
        arr = grp.sort_values("time")
        time = arr["time"].values.astype(float)
        flux = arr["flux"].values.astype(float)
        t, f = preprocess_curve(time, flux)
        if t is None: 
            continue
        img = lightcurve_to_image(f)
        X.append(img)
        # label preferimos o mais frequente (ou 1 se existir algum 1)
        lbl = arr["label"].dropna()
        if lbl.empty:
            continue
        y.append( 1 if (lbl.values.mean() >= 0.5) else 0 )
    if not X:
        return np.empty((0,IMG_SIZE,IMG_SIZE,3)), np.array([])
    return np.stack(X, axis=0), np.array(y).astype(int)

# ========= Modelo VGG =========
def build_vgg_model(input_shape=(IMG_SIZE,IMG_SIZE,3), lr=LR):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    # 1) Carrega metadados unificados
    df_cat = unify_catalogs(CSV_PATHS)

    # 2) Tenta extrair séries temporais diretamente dos CSVs (time/flux no próprio CSV)
    df_csv_series = get_all_time_flux_from_csvs(CSV_PATHS)

    # 3) Se não houver séries temporais suficientes, tenta FITS
    if df_csv_series.empty:
        df_fits = harvest_from_fits(df_cat, FITS_DIRS)
        if df_fits.empty:
            print("[ERRO] Nenhuma série temporal encontrada em CSV nem em FITS. Verifique caminhos.")
            sys.exit(1)
        df_all = df_fits
    else:
        # complementa label a partir do catálogo, se faltar
        if not df_cat.empty:
            lab_map = df_cat[["id","label"]].dropna().drop_duplicates().set_index("id")["label"].to_dict()
            df_csv_series["label"] = df_csv_series.apply(
                lambda r: lab_map.get(str(r["id"]), r["label"]), axis=1
            )
        df_all = df_csv_series

    # Limpeza final: remover linhas sem label
    df_all = df_all.dropna(subset=["time","flux"])
    if "label" not in df_all.columns or df_all["label"].isna().all():
        print("[WARN] Nenhum label encontrado nos CSVs; assumindo 0 por padrão (não recomendado).")
        df_all["label"] = 0.0

    # 4) Monta dataset de imagens (1 por id)
    X, y = build_image_dataset(df_all)
    if X.shape[0] == 0:
        print("[ERRO] Não foi possível montar imagens das curvas. Verifique dados.")
        sys.exit(1)

    # 5) Split
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=VAL_SIZE, stratify=y, random_state=42)

    # 6) Modelo
    model = build_vgg_model()

    # 7) Callbacks
    es  = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rlp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    mc  = callbacks.ModelCheckpoint("artifacts/vgg_model_best.h5", monitor="val_loss", save_best_only=True)

    # 8) Treino
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight="balanced",
        callbacks=[es, rlp, mc]
    )

    # 9) Salvar modelo final
    model.save("artifacts/vgg_model.h5")

    # 10) Curvas de treino
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/training_curve.png")

    # 11) Avaliação
    ypred = (model.predict(Xval) > 0.5).astype(int).ravel()
    rep = classification_report(yval, ypred)
    cm  = confusion_matrix(yval, ypred)
    with open("artifacts/classification_report.txt","w") as f:
        f.write(rep)
    np.save("artifacts/confusion_matrix.npy", cm)
    print(rep)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
