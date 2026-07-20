"""
================================================================================
 PHASE 1 PIPELINE — Data Understanding and Preprocessing
 Data Mining - Kelompok 7 (UCI Bank Marketing Dataset)
================================================================================

Pipeline ini adalah versi ".py" dari notebook
`Phase_1_Data_Understanding_and_Preprocessing.ipynb`, diorkestrasi dengan
Prefect (https://www.prefect.io/) sehingga setiap langkah preprocessing
menjadi task yang terpisah, dapat dipantau, dan dapat di-retry.

Cara menjalankan (dari dalam folder Phase_1):
    pip install -r ../requirements.txt
    python pipeline_phase1.py

Output yang dihasilkan (folder ./output):
    - raw_data.csv              -> salinan data mentah (belum diproses sama
                                    sekali), dipakai ulang oleh Phase 3 & 4
    - bank_marketing_clean.csv  -> dataset bersih hasil seluruh preprocessing
                                    (deliverable resmi notebook, sebelum
                                    seleksi 15 fitur terbaik)
    - X_selected.csv            -> 15 fitur terpilih (hasil correlation +
                                    mutual information filtering), dipakai
                                    Phase 2 (clustering) & Phase 4 (anomaly)
    - y_target.csv               -> target is_subscribed, sejajar indexnya
                                    dengan X_selected.csv
    - selected_features.json    -> daftar nama 15 fitur terpilih
    - figures/*.png             -> seluruh visualisasi yang ada di notebook

Semua path memakai path relatif terhadap lokasi file ini (__file__), jadi
skrip ini bisa dijalankan dari direktori mana pun.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend -> aman dijalankan tanpa display/GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler

import os
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")

from prefect import flow, get_run_logger, task

# --------------------------------------------------------------------------
# Konfigurasi path & konstanta
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Sumber data asli dari notebook. Bisa dioverride lewat environment variable
# BANK_DATA_SOURCE (misalnya untuk pengujian lokal dengan file CSV lokal).
DEFAULT_DATA_SOURCE = (
    "https://drive.google.com/uc?id=1j-m814Njq4DEQfueaP3SFHBvP_zLOrKE"
)
DATA_SOURCE = os.environ.get("BANK_DATA_SOURCE", DEFAULT_DATA_SOURCE)

TOP_N_FEATURES = 15
CORR_THRESHOLD = 0.8


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
@task(retries=2, retry_delay_seconds=5, log_prints=True)
def load_data(source: str) -> pd.DataFrame:
    """Load raw dataset dari sumber (URL Google Drive atau path lokal)."""
    logger = get_run_logger()
    logger.info(f"Membaca dataset dari: {source}")
    data = pd.read_csv(source)
    logger.info(f"Dataset berhasil dimuat. Shape: {data.shape}")
    return data


@task(log_prints=True)
def explore_data(data: pd.DataFrame) -> None:
    """Replikasi seluruh eksplorasi data (EDA) di notebook Phase 1."""
    logger = get_run_logger()
    logger.info(f"Jumlah baris & kolom: {data.shape}")
    logger.info(f"Jumlah nilai unik per kolom:\n{data.nunique()}")
    logger.info(f"Distribusi kolom 'contact':\n{data['contact'].value_counts()}")
    logger.info(f"Distribusi kolom 'poutcome':\n{data['poutcome'].value_counts()}")

    missing = data.isnull().sum().sort_values(ascending=False)
    missing_percent = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
    missing_report = pd.concat(
        [missing, missing_percent], axis=1, keys=["missing_count", "missing_percent"]
    )
    logger.info(f"Laporan missing value:\n{missing_report}")

    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

    # Histogram distribusi kolom numerik
    data[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_histogram_numeric.png", dpi=150)
    plt.close()

    # Boxplot kolom numerik (deteksi outlier awal)
    plt.figure()
    data[numeric_cols].boxplot(figsize=(12, 6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_boxplot_numeric.png", dpi=150)
    plt.close()

    # Ringkasan outlier IQR (eksploratif, sebelum scaling)
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_summary[col] = len(outliers)
    logger.info(f"Jumlah outlier per kolom (IQR, data mentah): {outlier_summary}")


@task(log_prints=True)
def clean_and_rename(data: pd.DataFrame) -> pd.DataFrame:
    """Rename kolom & bersihkan kategori tidak konsisten."""
    logger = get_run_logger()
    df = data.rename(columns={"y": "is_subscribed", "default": "credit_default"})
    df["job"] = df["job"].replace("admin.", "admin")
    n_duplicates = df.duplicated().sum()
    logger.info(f"Jumlah baris duplikat: {n_duplicates}")
    return df


@task(log_prints=True)
def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """RobustScaler pada seluruh kolom numerik (tahan terhadap outlier)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


@task(log_prints=True)
def bin_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binning age, balance, duration menjadi kategori kuartil, lalu drop asli."""
    df = df.copy()
    df["age_bin"] = pd.qcut(df["age"], q=4, labels=["young", "adult", "middle_age", "senior"])
    df["balance_bin"] = pd.qcut(
        df["balance"], q=4, labels=["low", "medium", "high", "very_high"]
    )
    df["duration_bin"] = pd.qcut(
        df["duration"], q=4, labels=["short", "medium", "long", "very_long"]
    )
    df = df.drop(["age", "balance", "duration"], axis=1)
    return df


@task(log_prints=True)
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mapping ordinal (month, education, binary yes/no) + one-hot encoding."""
    df = df.copy()

    month_mapping = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    df["month"] = df["month"].map(month_mapping)

    edu_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    df["education"] = df["education"].map(edu_mapping)

    binary_cols = ["credit_default", "housing", "loan", "is_subscribed"]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    df = pd.get_dummies(
        df, columns=["age_bin", "balance_bin", "duration_bin"], drop_first=True, dtype=int
    )

    df = pd.get_dummies(df, columns=["job", "marital", "contact", "poutcome"], drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


@task(log_prints=True)
def plot_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr_matrix = df[num_cols].corr()

    plt.figure(figsize=(18, 12))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 10}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_correlation_matrix.png", dpi=150)
    plt.close()
    logger.info("Correlation heatmap tersimpan.")
    return corr_matrix


@task(log_prints=True)
def drop_highly_correlated(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split X/y lalu buang fitur dengan korelasi antar-fitur > threshold."""
    logger = get_run_logger()
    X = df.drop("is_subscribed", axis=1)
    y = df["is_subscribed"]

    corr_matrix = X.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(abs(upper[col]) > CORR_THRESHOLD)]
    logger.info(f"Fitur dihapus (correlation > {CORR_THRESHOLD}): {to_drop}")

    X_corr = X.drop(columns=to_drop)
    return X_corr, y, X


@task(log_prints=True)
def select_features_mutual_info(
    X_corr: pd.DataFrame, y: pd.Series, X_full: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Ranking fitur dengan Mutual Information, ambil TOP_N_FEATURES teratas."""
    logger = get_run_logger()
    mi_scores = mutual_info_classif(X_corr, y, random_state=42)
    mi_df = pd.DataFrame(
        {"Feature": X_corr.columns, "MI Score": mi_scores}
    ).sort_values(by="MI Score", ascending=False)
    logger.info(f"Mutual Information scores:\n{mi_df}")

    plt.figure(figsize=(10, 8))
    plt.barh(mi_df["Feature"], mi_df["MI Score"])
    plt.gca().invert_yaxis()
    plt.title("Mutual Information Scores", fontsize=14)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_mutual_information_scores.png", dpi=150)
    plt.close()

    selected_features = mi_df.head(TOP_N_FEATURES)["Feature"].tolist()
    selected_features = [f for f in selected_features if f in X_full.columns]
    X_selected = X_full[selected_features]

    logger.info(f"Jumlah fitur awal: {X_full.shape[1]}")
    logger.info(f"Jumlah fitur terpilih: {X_selected.shape[1]}")
    logger.info(f"Fitur terpilih: {selected_features}")

    return X_selected, selected_features


@task(log_prints=True)
def export_outputs(
    raw_data: pd.DataFrame,
    X_full: pd.DataFrame,
    y: pd.Series,
    X_selected: pd.DataFrame,
    selected_features: list[str],
) -> None:
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw data mentah -> dipakai ulang oleh Phase 3 & Phase 4
    raw_data.to_csv(OUTPUT_DIR / "raw_data.csv", index=False)

    # 2. Dataset bersih hasil preprocessing penuh (deliverable resmi notebook)
    df_clean = X_full.copy()
    df_clean["is_subscribed"] = y
    df_clean.to_csv(OUTPUT_DIR / "bank_marketing_clean.csv", index=False)
    logger.info(
        f"Clean dataset berhasil diexport: {df_clean.shape[0]} baris, "
        f"{df_clean.shape[1]} kolom"
    )

    # 3. 15 fitur terpilih (dipakai Phase 2 & Phase 4)
    X_selected.to_csv(OUTPUT_DIR / "X_selected.csv", index=False)
    y.to_csv(OUTPUT_DIR / "y_target.csv", index=False)

    # 4. Daftar nama fitur terpilih
    with open(OUTPUT_DIR / "selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    logger.info(f"Semua output Phase 1 tersimpan di: {OUTPUT_DIR}")


# --------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------
@flow(name="Phase 1 - Data Understanding and Preprocessing")
def phase1_flow(data_source: str = DATA_SOURCE) -> Path:
    """Flow utama Phase 1. Mengembalikan path folder output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_data_original = load_data(data_source)
    explore_data(raw_data_original)

    # raw_data disimpan sebelum transformasi apa pun (copy asli, dipakai Phase 3 & 4)
    raw_data = raw_data_original.copy()

    df = clean_and_rename(raw_data_original)
    df = scale_numeric(df)
    df = bin_numeric_features(df)
    df = encode_features(df)

    plot_correlation_heatmap(df)
    X_corr, y, X_full = drop_highly_correlated(df)
    X_selected, selected_features = select_features_mutual_info(X_corr, y, X_full)

    export_outputs(raw_data, X_full, y, X_selected, selected_features)

    return OUTPUT_DIR


if __name__ == "__main__":
    result_path = phase1_flow()
    print(f"\n✅ Phase 1 selesai. Semua output tersedia di: {result_path}")
