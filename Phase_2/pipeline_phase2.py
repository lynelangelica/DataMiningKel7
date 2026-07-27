"""
================================================================================
 PHASE 2 PIPELINE — Segmentation via Clustering
 Data Mining - Kelompok 7 (UCI Bank Marketing Dataset)
================================================================================

Versi ".py" dari notebook `Phase_2_Segmentation_via_Clustering.ipynb`,
diorkestrasi dengan Prefect.

Ketergantungan (dependency):
    Phase 2 butuh output Phase 1: `X_selected.csv` dan `raw_data.csv`
    (di ../Phase_1/output/).

    CATATAN PENTING soal sumber data "raw_data":
    Di notebook `Phase_2_Segmentation_via_Clustering.ipynb` (cell 1), baris
    yang benar-benar dieksekusi adalah
    `raw_data = pd.read_csv("../Phase_1/bank_marketing_clean.csv")`,
    sementara baris `# raw_data = pd.read_csv("../Phase_1/raw_data.csv")`
    di-comment. TAPI ini terbukti sisa eksperimen/bug, bukan alur yang
    sebenarnya dipakai: notebook Phase 4 (cell 3, 21, 28) memakai kolom
    mentah asli dari "raw_data" seperti raw_data['age'], raw_data['balance'],
    raw_data['job'], raw_data['housing']=="yes", raw_data['y']=="yes" — semua
    ini MUSTAHIL ada kalau sumbernya bank_marketing_clean.csv, karena di sana
    age/balance/duration sudah di-drop diganti kolom bin, job/marital sudah
    di-one-hot, dan y/housing/loan sudah di-encode jadi 0/1. Karena itu,
    pipeline ini memakai `raw_data.csv` (data mentah asli) sebagai sumber,
    supaya kompatibel dengan Phase 3 & Phase 4 seperti notebook aslinya.

    Jika `raw_data.csv` / `X_selected.csv` belum ada, pipeline ini akan
    MENJALANKAN Phase 1 secara otomatis terlebih dahulu, sehingga Phase 2
    selalu bisa dijalankan sendirian tanpa perlu langkah manual.

Cara menjalankan (dari dalam folder Phase_2):
    pip install -r ../requirements.txt
    python pipeline_phase2.py

Output yang dihasilkan (folder ./output):
    - raw_data_with_clusters.csv -> raw_data (mentah, kolom asli) + kolom
                                     'cluster' (label KMeans), dipakai
                                     Phase 4 untuk cross-reference
    - cluster_profile.csv        -> rata-rata tiap fitur per cluster
    - figures/*.png              -> elbow method, silhouette, heatmap profil,
                                     k-distance graph (untuk eps DBSCAN),
                                     dendrogram (ward/complete/average)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import os
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")

from prefect import flow, get_run_logger, task

# --------------------------------------------------------------------------
# Konfigurasi path
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

PHASE1_DIR = SCRIPT_DIR.parent / "Phase_1"
PHASE1_OUTPUT_DIR = PHASE1_DIR / "output"

OPTIMAL_K = 4
# Nilai eps ini di notebook ditentukan lewat K-Distance Graph + KneeLocator
# (lihat task `find_dbscan_eps`), namun pada akhirnya di-hardcode langsung
# ke DBSCAN sebagai 8.1 (bukan variabel hasil KneeLocator). Kita replikasi
# perilaku itu persis: eps dihitung & di-log untuk transparansi, tapi nilai
# final yang dipakai tetap konstanta ini, sama seperti notebook aslinya.
DBSCAN_EPS = 8.1
DBSCAN_MIN_SAMPLES = 5
DBSCAN_KNN_NEIGHBORS = 5
DENDROGRAM_SAMPLE_SIZE = 3000
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
@task(log_prints=True)
def ensure_phase1_outputs() -> None:
    """Pastikan output Phase 1 tersedia; jalankan Phase 1 jika belum ada."""
    logger = get_run_logger()
    required = [
        PHASE1_OUTPUT_DIR / "X_selected.csv",
        PHASE1_OUTPUT_DIR / "raw_data.csv",
    ]

    if all(p.exists() for p in required):
        logger.info("Output Phase 1 sudah tersedia, lanjut ke Phase 2.")
        return

    logger.warning("Output Phase 1 belum ditemukan. Menjalankan Phase 1 terlebih dahulu...")
    result = subprocess.run(
        [sys.executable, str(PHASE1_DIR / "pipeline_phase1.py")],
        cwd=str(PHASE1_DIR),
        capture_output=True,
        text=True,
    )
    logger.info(result.stdout[-3000:])
    if result.returncode != 0:
        logger.error(result.stderr[-3000:])
        raise RuntimeError(
            "Gagal menjalankan Phase 1 secara otomatis. Jalankan "
            "'python ../Phase_1/pipeline_phase1.py' secara manual terlebih dahulu."
        )
    logger.info("Phase 1 berhasil dijalankan otomatis.")


@task(log_prints=True)
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muat X_selected (fitur terpilih, untuk fitting KMeans/DBSCAN) &
    raw_data (data mentah asli, dipakai untuk menempelkan label cluster
    supaya Phase 3 & Phase 4 tetap bisa membaca kolom aslinya)."""
    logger = get_run_logger()
    X_selected = pd.read_csv(PHASE1_OUTPUT_DIR / "X_selected.csv")
    raw_data = pd.read_csv(PHASE1_OUTPUT_DIR / "raw_data.csv")
    logger.info(f"X_selected: {X_selected.shape}, raw_data: {raw_data.shape}")
    return X_selected, raw_data


@task(log_prints=True)
def find_optimal_k(X_selected: pd.DataFrame) -> None:
    """Elbow method + silhouette score untuk menentukan jumlah cluster optimal."""
    logger = get_run_logger()
    K_range = range(2, 10)

    inertia = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_selected)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(list(K_range), inertia, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(FIGURES_DIR / "01_elbow_method.png", dpi=150)
    plt.close()

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_selected)
        score = silhouette_score(X_selected, labels)
        logger.info(f"K={k}, Silhouette Score={score:.4f}")


@task(log_prints=True)
def run_kmeans(
    X_selected: pd.DataFrame, raw_data: pd.DataFrame, k: int = OPTIMAL_K
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """KMeans final dengan K terpilih, hasilkan cluster_profile & raw_data+cluster."""
    logger = get_run_logger()
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_selected)

    df_cluster = X_selected.copy()
    df_cluster["cluster"] = clusters

    raw_data = raw_data.copy()
    raw_data["cluster"] = clusters

    cluster_profile = df_cluster.groupby("cluster").mean()
    logger.info(f"Cluster profile (rata-rata fitur per cluster):\n{cluster_profile}")

    return df_cluster, raw_data, cluster_profile


@task(log_prints=True)
def plot_cluster_heatmap(cluster_profile: pd.DataFrame) -> None:
    scaler = StandardScaler()
    scaled_profile = scaler.fit_transform(cluster_profile)
    scaled_profile_df = pd.DataFrame(
        scaled_profile, index=cluster_profile.index, columns=cluster_profile.columns
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(scaled_profile_df, cmap="coolwarm", center=0)
    plt.title("Standardized Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_standardized_cluster_heatmap.png", dpi=150)
    plt.close()


@task(log_prints=True)
def find_dbscan_eps(X_selected: pd.DataFrame) -> float:
    """K-Distance Graph + KneeLocator untuk menentukan eps DBSCAN (cell 11-12 notebook)."""
    logger = get_run_logger()
    neighbors = NearestNeighbors(n_neighbors=DBSCAN_KNN_NEIGHBORS)
    neighbors_fit = neighbors.fit(X_selected)
    distances, _ = neighbors_fit.kneighbors(X_selected)
    distances = np.sort(distances[:, DBSCAN_KNN_NEIGHBORS - 1])

    plt.figure()
    plt.plot(distances)
    plt.ylabel("5-NN Distance")
    plt.title("K-Distance Graph untuk menentukan eps")
    plt.savefig(FIGURES_DIR / "03_k_distance_graph.png", dpi=150)
    plt.close()

    kneedle = KneeLocator(
        range(len(distances)), distances, curve="convex", direction="increasing"
    )
    eps_detected = float(distances[kneedle.knee])
    logger.info(f"Nilai eps hasil KneeLocator = {eps_detected:.4f}")
    logger.info(
        f"Eps yang benar-benar dipakai di DBSCAN (sesuai notebook, hardcoded) = {DBSCAN_EPS}"
    )
    return eps_detected


@task(log_prints=True)
def run_dbscan(X_selected: pd.DataFrame, df_cluster: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    db_labels = dbscan.fit_predict(X_selected)

    df_cluster = df_cluster.copy()
    df_cluster["dbscan_cluster"] = db_labels

    logger.info(f"Cluster DBSCAN: {sorted(set(db_labels))}")
    logger.info(f"Jumlah noise (-1): {int(np.sum(db_labels == -1))}")
    return df_cluster


@task(log_prints=True)
def run_hierarchical_dendrograms(X_selected: pd.DataFrame) -> None:
    """Dendrogram ward/complete/average pada sampel 3000 data (efisiensi memori)."""
    logger = get_run_logger()
    sample_size = min(DENDROGRAM_SAMPLE_SIZE, len(X_selected))
    sample = X_selected.sample(sample_size, random_state=RANDOM_STATE)

    methods = ["ward", "complete", "average"]
    for method in methods:
        linked = linkage(sample, method=method)
        plt.figure(figsize=(10, 5))
        dendrogram(linked)
        plt.title(f"Dendrogram - {method}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"04_dendrogram_{method}.png", dpi=150)
        plt.close()
    logger.info(f"Dendrogram dibuat dari sampel {sample_size} baris untuk metode: {methods}")


@task(log_prints=True)
def export_outputs(raw_data_with_clusters: pd.DataFrame, cluster_profile: pd.DataFrame) -> None:
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data_with_clusters.to_csv(OUTPUT_DIR / "raw_data_with_clusters.csv", index=False)
    cluster_profile.to_csv(OUTPUT_DIR / "cluster_profile.csv")

    logger.info(f"Semua output Phase 2 tersimpan di: {OUTPUT_DIR}")


# --------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------
@flow(name="Phase 2 - Segmentation via Clustering")
def phase2_flow(k: int = OPTIMAL_K) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ensure_phase1_outputs()
    X_selected, raw_data = load_inputs()

    find_optimal_k(X_selected)
    df_cluster, raw_data_with_clusters, cluster_profile = run_kmeans(X_selected, raw_data, k)

    plot_cluster_heatmap(cluster_profile)
    find_dbscan_eps(X_selected)
    df_cluster = run_dbscan(X_selected, df_cluster)
    run_hierarchical_dendrograms(X_selected)

    export_outputs(raw_data_with_clusters, cluster_profile)

    return OUTPUT_DIR


if __name__ == "__main__":
    result_path = phase2_flow()
    print(f"\n✅ Phase 2 selesai. Semua output tersedia di: {result_path}")