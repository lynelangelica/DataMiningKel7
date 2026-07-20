"""
================================================================================
 PHASE 4 PIPELINE — Anomaly and Outlier Detection
 Data Mining - Kelompok 7 (UCI Bank Marketing Dataset)
================================================================================

Versi ".py" dari notebook `Phase_4_Anomaly_and_Outlier_Detection.ipynb`,
diorkestrasi dengan Prefect. Ini adalah phase terakhir dan menghasilkan
seluruh file CSV yang dipakai dashboard.

Ketergantungan (dependency):
    - Phase 1 (../Phase_1/output/): raw_data.csv, X_selected.csv,
      selected_features.json
    - Phase 2 (../Phase_2/output/): raw_data_with_clusters.csv (kolom 'cluster')
    - Phase 3 (../Phase_3/output/): association_rules_full.pkl

    Jika salah satu belum tersedia, pipeline ini akan menjalankan phase
    yang bersangkutan secara otomatis (Phase 1 -> Phase 2 -> Phase 3),
    sehingga Phase 4 selalu bisa dijalankan sendirian tanpa langkah manual.

Cara menjalankan (dari dalam folder Phase_4):
    pip install -r ../requirements.txt
    python pipeline_phase4.py

Output yang dihasilkan (folder ./output):
    - cluster_result.csv          -> untuk pie chart & scatter segmentasi
    - outlier_result.csv          -> semua baris + anomaly_score + label Outlier/Normal
    - outlier_classification.csv  -> 10 anomali teratas + klasifikasi & alasan bisnis
    - cluster_summary.csv         -> profil tiap cluster dalam skala asli
    - association_rules.csv       -> gabungan top 10 rule positif & negatif
    - dashboard_kpi.csv           -> ringkasan KPI satu baris
    - figures/*.png               -> seluruh visualisasi Phase 4
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
from scipy import stats
from sklearn.ensemble import IsolationForest

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
PHASE2_DIR = SCRIPT_DIR.parent / "Phase_2"
PHASE3_DIR = SCRIPT_DIR.parent / "Phase_3"

PHASE1_OUTPUT_DIR = PHASE1_DIR / "output"
PHASE2_OUTPUT_DIR = PHASE2_DIR / "output"
PHASE3_OUTPUT_DIR = PHASE3_DIR / "output"

NUM_COLS = ["age", "balance", "duration", "campaign", "pdays", "previous"]
ISO_CONTAMINATION = 0.05
ISO_N_ESTIMATORS = 200
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Helper: menjalankan phase lain via subprocess (self-healing dependency)
# --------------------------------------------------------------------------
def _run_phase_script(phase_dir: Path, script_name: str, logger) -> None:
    result = subprocess.run(
        [sys.executable, str(phase_dir / script_name)],
        cwd=str(phase_dir),
        capture_output=True,
        text=True,
    )
    logger.info(result.stdout[-3000:])
    if result.returncode != 0:
        logger.error(result.stderr[-3000:])
        raise RuntimeError(
            f"Gagal menjalankan {script_name} secara otomatis. Jalankan "
            f"'python {phase_dir}/{script_name}' secara manual terlebih dahulu."
        )


# --------------------------------------------------------------------------
# Tasks: memastikan dependency dari phase sebelumnya tersedia
# --------------------------------------------------------------------------
@task(log_prints=True)
def ensure_phase1_outputs() -> None:
    logger = get_run_logger()
    required = [
        PHASE1_OUTPUT_DIR / "raw_data.csv",
        PHASE1_OUTPUT_DIR / "X_selected.csv",
        PHASE1_OUTPUT_DIR / "selected_features.json",
    ]
    if all(p.exists() for p in required):
        logger.info("Output Phase 1 sudah tersedia.")
        return
    logger.warning("Output Phase 1 belum ditemukan. Menjalankan Phase 1 terlebih dahulu...")
    _run_phase_script(PHASE1_DIR, "pipeline_phase1.py", logger)
    logger.info("Phase 1 berhasil dijalankan otomatis.")


@task(log_prints=True)
def ensure_phase2_outputs() -> None:
    logger = get_run_logger()
    required = PHASE2_OUTPUT_DIR / "raw_data_with_clusters.csv"
    if required.exists():
        logger.info("Output Phase 2 sudah tersedia.")
        return
    logger.warning("Output Phase 2 belum ditemukan. Menjalankan Phase 2 terlebih dahulu...")
    _run_phase_script(PHASE2_DIR, "pipeline_phase2.py", logger)
    logger.info("Phase 2 berhasil dijalankan otomatis.")


@task(log_prints=True)
def ensure_phase3_outputs() -> None:
    logger = get_run_logger()
    required = PHASE3_OUTPUT_DIR / "association_rules_full.pkl"
    if required.exists():
        logger.info("Output Phase 3 sudah tersedia.")
        return
    logger.warning("Output Phase 3 belum ditemukan. Menjalankan Phase 3 terlebih dahulu...")
    _run_phase_script(PHASE3_DIR, "pipeline_phase3.py", logger)
    logger.info("Phase 3 berhasil dijalankan otomatis.")


@task(log_prints=True)
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    logger = get_run_logger()
    import json

    raw_data = pd.read_csv(PHASE2_OUTPUT_DIR / "raw_data_with_clusters.csv")
    X_selected = pd.read_csv(PHASE1_OUTPUT_DIR / "X_selected.csv")
    with open(PHASE1_OUTPUT_DIR / "selected_features.json") as f:
        selected_features = json.load(f)
    rules_sorted = pd.read_pickle(PHASE3_OUTPUT_DIR / "association_rules_full.pkl")

    logger.info(
        f"raw_data (w/ cluster): {raw_data.shape}, X_selected: {X_selected.shape}, "
        f"rules_sorted: {rules_sorted.shape}"
    )
    return raw_data, X_selected, selected_features, rules_sorted


def display_clean_rules(rules_df: pd.DataFrame, target: str = "y_yes", top_n: int = 10) -> pd.DataFrame:
    """Sama persis dengan versi di Phase 3 -- diduplikasi di sini agar Phase 4
    tetap bisa berjalan mandiri tanpa perlu import lintas-folder."""
    filtered = rules_df[rules_df["consequents"].apply(lambda x: target in x)].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered = filtered[filtered["antecedents"].apply(lambda x: len(x) <= 3)]
    filtered = filtered.sort_values(by=["lift", "confidence", "support"], ascending=False)
    filtered["pattern"] = filtered["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    filtered = filtered.drop_duplicates(subset="pattern", keep="first")
    filtered = filtered.head(top_n).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Rank": range(1, len(filtered) + 1),
            "Customer Pattern": filtered["pattern"],
            "Prediction": target,
            "Support": filtered["support"].round(4),
            "Confidence": filtered["confidence"].round(4),
            "Lift": filtered["lift"].round(4),
        }
    )


# --------------------------------------------------------------------------
# Tasks: analisis Phase 4
# --------------------------------------------------------------------------
@task(log_prints=True)
def detect_statistical_outliers(raw_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """IQR & Z-Score outlier detection pada 6 kolom numerik asli."""
    logger = get_run_logger()
    df_stat = raw_data[NUM_COLS].copy()

    # --- IQR ---
    Q1 = df_stat.quantile(0.25)
    Q3 = df_stat.quantile(0.75)
    IQR = Q3 - Q1
    iqr_mask = (df_stat < (Q1 - 1.5 * IQR)) | (df_stat > (Q3 + 1.5 * IQR))
    iqr_outlier = iqr_mask.any(axis=1)
    logger.info(
        f"Total baris dengan outlier IQR (min 1 kolom): {iqr_outlier.sum():,} "
        f"({iqr_outlier.mean()*100:.2f}%)"
    )

    # --- Z-Score ---
    z_scores = np.abs(stats.zscore(df_stat))
    z_mask = z_scores > 3
    z_outlier = pd.Series(z_mask.any(axis=1), index=df_stat.index)
    logger.info(
        f"Total baris dengan outlier Z-Score (min 1 kolom): {z_outlier.sum():,} "
        f"({z_outlier.mean()*100:.2f}%)"
    )

    # Boxplot per kolom
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUM_COLS):
        axes[i].boxplot(
            df_stat[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="navy"),
            medianprops=dict(color="crimson", linewidth=2),
            flierprops=dict(marker="o", color="crimson", alpha=0.3, markersize=3),
        )
        axes[i].set_title(f"{col}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Value")
        axes[i].grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Boxplot IQR — Deteksi Outlier per Kolom Numerik", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_boxplot_iqr.png", dpi=150)
    plt.close()

    return iqr_outlier, z_outlier


@task(log_prints=True)
def detect_isolation_forest_outliers(
    X_selected: pd.DataFrame, raw_data: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    logger = get_run_logger()
    raw_data = raw_data.copy()

    iso_forest = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=ISO_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_labels = iso_forest.fit_predict(X_selected)
    iso_scores = iso_forest.decision_function(X_selected)

    raw_data["iso_label"] = iso_labels
    raw_data["iso_score"] = iso_scores

    n_anomalies = int((iso_labels == -1).sum())
    logger.info(
        f"Isolation Forest — Anomali terdeteksi: {n_anomalies:,} "
        f"({n_anomalies/len(iso_labels)*100:.2f}%)"
    )
    logger.info(f"Data normal: {int((iso_labels == 1).sum()):,}")

    # Histogram distribusi skor + boxplot per cluster (kalau kolom cluster ada)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(iso_scores[iso_labels == 1], bins=60, alpha=0.7, color="steelblue", label="Normal")
    axes[0].hist(iso_scores[iso_labels == -1], bins=60, alpha=0.7, color="crimson", label="Anomaly")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2, label="Threshold = 0")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Anomaly Scores")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    if "cluster" in raw_data.columns:
        cluster_ids = sorted(raw_data["cluster"].unique())
        bp_data = [raw_data.loc[raw_data["cluster"] == k, "iso_score"] for k in cluster_ids]
        _boxplot_label_kwarg = (
            {"tick_labels": [f"Cluster {k}" for k in cluster_ids]}
            if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 9)
            else {"labels": [f"Cluster {k}" for k in cluster_ids]}
        )
        axes[1].boxplot(
            bp_data,
            patch_artist=True,
            **_boxplot_label_kwarg,
            boxprops=dict(facecolor="lightblue", color="navy"),
            medianprops=dict(color="crimson", linewidth=2),
        )
        axes[1].axhline(0, color="crimson", linestyle="--", linewidth=1, label="Anomaly Threshold")
        axes[1].set_ylabel("Anomaly Score")
        axes[1].set_title("Anomaly Score by Cluster")
        axes[1].legend()
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_isolation_forest_scores.png", dpi=150)
    plt.close()

    return raw_data, iso_labels, iso_scores


@task(log_prints=True)
def compare_methods(
    raw_data: pd.DataFrame, iqr_outlier: pd.Series, z_outlier: pd.Series, iso_labels: np.ndarray
) -> pd.DataFrame:
    logger = get_run_logger()
    comparison_df = pd.DataFrame(
        {
            "Detection Method": ["IQR", "Z-Score", "Isolation Forest"],
            "Detected Anomalies": [
                int(iqr_outlier.sum()),
                int(z_outlier.sum()),
                int((iso_labels == -1).sum()),
            ],
        }
    )
    comparison_df["Percentage (%)"] = (
        comparison_df["Detected Anomalies"] / len(raw_data) * 100
    ).round(2)
    logger.info(f"Perbandingan metode deteksi anomali:\n{comparison_df}")
    return comparison_df


@task(log_prints=True)
def cross_reference_with_clusters(
    raw_data: pd.DataFrame, iqr_outlier: pd.Series, z_outlier: pd.Series
) -> pd.DataFrame:
    logger = get_run_logger()

    if "cluster" not in raw_data.columns:
        logger.warning("Kolom 'cluster' tidak ditemukan; melewati cross-reference.")
        return pd.DataFrame()

    cross_reference = raw_data.groupby("cluster").agg(
        Total_Data=("cluster", "count"),
        IQR_Anomaly=("cluster", lambda x: iqr_outlier[x.index].sum()),
        ZScore_Anomaly=("cluster", lambda x: z_outlier[x.index].sum()),
        IsolationForest_Anomaly=("iso_label", lambda x: (x == -1).sum()),
    )
    cross_reference["IQR (%)"] = (cross_reference["IQR_Anomaly"] / cross_reference["Total_Data"] * 100).round(2)
    cross_reference["Z-Score (%)"] = (cross_reference["ZScore_Anomaly"] / cross_reference["Total_Data"] * 100).round(2)
    cross_reference["Isolation Forest (%)"] = (
        cross_reference["IsolationForest_Anomaly"] / cross_reference["Total_Data"] * 100
    ).round(2)

    logger.info(f"Cross-reference cluster vs anomaly:\n{cross_reference}")

    ax = cross_reference[["IQR_Anomaly", "ZScore_Anomaly", "IsolationForest_Anomaly"]].plot(
        kind="bar", figsize=(8, 5)
    )
    ax.set_title("Comparison of Detected Anomalies Across Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Anomalies")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_anomalies_across_clusters.png", dpi=150)
    plt.close()

    return cross_reference


def classify_with_reason(row: pd.Series) -> tuple[str, str]:
    """Rule-based classifier untuk 10 anomali teratas -> (kategori, alasan bisnis)."""
    if row["balance"] < 0 and row["previous"] >= 10:
        return (
            "Potential Risk Signal",
            "Nasabah memiliki saldo negatif tetapi telah dihubungi berkali-kali. "
            "Hal ini mengindikasikan adanya upaya pemasaran yang intensif terhadap "
            "pelanggan dengan potensi konversi yang rendah.",
        )
    if row["balance"] >= 10000:
        return (
            "Rare Legitimate Case",
            "Saldo rekening jauh lebih tinggi dibandingkan mayoritas pelanggan "
            "sehingga kemungkinan merupakan nasabah bernilai tinggi (high-value "
            "customer), bukan kesalahan data.",
        )
    if row["previous"] >= 10:
        return (
            "Potential Risk Signal",
            f"Nasabah telah dihubungi sebelumnya sebanyak {row['previous']} kali. "
            "Frekuensi kontak yang sangat tinggi menunjukkan pola pemasaran yang "
            "tidak biasa dan perlu dievaluasi efektivitasnya.",
        )
    if row["campaign"] >= 6:
        return (
            "Potential Risk Signal",
            f"Nasabah telah dihubungi sebanyak {row['campaign']} kali pada kampanye "
            "saat ini. Intensitas kampanye yang tinggi dapat mengindikasikan "
            "strategi pemasaran yang kurang efektif.",
        )
    if row["duration"] >= 1000:
        return (
            "Needs Further Investigation",
            "Durasi percakapan jauh lebih panjang dibandingkan mayoritas "
            "pelanggan. Kondisi ini menunjukkan perilaku pelanggan yang tidak "
            "umum sehingga perlu dianalisis lebih lanjut.",
        )
    return (
        "Needs Further Investigation",
        "Karakteristik pelanggan berbeda dari mayoritas populasi, namun belum "
        "terdapat indikator yang cukup untuk menentukan penyebab utama anomali.",
    )


@task(log_prints=True)
def flag_and_classify_top_anomalies(raw_data: pd.DataFrame, iso_scores: np.ndarray) -> pd.DataFrame:
    logger = get_run_logger()

    flagged = raw_data[raw_data["iso_label"] == -1].copy()
    flagged["anomaly_score"] = iso_scores[raw_data["iso_label"] == -1]
    flagged = flagged.sort_values("anomaly_score").head(10).reset_index(drop=True)

    flagged[["Classification", "Reason"]] = flagged.apply(
        lambda row: pd.Series(classify_with_reason(row)), axis=1
    )

    columns_to_show = [
        "cluster", "age", "balance", "duration", "campaign", "pdays", "previous",
        "anomaly_score", "Classification", "Reason",
    ]
    columns_to_show = [c for c in columns_to_show if c in flagged.columns]
    logger.info(f"Top 10 flagged anomalies:\n{flagged[columns_to_show]}")

    return flagged[columns_to_show].round({"anomaly_score": 4})


@task(log_prints=True)
def export_dashboard_csvs(
    raw_data: pd.DataFrame,
    iso_labels: np.ndarray,
    iso_scores: np.ndarray,
    outlier_classification: pd.DataFrame,
    rules_sorted: pd.DataFrame,
) -> None:
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. cluster_result.csv
    cluster_cols = [
        c for c in [
            "age", "balance", "duration", "campaign", "pdays", "previous",
            "job", "marital", "education", "housing", "loan", "y", "cluster",
        ] if c in raw_data.columns
    ]
    raw_data[cluster_cols].to_csv(OUTPUT_DIR / "cluster_result.csv", index=False)

    # 2. outlier_result.csv
    outlier_cols = [c for c in ["age", "balance", "duration", "campaign", "pdays", "previous", "cluster"] if c in raw_data.columns]
    outlier_result = raw_data[outlier_cols].copy()
    outlier_result["anomaly_score"] = iso_scores
    outlier_result["Outlier"] = np.where(iso_labels == -1, "Outlier", "Normal")
    outlier_result.to_csv(OUTPUT_DIR / "outlier_result.csv", index=False)

    # 3. outlier_classification.csv
    outlier_classification.to_csv(OUTPUT_DIR / "outlier_classification.csv", index=False)

    # 4. cluster_summary.csv
    if "cluster" in raw_data.columns:
        cluster_summary = raw_data.groupby("cluster").agg(
            size=("cluster", "count"),
            avg_age=("age", "mean"),
            avg_balance=("balance", "mean"),
            avg_duration=("duration", "mean"),
            avg_campaign=("campaign", "mean"),
            avg_pdays=("pdays", "mean"),
            avg_previous=("previous", "mean"),
            pct_housing=("housing", lambda x: (x == "yes").mean() * 100),
            pct_subscribed=("y", lambda x: (x == "yes").mean() * 100),
        ).round(2).reset_index()
        cluster_summary["pct_of_total"] = (cluster_summary["size"] / len(raw_data) * 100).round(2)
        cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    # 5. association_rules.csv
    pos_rules = display_clean_rules(rules_sorted, target="y_yes", top_n=10)
    neg_rules = display_clean_rules(rules_sorted, target="y_no", top_n=10)
    association_rules_export = pd.concat([pos_rules, neg_rules], ignore_index=True)
    association_rules_export.columns = ["rank", "pattern", "prediction", "support", "confidence", "lift"]
    association_rules_export.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)

    # 6. dashboard_kpi.csv
    dashboard_kpi = pd.DataFrame(
        [
            {
                "total_customers": len(raw_data),
                "n_clusters": raw_data["cluster"].nunique() if "cluster" in raw_data.columns else None,
                "top_rules_shown": len(association_rules_export),
                "total_outliers": int((iso_labels == -1).sum()),
            }
        ]
    )
    dashboard_kpi.to_csv(OUTPUT_DIR / "dashboard_kpi.csv", index=False)

    logger.info("6 file CSV dashboard berhasil dibuat di: " + str(OUTPUT_DIR))
    for f in [
        "cluster_result.csv", "outlier_result.csv", "outlier_classification.csv",
        "cluster_summary.csv", "association_rules.csv", "dashboard_kpi.csv",
    ]:
        logger.info(f" - {f}")


# --------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------
@flow(name="Phase 4 - Anomaly and Outlier Detection")
def phase4_flow() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ensure_phase1_outputs()
    ensure_phase2_outputs()
    ensure_phase3_outputs()

    raw_data, X_selected, selected_features, rules_sorted = load_inputs()

    iqr_outlier, z_outlier = detect_statistical_outliers(raw_data)
    raw_data, iso_labels, iso_scores = detect_isolation_forest_outliers(X_selected, raw_data)

    compare_methods(raw_data, iqr_outlier, z_outlier, iso_labels)
    cross_reference_with_clusters(raw_data, iqr_outlier, z_outlier)

    outlier_classification = flag_and_classify_top_anomalies(raw_data, iso_scores)

    export_dashboard_csvs(raw_data, iso_labels, iso_scores, outlier_classification, rules_sorted)

    return OUTPUT_DIR


if __name__ == "__main__":
    result_path = phase4_flow()
    print(f"\n✅ Phase 4 selesai. Semua output dashboard tersedia di: {result_path}")
