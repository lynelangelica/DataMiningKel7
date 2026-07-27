"""
================================================================================
 PHASE 3 PIPELINE — Association Rule Mining
 Data Mining - Kelompok 7 (UCI Bank Marketing Dataset)
================================================================================

Versi ".py" dari notebook `Phase_3_Association_Rule_Mining.ipynb`,
diorkestrasi dengan Prefect.

Ketergantungan (dependency):
    Phase 3 hanya butuh `raw_data.csv` dari Phase 1 (../Phase_1/output/).
    Jika belum ada, pipeline ini akan menjalankan Phase 1 secara otomatis.

Cara menjalankan (dari dalam folder Phase_3):
    pip install -r ../requirements.txt
    python pipeline_phase3.py

Output yang dihasilkan (folder ./output):
    - association_rules_full.pkl  -> seluruh association rules (pandas
                                    DataFrame, disimpan sebagai pickle agar
                                    kolom antecedents/consequents berupa
                                    frozenset tetap utuh) — dipakai Phase 4
    - rules_sorted.csv            -> seluruh association rules dalam format
                                    CSV (setara notebook cell terakhir,
                                    frozenset otomatis jadi string saat
                                    disimpan CSV — untuk pemakaian ulang
                                    programatik pakai file .pkl di atas)
    - top_positive_rules.csv      -> top 10 rules menuju y_yes (nasabah setuju)
    - top_negative_rules.csv      -> top 10 rules menuju y_no  (nasabah tolak)
    - figures/*.png               -> scatter plot support vs confidence vs lift
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
from mlxtend.frequent_patterns import apriori, association_rules

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

PHASE1_DIR = SCRIPT_DIR.parent / "Phase_1"
PHASE1_OUTPUT_DIR = PHASE1_DIR / "output"

MIN_SUPPORT = 0.05
MIN_LIFT = 1.3


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
@task(log_prints=True)
def ensure_phase1_outputs() -> None:
    logger = get_run_logger()
    required = PHASE1_OUTPUT_DIR / "raw_data.csv"

    if required.exists():
        logger.info("Output Phase 1 sudah tersedia, lanjut ke Phase 3.")
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
def load_raw_data() -> pd.DataFrame:
    logger = get_run_logger()
    raw_data = pd.read_csv(PHASE1_OUTPUT_DIR / "raw_data.csv")
    logger.info(f"raw_data: {raw_data.shape}")
    return raw_data


@task(log_prints=True)
def discretize_and_encode(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Diskretisasi variabel kontinu ke kategori bisnis, lalu one-hot encoding."""
    logger = get_run_logger()
    df_association = raw_data.copy()

    age_bins = [0, 25, 40, 60, 100]
    age_labels = ["Age_Under_25", "Age_25_to_40", "Age_41_to_60", "Age_Above_60"]
    df_association["age_group"] = pd.cut(
        df_association["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )

    balance_bins = [-np.inf, 0, 500, 2000, np.inf]
    balance_labels = ["Balance_Negative_or_Zero", "Balance_Low", "Balance_Medium", "Balance_High"]
    df_association["balance_group"] = pd.cut(
        df_association["balance"], bins=balance_bins, labels=balance_labels
    )

    duration_bins = [0, 60, 180, 300, np.inf]
    duration_labels = ["Dur_Very_Short", "Dur_Short", "Dur_Medium", "Dur_Long"]
    df_association["duration_group"] = pd.cut(
        df_association["duration"], bins=duration_bins, labels=duration_labels
    )

    columns_to_drop = ["age", "balance", "duration", "day", "campaign", "pdays", "previous"]
    df_association = df_association.drop(
        columns=[col for col in columns_to_drop if col in df_association.columns]
    )

    df_encoded = pd.get_dummies(df_association)
    df_encoded = df_encoded.astype(bool)

    logger.info(f"Data siap diproses. Dimensi setelah One-Hot Encoding: {df_encoded.shape}")
    return df_encoded


@task(log_prints=True)
def mine_association_rules(df_encoded: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    logger.info(
        f"Berhasil menemukan {len(frequent_itemsets)} frequent itemsets "
        f"dengan min_support={MIN_SUPPORT}"
    )

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=MIN_LIFT,
        num_itemsets=len(frequent_itemsets),
    )
    rules_sorted = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
    logger.info(f"Berhasil membentuk {len(rules_sorted)} aturan asosiasi.")
    return rules_sorted


def display_clean_rules(rules_df: pd.DataFrame, target: str = "y_yes", top_n: int = 10) -> pd.DataFrame:
    """Utility murni (bukan task) — dipakai ulang juga oleh Phase 4."""
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


@task(log_prints=True)
def summarize_rules(rules_sorted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()
    positive_rules = display_clean_rules(rules_sorted, target="y_yes", top_n=10)
    negative_rules = display_clean_rules(rules_sorted, target="y_no", top_n=10)

    logger.info(f"TOP 10 POSITIVE RULES (Customer Accepts Deposit):\n{positive_rules}")
    logger.info(f"TOP 10 NEGATIVE RULES (Customer Rejects Deposit):\n{negative_rules}")
    return positive_rules, negative_rules


@task(log_prints=True)
def plot_rules_scatter(rules_sorted: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=rules_sorted["support"],
        y=rules_sorted["confidence"],
        hue=rules_sorted["lift"],
        palette="viridis",
        size=rules_sorted["lift"],
        sizes=(20, 200),
    )
    plt.title("Interestingness Measures Network: Support vs Confidence", fontsize=14, fontweight="bold")
    plt.xlabel("Support (Popularitas Pola)", fontsize=12)
    plt.ylabel("Confidence (Tingkat Kepastian Aturan)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Lift Score", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_support_confidence_lift_scatter.png", dpi=150)
    plt.close()


@task(log_prints=True)
def export_outputs(
    rules_sorted: pd.DataFrame, positive_rules: pd.DataFrame, negative_rules: pd.DataFrame
) -> None:
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pickle -> mempertahankan tipe data frozenset di antecedents/consequents,
    # dibutuhkan Phase 4 untuk memanggil ulang display_clean_rules().
    rules_sorted.to_pickle(OUTPUT_DIR / "association_rules_full.pkl")

    # CSV -> setara notebook cell terakhir (rules_sorted.to_csv("rules_sorted.csv")).
    # frozenset di antecedents/consequents otomatis jadi representasi string
    # saat disimpan CSV; untuk pemrosesan ulang programatik pakai file .pkl.
    rules_sorted.to_csv(OUTPUT_DIR / "rules_sorted.csv", index=False)

    positive_rules.to_csv(OUTPUT_DIR / "top_positive_rules.csv", index=False)
    negative_rules.to_csv(OUTPUT_DIR / "top_negative_rules.csv", index=False)

    logger.info(f"Semua output Phase 3 tersimpan di: {OUTPUT_DIR}")


# --------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------
@flow(name="Phase 3 - Association Rule Mining")
def phase3_flow() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ensure_phase1_outputs()
    raw_data = load_raw_data()

    df_encoded = discretize_and_encode(raw_data)
    rules_sorted = mine_association_rules(df_encoded)
    positive_rules, negative_rules = summarize_rules(rules_sorted)
    plot_rules_scatter(rules_sorted)

    export_outputs(rules_sorted, positive_rules, negative_rules)

    return OUTPUT_DIR


if __name__ == "__main__":
    result_path = phase3_flow()
    print(f"\n✅ Phase 3 selesai. Semua output tersedia di: {result_path}")