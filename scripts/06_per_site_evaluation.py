"""Langkah 5: evaluasi per situs dari prediksi out-of-fold (tanpa pelatihan baru)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.config import prepare_environment, TABEL_DIR, METRICS_DIR, FIGURE_DIR
from src.data import prepare_dataset
from src.site_eval import load_models, per_site_tables, site_summary, site_wins, plot_sites


def main():
    prepare_environment()
    D = prepare_dataset(); y, site = D["y"], D["site"].astype(str)
    models = load_models(y, METRICS_DIR)
    auc_t, acc_t, rec_t = per_site_tables(models, y, site)
    summ, wins = site_summary(models, y, auc_t), site_wins(models, auc_t)
    pd.set_option("display.width", 250)
    print("=== AUC per situs ===\n", auc_t.round(3).to_string(index=False))
    print("\n=== Ringkasan ===\n", summ.to_string(index=False))
    print("\n=== Late fusion vs lain ===\n", wins.to_string(index=False))
    for name, t in [("auc", auc_t), ("akurasi", acc_t), ("recall", rec_t)]:
        t.to_csv(TABEL_DIR + f"situs_{name}.csv", index=False)
    summ.to_csv(TABEL_DIR + "situs_ringkasan.csv", index=False); wins.to_csv(TABEL_DIR + "situs_late_fusion_vs_lain.csv", index=False)
    plot_sites(auc_t, FIGURE_DIR + "situs_auc.png")


if __name__ == "__main__":
    main()
