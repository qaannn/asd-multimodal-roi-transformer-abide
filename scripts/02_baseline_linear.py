"""Langkah 2: baseline linear 5-fold (situs, klinis, konektivitas, early/late fusion) + prediksi out-of-fold."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.config import prepare_environment, TABEL_DIR, METRICS_DIR
from src.data import prepare_dataset
from src.baselines import run_all, oof_linear


def main():
    prepare_environment()
    D = prepare_dataset()
    print("Batas tebakan (selalu 'kontrol'):", round(1 - D["y"].mean(), 3))
    per_fold, table = run_all(D)
    pd.set_option("display.width", 220); print(table.to_string())
    per_fold.to_csv(TABEL_DIR + "baseline_linear_per_fold.csv", index=False)
    table.to_csv(TABEL_DIR + "baseline_linear_ringkas.csv")
    oof = oof_linear(D)                                   # untuk uji signifikansi (langkah 4)
    np.savez(METRICS_DIR + "oof_linear.npz", **{name: p for name, (p, thr) in oof.items()})


if __name__ == "__main__":
    main()
