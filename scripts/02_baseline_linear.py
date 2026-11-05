"""Langkah 2: baseline linear 5-fold (situs, klinis, konektivitas, early/late fusion) + prediksi out-of-fold."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
from src.config import prepare_environment, TABEL_DIR, METRICS_DIR
from src.data import prepare_dataset
from src.baselines import run_all, oof_linear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abide1_pcp", choices=["abide1_pcp", "abide1_dcan", "abide12_dcan"])
    ap.add_argument("--tag", default="", help="akhiran nama berkas keluaran, mis. _abide12")
    ap.add_argument("--harmonize", action="store_true", help="ComBat (dipelajari dari data latih tiap fold) pada fitur konektivitas")
    a = ap.parse_args()
    prepare_environment()
    D = prepare_dataset(name=a.dataset)
    print("Batas tebakan (selalu 'kontrol'):", round(1 - D["y"].mean(), 3))
    per_fold, table = run_all(D)
    pd.set_option("display.width", 220); print(table.to_string())
    per_fold.to_csv(TABEL_DIR + f"baseline_linear_per_fold{a.tag}.csv", index=False)
    table.to_csv(TABEL_DIR + f"baseline_linear_ringkas{a.tag}.csv")
    oof = oof_linear(D, harmonize=a.harmonize)                                   # untuk uji signifikansi (langkah 4)
    np.savez(METRICS_DIR + f"oof_linear{a.tag}.npz", **{name: p for name, (p, thr) in oof.items()})


if __name__ == "__main__":
    main()
