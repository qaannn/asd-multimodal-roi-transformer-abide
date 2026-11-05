"""Langkah 4: uji signifikansi lintas model (McNemar exact + Holm, selisih AUC bootstrap berpasangan)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd
from src.config import prepare_environment, TABEL_DIR, METRICS_DIR, SEED
from src.data import prepare_dataset
from src.baselines import oof_linear
from src.stats import load_deep, summary, pairwise


def main():
    prepare_environment()
    rng = np.random.default_rng(SEED)
    D = prepare_dataset(); y = D["y"]
    path = METRICS_DIR + "oof_linear.npz"
    if os.path.exists(path):
        z = np.load(path); linear = {k: (z[k], 0.5) for k in z.files}
    else:
        linear = oof_linear(D)
    models = {**linear, **load_deep(y, METRICS_DIR)}
    s, t = summary(models, y, rng), pairwise(models, y, rng)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print("=== RINGKAS (prediksi out-of-fold digabung, N=%d) ===\n%s" % (len(y), s.to_string(index=False)))
    print("\n=== UJI BERPASANGAN ===\n%s" % t.to_string(index=False))
    s.to_csv(TABEL_DIR + "uji_ringkas_oof.csv", index=False); t.to_csv(TABEL_DIR + "uji_berpasangan.csv", index=False)


if __name__ == "__main__":
    main()
