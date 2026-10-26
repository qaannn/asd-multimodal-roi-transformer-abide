"""Langkah 3: ROI-Transformer (none | concat | cross) - random search per varian, lalu 5-fold x ensemble seed. Butuh GPU."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
from src.config import (prepare_environment, TABEL_DIR, METRICS_DIR, CFG, MODEL_KW, DO_TUNE, N_TRIALS, SEARCH_SEED,
                        SEEDS_FINAL, MIN_RECALL)
from src.data import prepare_dataset
from src.training import DEV, sample_grid, tune, run_experiment, summarize_pct


def main():
    prepare_environment()
    print("Perangkat:", DEV)
    if DEV == "cpu":
        raise RuntimeError("Runtime masih CPU (sangat lambat). Aktifkan GPU (Colab: Runtime > Change runtime type > T4).")
    D = prepare_dataset()
    grid = sample_grid(N_TRIALS, SEARCH_SEED) if DO_TUNE else {"base": ({}, {})}
    all_res, chosen = [], {}
    for fusion in ["none", "concat", "cross"]:
        print("\n" + "=" * 60 + f"\nVARIAN: {fusion}\n" + "=" * 60)
        tab = tune(D, fusion, grid)                        # pilih konfigurasi dari VAL AUC (bukan test)
        tab.to_csv(TABEL_DIR + f"tuning_{fusion}.csv", index=False)
        best = tab.loc[0, "config"]
        print(tab[["config", "val_auc", "test_auc_info"]].head(5).to_string(), f"\n>> terpilih: {best}")
        c, m = grid[best]
        chosen[fusion] = dict(name=best, cfg=c, model=m)
        df_f = run_experiment(D, fusion, seeds=SEEDS_FINAL, min_recall=MIN_RECALL,
                              cfg={**CFG, **c}, model_kw={**MODEL_KW, **m},
                              oof_path=METRICS_DIR + f"oof_{fusion}.npz")
        df_f.to_csv(TABEL_DIR + f"deep_{fusion}_per_fold.csv", index=False)   # aman bila sesi putus
        all_res.append(df_f)
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(TABEL_DIR + "deep_ablation_per_fold.csv", index=False)
    table = summarize_pct(res)
    print("\n*_05 = ambang 0.5 | *_thr = ambang dari validasi | auc = AUC tes\n" + table.to_string())
    table.to_csv(TABEL_DIR + "deep_ablation_ringkas.csv")
    json.dump(chosen, open(METRICS_DIR + "konfigurasi_terpilih.json", "w"), indent=2)


if __name__ == "__main__":
    main()
