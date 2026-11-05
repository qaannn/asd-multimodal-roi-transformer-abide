"""Langkah 3: ROI-Transformer (none | concat | cross) - random search per varian, lalu 5-fold x ensemble seed. Butuh GPU."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import pandas as pd
from src.config import (prepare_environment, TABEL_DIR, METRICS_DIR, CFG, MODEL_KW, DO_TUNE, N_TRIALS, SEARCH_SEED,
                        SEEDS_FINAL, MIN_RECALL)
from src.data import prepare_dataset
from src.config import CKPT_DIR
from src.training import DEV, sample_grid, tune, run_experiment, summarize_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abide1_pcp", choices=["abide1_pcp", "abide1_dcan", "abide12_dcan"])
    ap.add_argument("--tag", default=""); ap.add_argument("--harmonize", action="store_true")
    ap.add_argument("--ckpt", action="store_true", help="simpan bobot terbaik (.pth) per fold/seed + *_best.pth")
    ap.add_argument("--criterion", default="youden", choices=["youden", "acc"], help="kriteria ambang dari validasi")
    a = ap.parse_args()
    prepare_environment()
    print("Perangkat:", DEV)
    if DEV == "cpu":
        raise RuntimeError("Runtime masih CPU (sangat lambat). Aktifkan GPU (Colab: Runtime > Change runtime type > T4).")
    D = prepare_dataset(name=a.dataset)
    grid = sample_grid(N_TRIALS, SEARCH_SEED) if DO_TUNE else {"base": ({}, {})}
    all_res, chosen = [], {}
    for fusion in ["none", "concat", "cross"]:
        print("\n" + "=" * 60 + f"\nVARIAN: {fusion}\n" + "=" * 60)
        tab = tune(D, fusion, grid)                        # pilih konfigurasi dari VAL AUC (bukan test)
        tab.to_csv(TABEL_DIR + f"tuning_{fusion}{a.tag}.csv", index=False)
        best = tab.loc[0, "config"]
        print(tab[["config", "val_auc", "test_auc_info"]].head(5).to_string(), f"\n>> terpilih: {best}")
        c, m = grid[best]
        chosen[fusion] = dict(name=best, cfg=c, model=m)
        df_f = run_experiment(D, fusion, seeds=SEEDS_FINAL, min_recall=MIN_RECALL,
                              cfg={**CFG, **c}, model_kw={**MODEL_KW, **m},
                              oof_path=METRICS_DIR + f"oof_{fusion}{a.tag}.npz",
                              harmonize=a.harmonize, criterion=a.criterion,
                              ckpt_dir=CKPT_DIR if a.ckpt else None, ckpt_tag=f"{fusion}{a.tag}")
        df_f.to_csv(TABEL_DIR + f"deep_{fusion}_per_fold{a.tag}.csv", index=False)   # aman bila sesi putus
        all_res.append(df_f)
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(TABEL_DIR + f"deep_ablation_per_fold{a.tag}.csv", index=False)
    table = summarize_pct(res)
    print("\n*_05 = ambang 0.5 | *_thr = ambang dari validasi | auc = AUC tes\n" + table.to_string())
    table.to_csv(TABEL_DIR + f"deep_ablation_ringkas{a.tag}.csv")
    json.dump(chosen, open(METRICS_DIR + f"konfigurasi_terpilih{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
