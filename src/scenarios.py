"""Lima skenario resmi (maks. 5 menurut dosen) + analisis tambahan. Metrik utama: AKURASI (ambang 0,5 untuk semua
skenario, aturan sama), AUC sekunder. Setiap skenario menyimpan hasil per-fold dan prediksi out-of-fold."""
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from .config import prepare_environment, METRICS_DIR, CKPT_DIR, CFG, MODEL_KW, SEARCH_SEED, SEEDS_FINAL
from .data import prepare_dataset
from . import baselines as B

SCENARIOS = {
    "S1": dict(nama="Baseline ML monomodal", dataset="abide1_dcan", model="Konektivitas fMRI + LogReg", family="linear_conn"),
    "S2": dict(nama="Baseline DL lama", dataset="abide1_dcan", model="ResNet-50 (beku) + SVM, konektivitas", family="resnet"),
    "S3": dict(nama="Model baru, ABIDE I", dataset="abide1_dcan", model="ROI-Transformer + klinis (cross-attention)", family="deep_cross"),
    "S4": dict(nama="Baseline, ABIDE I+II", dataset="abide12_dcan", model="Konektivitas fMRI + LogReg", family="linear_conn"),
    "S5": dict(nama="Model baru, ABIDE I+II (final)", dataset="abide12_dcan", model="ROI-Transformer + klinis (cross-attention)", family="deep_cross"),
}
SUPPLEMENT = {   # analisis tambahan (BUKAN skenario utama)
    "T1": ("Klinis saja (SVM)", "linear_clin"),
    "T2": ("Late fusion linear (LogReg konektivitas + SVM klinis)", "linear_late"),
    "T3": ("ROI-Transformer konektivitas saja", "deep_none"),
    "T4": ("ROI-Transformer + klinis (concat)", "deep_concat"),
    "T5": ("Konektivitas + LogReg + ComBat", "linear_conn_combat"),
    "T6": ("ROI-Transformer cross-attention + ComBat", "deep_cross_combat"),
}


def resolve_dataset(name, abide1_pipeline="dcan"):
    return "abide1_pcp" if (name == "abide1_dcan" and abide1_pipeline == "pcp") else name


def fold_metrics(D, pred, score):
    rows = []
    for k in sorted(np.unique(D["fold"])):
        m = D["fold"] == k; y = D["y"][m]
        rows.append(dict(fold=int(k), n_test=int(m.sum()), acc=accuracy_score(y, pred[m]),
                         auc=roc_auc_score(y, score[m]) if len(np.unique(y)) == 2 else np.nan,
                         recall=recall_score(y, pred[m], zero_division=0), precision=precision_score(y, pred[m], zero_division=0),
                         f1=f1_score(y, pred[m], zero_division=0)))
    return pd.DataFrame(rows)


def _save(sid, df, D, pred, score, extra=None):
    df = df.copy(); df.insert(0, "skenario", sid); df["n_total"] = len(D["y"])
    if extra:
        for k, v in extra.items():
            df[k] = v
    df.to_csv(METRICS_DIR + f"skenario_{sid}_per_fold.csv", index=False)
    np.savez(METRICS_DIR + f"oof_{sid}.npz", pred=pred, score=score, y=D["y"], fold=D["fold"],
             sub_id=D.get("sub_id", np.arange(len(D["y"]))))
    return df


def deep_config(D, fusion, sid, trials, retune=False, reuse_from=None):
    """Konfigurasi model dalam dipilih lewat VAL AUC (random search). S5 memakai ulang konfigurasi S3 (dipilih pada data
    ABIDE I saja) kecuali retune=True, supaya hemat waktu dan tanpa memilih konfigurasi dari hasil uji."""
    from .training import sample_grid, tune
    path = METRICS_DIR + f"config_{sid}.json"
    src = METRICS_DIR + f"config_{reuse_from}.json" if reuse_from else None
    if src and os.path.exists(src) and not retune:
        j = json.load(open(src)); json.dump({**j, "diambil_dari": reuse_from}, open(path, "w"), indent=2)
        return j["cfg"], j["model"], f"{j['name']} (dari {reuse_from})"
    if trials and trials > 1:
        grid = sample_grid(trials, SEARCH_SEED)
        tab = tune(D, fusion, grid); name = tab.loc[0, "config"]; c, m = grid[name]
    else:
        name, c, m = "base", {}, {}
    json.dump(dict(name=name, cfg=c, model=m), open(path, "w"), indent=2)
    return c, m, name


def _run_deep(D, sid, fusion, trials, seeds, retune, reuse_from, harmonize, ckpt):
    from .training import run_experiment
    c, m, cname = deep_config(D, fusion, sid, trials, retune, reuse_from)
    print(f"[{sid}] konfigurasi {fusion}: {cname}")
    oof_tmp = METRICS_DIR + f"_oof_tmp_{sid}.npz"
    df = run_experiment(D, fusion, seeds=seeds, cfg={**CFG, **c}, model_kw={**MODEL_KW, **m}, criterion="acc",
                        harmonize=harmonize, oof_path=oof_tmp, ckpt_dir=CKPT_DIR if ckpt else None, ckpt_tag=sid)
    z = np.load(oof_tmp); score = z["p"].astype(float); pred = (score >= 0.5).astype(int)
    out = df.rename(columns={"acc_05": "acc", "recall_05": "recall", "precision_05": "precision", "f1_05": "f1"})
    out["n_test"] = [int((D["fold"] == k).sum()) for k in out["fold"]]
    keep = ["fold", "n_test", "acc", "auc", "recall", "precision", "f1", "acc_thr", "thr", "val_auc"]
    os.remove(oof_tmp)
    return out[keep], pred, score, cname


def run_family(D, family, sid, dataset_tag, trials=12, seeds=SEEDS_FINAL, retune=False, reuse_from=None, ckpt=False):
    """Mengembalikan (df per-fold, pred, score, extra)."""
    if family in ("linear_conn", "linear_conn_combat"):
        df, pred, score = B.run_cv_oof(D, "konek", B.conn_raw_block(D, harmonize=(family == "linear_conn_combat")), B.LR_STRONG)
        return fold_metrics(D, pred, score), pred, score, {}
    if family == "linear_clin":
        df, pred, score = B.run_cv_oof(D, "klinis", B.clin_block(D), B.SVM)
        return fold_metrics(D, pred, score), pred, score, {}
    if family == "linear_late":
        p = B.oof_linear(D)["Late fusion (LogReg + SVM)"][0]
        pred = (p >= 0.5).astype(int)
        return fold_metrics(D, pred, p), pred, p, {}
    if family == "resnet":
        from .resnet_baseline import get_features, resnet_block
        df, pred, score = B.run_cv_oof(D, "resnet", resnet_block(get_features(D, dataset_tag)), B.SVM)
        return fold_metrics(D, pred, score), pred, score, {}
    fusion = {"deep_cross": "cross", "deep_none": "none", "deep_concat": "concat", "deep_cross_combat": "cross"}[family]
    df, pred, score, cname = _run_deep(D, sid, fusion, trials, seeds, retune, reuse_from,
                                       harmonize=family.endswith("combat"), ckpt=ckpt)
    return df, pred, score, {"konfigurasi": cname}


def run_scenario(sid, abide1_pipeline="dcan", trials=12, seeds=SEEDS_FINAL, retune=False):
    prepare_environment()
    spec = SCENARIOS[sid]; ds = resolve_dataset(spec["dataset"], abide1_pipeline)
    D = prepare_dataset(name=ds)
    print(f"\n===== {sid}: {spec['nama']} | {spec['model']} | {ds} | N={len(D['y'])} =====")
    reuse = "S3" if sid == "S5" else None
    ckpt = sid in ("S3", "S5")
    df, pred, score, extra = run_family(D, spec["family"], sid, ds, trials, seeds, retune, reuse, ckpt)
    extra = {**extra, "dataset": ds, "model": spec["model"]}
    out = _save(sid, df, D, pred, score, extra)
    print(f"[{sid}] Akurasi {out.acc.mean() * 100:.1f} ± {out.acc.std() * 100:.1f}% | AUC {out.auc.mean() * 100:.1f} ± {out.auc.std() * 100:.1f}%")
    return out


def run_supplement(tid, dataset, abide1_pipeline="dcan", trials=12, seeds=SEEDS_FINAL):
    prepare_environment()
    nama, family = SUPPLEMENT[tid]; ds = resolve_dataset(dataset, abide1_pipeline)
    D = prepare_dataset(name=ds); key = f"{tid}_{ds}"
    print(f"\n===== Tambahan {tid}: {nama} | {ds} | N={len(D['y'])} =====")
    df, pred, score, extra = run_family(D, family, key, ds, trials, seeds, False, "S3" if family.startswith("deep") else None)
    df = df.copy(); df.insert(0, "skenario", tid); df["dataset"] = ds; df["model"] = nama; df["n_total"] = len(D["y"])
    df.to_csv(METRICS_DIR + f"tambahan_{key}_per_fold.csv", index=False)
    print(f"[{tid}] Akurasi {df.acc.mean() * 100:.1f} ± {df.acc.std() * 100:.1f}% | AUC {df.auc.mean() * 100:.1f}%")
    return df
