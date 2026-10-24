"""Pelatihan 5-fold: early stopping (val AUC), ensemble seed, ambang dari validasi, random search."""
import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve)

from .config import CLIN7, CFG, MODEL_KW, N_TRIALS, SEARCH_SEED
from .models import ROIFusionNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sample_grid(n_trials=N_TRIALS, seed=SEARCH_SEED):
    """Random search: 'base' + (n_trials-1) konfigurasi acak. Dipilih dari VAL AUC, bukan test."""
    rng = np.random.default_rng(seed)
    grid = {"base": ({}, {})}
    for i in range(1, n_trials):
        c = dict(lr=float(10 ** rng.uniform(-4, -3)), wd=float(10 ** rng.uniform(-2, -0.5)),
                 noise=float(rng.choice([0.0, 0.02, 0.05, 0.10])))
        m = dict(d_model=int(rng.choice([32, 64, 128])), n_enc_layers=int(rng.choice([1, 2])),
                 dropout=float(rng.choice([0.2, 0.3, 0.4])), head_dropout=float(rng.choice([0.3, 0.5])))
        grid[f"rs{i:02d}"] = (c, m)
    return grid


def prep_clin_safe(D, fit_idx):
    """Anti-kebocoran: imputasi median + scaler di-fit HANYA pada data latih fold ini."""
    X = D["clin"][:, CLIN7].astype(np.float64)
    miss = np.isnan(X).astype(np.float32)
    imp = SimpleImputer(strategy="median").fit(X[fit_idx])
    sc = StandardScaler().fit(imp.transform(X[fit_idx]))
    return sc.transform(imp.transform(X)).astype(np.float32), miss

@torch.no_grad()
def predict_proba(model, idx, M_t, X_t, Ms_t, bs=64):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], device=DEV)
        out.append(torch.softmax(model(M_t[j], X_t[j], Ms_t[j]), 1)[:, 1].cpu())
    return torch.cat(out).numpy()

def train_one_fold(D, k, fusion, cfg=CFG, model_kw=MODEL_KW, seed=42, split_seed=42, verbose=True):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    y = D["y"]
    te = np.where(D["fold"] == k)[0]
    tr_all = np.where(D["fold"] != k)[0]
    tr, va = train_test_split(tr_all, test_size=0.15, stratify=y[tr_all], random_state=split_seed)

    Xs, miss = prep_clin_safe(D, tr)
    M_t = torch.from_numpy(D["M"]).float().to(DEV)
    X_t, Ms_t = torch.from_numpy(Xs).to(DEV), torch.from_numpy(miss).to(DEV)
    y_t = torch.from_numpy(y).long().to(DEV)
    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(len(tr) / (2.0 * cnt), dtype=torch.float32, device=DEV)      # bobot kelas seimbang

    model = ROIFusionNet(fusion=fusion, n_clin=len(CLIN7), **model_kw).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_auc, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); perm = np.random.permutation(tr); losses = []
        for i in range(0, len(perm), cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            if len(idx) < 2:
                continue
            j = torch.as_tensor(idx, device=DEV)
            m = M_t[j] + cfg["noise"] * torch.randn_like(M_t[j])               # derau kecil melawan overfitting
            loss = Fn.cross_entropy(model(m, X_t[j], Ms_t[j]), y_t[j], weight=w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        sched.step()
        val_auc = roc_auc_score(y[va], predict_proba(model, va, M_t, X_t, Ms_t))
        if val_auc > best_auc:                                                  # early stopping
            best_auc, best_ep, bad = val_auc, ep, 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  [{fusion} f{k} s{seed}] ep {ep:3d} | loss {np.mean(losses):.3f} | val AUC {val_auc:.3f} (best {best_auc:.3f}@{best_ep})")
        if ep >= cfg["min_epochs"] and bad >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    return dict(te=te, va=va, best_epoch=best_ep,
                p_te=predict_proba(model, te, M_t, X_t, Ms_t),
                p_va=predict_proba(model, va, M_t, X_t, Ms_t))

def _metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(acc=accuracy_score(y, pred), recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))

def pick_threshold(y_va, p_va, min_recall=None):
    """Ambang dipilih HANYA dari validasi. Default Youden; min_recall=0.8 -> prioritas recall."""
    fpr, tpr, thr = roc_curve(y_va, p_va)
    i = int(np.argmax(tpr - fpr))
    if min_recall is not None:
        ok = np.where(tpr >= min_recall)[0]
        if len(ok):
            i = int(ok[0])
    return float(np.clip(thr[i], 0.0, 1.0))

def run_experiment(D, fusion, folds=range(5), seeds=(42,), min_recall=None, quiet=False, oof_path=None, **kw):
    """Per fold: latih beberapa seed, rata-ratakan probabilitas (ensemble), pilih ambang dari validasi."""
    y, rows = D["y"], []
    oof_p = np.full(len(y), np.nan, dtype=np.float32); oof_thr = np.full(len(y), np.nan, dtype=np.float32)
    for k in folds:
        t0 = time.time()
        outs = [train_one_fold(D, k, fusion, seed=s, verbose=not quiet, **kw) for s in seeds]
        te, va = outs[0]["te"], outs[0]["va"]
        p_te = np.mean([o["p_te"] for o in outs], axis=0)
        p_va = np.mean([o["p_va"] for o in outs], axis=0)
        thr = pick_threshold(y[va], p_va, min_recall)
        oof_p[te], oof_thr[te] = p_te, thr
        row = dict(model=fusion, fold=k, n_seeds=len(seeds), best_epochs=[o["best_epoch"] for o in outs],
                   val_auc=roc_auc_score(y[va], p_va), auc=roc_auc_score(y[te], p_te), thr=round(thr, 3))
        row.update({f"{a}_05": v for a, v in _metrics_at(y[te], p_te, 0.5).items()})
        row.update({f"{a}_thr": v for a, v in _metrics_at(y[te], p_te, thr).items()})
        rows.append(row)
        if not quiet:
            print(f"== {fusion} | fold {k} | val AUC {row['val_auc']:.3f} | TEST AUC {row['auc']:.3f} | "
                  f"ambang {thr:.2f}: acc {row['acc_thr']:.3f} recall {row['recall_thr']:.3f} | {time.time() - t0:.0f} dtk")
    if oof_path:                                   # prediksi per subjek (untuk uji McNemar nanti)
        np.savez(oof_path, p=oof_p, thr=oof_thr, y=y, fold=D["fold"], sub_id=D.get("sub_id", np.arange(len(y))))
    return pd.DataFrame(rows)

def tune(D, fusion, grid, folds=range(5), seeds=(42,)):
    """Pilih konfigurasi dari VAL AUC. (test_auc_info hanya informasi, JANGAN dipakai memilih.)"""
    rows = []
    for name, (c, m) in grid.items():
        t0 = time.time()
        df = run_experiment(D, fusion, folds=folds, seeds=seeds, quiet=True,
                            cfg={**CFG, **c}, model_kw={**MODEL_KW, **m})
        rows.append(dict(config=name, val_auc=round(df.val_auc.mean(), 3), test_auc_info=round(df.auc.mean(), 3),
                         params=json.dumps({**c, **m})))
        print(f"   [tuning {fusion}] {name}: val AUC {rows[-1]['val_auc']:.3f} | {time.time() - t0:.0f} dtk")
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)

def summarize_pct(df):
    cols = ["val_auc", "auc", "acc_05", "recall_05", "acc_thr", "recall_thr", "precision_thr", "f1_thr"]
    agg = df.groupby("model", sort=False)[cols].agg(["mean", "std"]) * 100
    return pd.DataFrame({c: agg[c]["mean"].round(1).astype(str) + "% ± " + agg[c]["std"].round(1).astype(str) + "%"
                         for c in cols})
