"""Statistik perbandingan model: AUC + CI bootstrap, McNemar exact + Holm, selisih AUC berpasangan."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

N_BOOT = 1000


def load_deep(y, d):
    out = {}
    for f, name in [("none", "Deep: ROI-Transformer (konek. saja)"), ("concat", "Deep: + klinis (concat)"),
                    ("cross", "Deep: + klinis (cross-attention)")]:
        z = np.load(f"{d}oof_{f}.npz")
        assert (z["y"] == y).all(), "Urutan subjek OOF tidak sama dengan data_v2!"
        assert not np.isnan(z["p"]).any(), "OOF berisi NaN (fold belum lengkap)"
        out[name] = (z["p"].astype(float), z["thr"].astype(float))
    return out


def boot_ci_auc(y, p, rng, n=N_BOOT):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])

def summary(models, y, rng):
    rows = []
    for name, (p, thr) in models.items():
        pred = (p >= thr).astype(int)
        tp, fn, fp = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum()
        lo, hi = boot_ci_auc(y, p, rng)
        rows.append(dict(Model=name, AUC=round(roc_auc_score(y, p), 3), CI95=f"[{lo:.3f}, {hi:.3f}]",
                         Akurasi=round((pred == y).mean(), 3), Recall=round(tp / max(tp + fn, 1), 3),
                         Presisi=round(tp / max(tp + fp, 1), 3)))
    return pd.DataFrame(rows)

def mcnemar_exact(c1, c2):
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    n = b + c
    return b, c, (1.0 if n == 0 else stats.binomtest(min(b, c), n, 0.5).pvalue)

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj

def paired_boot_dauc(y, p1, p2, rng, n=N_BOOT):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])

PAIRS = [  # (A, B)  -> A dibanding B
    ("Late fusion (LogReg + SVM)", "Konektivitas saja (LogReg)"),
    ("Late fusion (LogReg + SVM)", "Klinis saja (SVM)"),
    ("Late fusion (LogReg + SVM)", "Deep: + klinis (concat)"),
    ("Late fusion (LogReg + SVM)", "Deep: + klinis (cross-attention)"),
    ("Konektivitas saja (LogReg)", "Deep: ROI-Transformer (konek. saja)"),
    ("Deep: ROI-Transformer (konek. saja)", "Deep: + klinis (concat)"),
    ("Deep: + klinis (concat)", "Deep: + klinis (cross-attention)"),
]

def pairwise(models, y, rng):
    rows = []
    for a, b in PAIRS:
        (pa, ta), (pb, tb) = models[a], models[b]
        b01, c01, p = mcnemar_exact((pa >= ta).astype(int) == y, (pb >= tb).astype(int) == y)
        dm, (lo, hi) = paired_boot_dauc(y, pa, pb, rng)
        rows.append(dict(A=a, B=b, **{"A benar/B salah": b01, "A salah/B benar": c01}, p_McNemar=p,
                         dAUC=round(dm, 3), CI95_dAUC=f"[{lo:.3f}, {hi:.3f}]"))
    df = pd.DataFrame(rows)
    df["p_Holm"] = holm(df["p_McNemar"].to_numpy())
    df["Signifikan (Holm<0.05)"] = np.where(df["p_Holm"] < 0.05, "Ya", "Tidak")
    df["p_McNemar"], df["p_Holm"] = df["p_McNemar"].round(4), df["p_Holm"].round(4)
    return df
