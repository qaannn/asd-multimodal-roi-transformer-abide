"""Evaluasi per situs dari prediksi out-of-fold (AUC dalam-situs, ringkasan, perbandingan lintas situs)."""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

MIN_N = 30        # situs dihitung AUC-nya hanya bila N >= 30 ...
MIN_MINOR = 8     # ... dan kelas minoritas >= 8 subjek (AUC pada sampel kecil sangat tidak stabil)
KEY_MODELS = ["Late fusion (LogReg + SVM)", "Konektivitas saja (LogReg)", "Klinis saja (SVM)",
              "Deep: + klinis (concat)", "Deep: ROI-Transformer (konek. saja)"]
DEEP_NAMES = {"none": "Deep: ROI-Transformer (konek. saja)", "concat": "Deep: + klinis (concat)",
              "cross": "Deep: + klinis (cross-attention)"}


def load_models(y, metrics_dir):
    """Muat prediksi out-of-fold: baseline linear (oof_linear.npz) + 3 model dalam (oof_*.npz)."""
    path = os.path.join(metrics_dir, "oof_linear.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("oof_linear.npz tidak ada. Jalankan uji signifikansi (05 / scripts/04) atau scripts/02 dulu.")
    z = np.load(path)
    models = {k.replace("_", " "): (z[k].astype(float), 0.5) for k in z.files}   # nama bisa bergaris-bawah
    for f, name in DEEP_NAMES.items():
        d = np.load(os.path.join(metrics_dir, f"oof_{f}.npz"))
        assert (d["y"] == y).all(), f"urutan subjek oof_{f}.npz tidak sama dengan data_v2"
        assert not np.isnan(d["p"]).any(), f"oof_{f}.npz berisi NaN"
        models[name] = (d["p"].astype(float), d["thr"].astype(float))
    return models


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def per_site_tables(models, y, site):
    """Tiga tabel (AUC, akurasi, recall) per situs. AUC hanya dihitung untuk situs yang memenuhi MIN_N/MIN_MINOR."""
    auc_rows, acc_rows, rec_rows = [], [], []
    for s in sorted(np.unique(site)):
        m = site == s
        n, n1 = int(m.sum()), int(y[m].sum()); n0 = n - n1
        ok = n >= MIN_N and min(n0, n1) >= MIN_MINOR
        base = dict(Situs=s, N=n, ASD=n1, Kontrol=n0, Dihitung="Ya" if ok else "Tidak")
        ra, rc, rr = dict(base), dict(base), dict(base)
        for name, (p, thr) in models.items():
            t = thr[m] if np.ndim(thr) else thr
            pred = (p[m] >= t).astype(int)
            ra[name] = _auc(y[m], p[m]) if ok else np.nan
            rc[name] = (pred == y[m]).mean()
            rr[name] = ((pred == 1) & (y[m] == 1)).sum() / max(n1, 1)
        auc_rows.append(ra); acc_rows.append(rc); rec_rows.append(rr)
    return tuple(pd.DataFrame(r).sort_values("N", ascending=False).reset_index(drop=True)
                 for r in (auc_rows, acc_rows, rec_rows))


def site_summary(models, y, auc_tab):
    """AUC pooled vs AUC dalam-situs (rata-rata berbobot N dan tak berbobot) pada situs yang dihitung."""
    ok = auc_tab[auc_tab["Dihitung"] == "Ya"]
    rows = []
    for name, (p, thr) in models.items():
        w = ok["N"].to_numpy(float); a = ok[name].to_numpy(float)
        rows.append({"Model": name, "AUC pooled": round(roc_auc_score(y, p), 3),
                     "AUC dalam-situs (berbobot N)": round(float((w * a).sum() / w.sum()), 3),
                     "AUC dalam-situs (rata-rata ± SD)": f"{a.mean():.3f} ± {a.std(ddof=1):.3f}",
                     "Situs dihitung": len(ok)})
    return pd.DataFrame(rows)


def site_wins(models, auc_tab, ref="Late fusion (LogReg + SVM)"):
    """Berapa situs di mana model referensi lebih tinggi AUC-nya, plus Wilcoxon (daya rendah; hanya indikatif)."""
    ok = auc_tab[auc_tab["Dihitung"] == "Ya"]
    rows = []
    for name in models:
        if name == ref:
            continue
        d = ok[ref].to_numpy(float) - ok[name].to_numpy(float)
        try:
            pw = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan
        except ValueError:
            pw = np.nan
        rows.append({"Referensi": ref, "Pembanding": name, "Situs ref lebih tinggi": f"{int((d > 0).sum())} dari {len(d)}",
                     "Rata-rata selisih AUC": round(float(d.mean()), 3), "p Wilcoxon (indikatif)": round(float(pw), 4) if pw == pw else np.nan})
    return pd.DataFrame(rows)


def plot_sites(auc_tab, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib tidak tersedia; gambar dilewati"); return
    ok = auc_tab[auc_tab["Dihitung"] == "Ya"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, mk in zip([k for k in KEY_MODELS if k in ok.columns], ["o", "s", "^", "D", "v"]):
        ax.plot(range(len(ok)), ok[name], marker=mk, linestyle="-", linewidth=1, label=name)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(ok))); ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(ok["Situs"], ok["N"])], fontsize=7)
    ax.set_ylabel("AUC dalam situs"); ax.set_ylim(0.3, 1.0); ax.legend(fontsize=7, ncol=2)
    ax.set_title("AUC per situs (prediksi out-of-fold; hanya situs dengan sampel memadai)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("gambar:", path)
