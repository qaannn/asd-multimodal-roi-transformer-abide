"""Kompilasi hasil 5 skenario ke CSV ringkas siap-lampir (BAB 4): akurasi diutamakan, AUC di sebelahnya."""
import glob, os
import numpy as np
import pandas as pd

from .scenarios import SCENARIOS
from .stats import mcnemar_exact, holm


def _ms(x, scale=100.0):
    x = np.asarray(x, float) * scale
    return f"{np.nanmean(x):.1f} ± {np.nanstd(x, ddof=1):.1f}"


def summarize_frame(df, with_extra=True):
    r = {"N subjek": int(df["n_total"].iloc[0]), "Akurasi (%)": _ms(df["acc"]), "AUC (%)": _ms(df["auc"]),
         "Sensitivitas (%)": _ms(df["recall"]), "Presisi (%)": _ms(df["precision"]), "F1 (%)": _ms(df["f1"])}
    if with_extra and "acc_thr" in df.columns and df["acc_thr"].notna().any():
        r["Akurasi ambang-validasi (%)"] = _ms(df["acc_thr"])
    return r


def summarize_scenarios(metrics_dir, tabel_dir):
    rows = []
    for sid, spec in SCENARIOS.items():
        p = metrics_dir + f"skenario_{sid}_per_fold.csv"
        if not os.path.exists(p):
            rows.append({"Skenario": sid, "Deskripsi": spec["nama"], "Dataset": spec["dataset"], "Model": spec["model"],
                         "N subjek": "", "Akurasi (%)": "(belum dijalankan)"}); continue
        df = pd.read_csv(p)
        rows.append({"Skenario": sid, "Deskripsi": spec["nama"], "Dataset": df["dataset"].iloc[0], "Model": df["model"].iloc[0],
                     **summarize_frame(df)})
    out = pd.DataFrame(rows)
    out.to_csv(tabel_dir + "skenario_ringkasan.csv", index=False)
    cols = [c for c in ["Skenario", "Dataset", "Model", "N subjek", "Akurasi (%)", "AUC (%)", "Sensitivitas (%)", "Presisi (%)", "F1 (%)"] if c in out.columns]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in out.iterrows()]
    open(tabel_dir + "skenario_ringkasan_bab4.md", "w").write("\n".join(md) + "\n")
    return out


def summarize_supplement(metrics_dir, tabel_dir):
    rows = []
    for p in sorted(glob.glob(metrics_dir + "tambahan_*_per_fold.csv")):
        df = pd.read_csv(p)
        rows.append({"Kode": df["skenario"].iloc[0], "Dataset": df["dataset"].iloc[0], "Model": df["model"].iloc[0], **summarize_frame(df, False)})
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(tabel_dir + "analisis_tambahan_ringkasan.csv", index=False)
    return out


def _boot_dacc(c1, c2, rng, n=1000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(c1), len(c1)); d.append(c1[i].mean() - c2[i].mean())
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


PAIRS = [("S1", "S2"), ("S1", "S3"), ("S2", "S3"), ("S4", "S5")]     # hanya pasangan pada subjek yang sama


def paired_tests(metrics_dir, tabel_dir, seed=42):
    rng = np.random.default_rng(seed); rows = []
    for a, b in PAIRS:
        pa, pb = metrics_dir + f"oof_{a}.npz", metrics_dir + f"oof_{b}.npz"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        za, zb = np.load(pa), np.load(pb)
        if len(za["y"]) != len(zb["y"]) or not (za["sub_id"] == zb["sub_id"]).all():
            continue
        ca, cb = za["pred"] == za["y"], zb["pred"] == zb["y"]
        n10, n01, p = mcnemar_exact(ca, cb); dm, (lo, hi) = _boot_dacc(ca, cb, rng)
        rows.append({"A": a, "B": b, "A benar/B salah": n10, "A salah/B benar": n01, "p_McNemar": p,
                     "dAkurasi (A-B, poin %)": round(dm * 100, 2), "CI95": f"[{lo * 100:.1f}, {hi * 100:.1f}]"})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_Holm"] = holm(out["p_McNemar"].to_numpy()); out["Signifikan (Holm<0.05)"] = np.where(out["p_Holm"] < 0.05, "Ya", "Tidak")
        out[["p_McNemar", "p_Holm"]] = out[["p_McNemar", "p_Holm"]].round(4)
        out.to_csv(tabel_dir + "skenario_uji_akurasi.csv", index=False)
    return out
