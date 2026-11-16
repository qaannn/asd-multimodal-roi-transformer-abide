"""Tabel untuk naskah: alur kohort, demografi (+p-value), data hilang, jumlah per situs."""
import numpy as np
import pandas as pd
from scipy import stats

from .config import FEATURES


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def ms(x):
    x = x[~np.isnan(x)]
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f} (n={len(x)})"

def tabel_alur_kohort(raw, sub_final, y_final):
    r = raw[raw["DX_GROUP"].isin([1, 2])]
    s1 = r[r["FILE_ID"].astype(str).str.strip().str.lower() != "no_filename"]
    asd = lambda d: int((d["DX_GROUP"] == 1).sum())
    rows = [
        ("0. Fenotipe ABIDE I (ASD + kontrol)", len(r), "-", asd(r), len(r) - asd(r)),
        ("1. FILE_ID ≠ no_filename (citra tersedia)", len(s1), len(r) - len(s1), asd(s1), len(s1) - asd(s1)),
        ("2. Tersedia di ABIDE PCP (C-PAC, quality-checked) & cocok SUB_ID", len(sub_final), len(s1) - len(sub_final),
         int(y_final.sum()), int((1 - y_final).sum())),
    ]
    return pd.DataFrame(rows, columns=["Tahap", "Subjek tersisa", "Dikeluarkan", "ASD", "Kontrol"])

def tabel_demografi(clin, y):
    a, c = y == 1, y == 0
    rows = [("Jumlah subjek", f"{a.sum()}", f"{c.sum()}", "-")]
    for name, j in [("Umur (tahun)", 0), ("Full-scale IQ", 3), ("Verbal IQ", 4), ("Performance IQ", 5)]:
        xa, xc = clin[a, j], clin[c, j]
        p = stats.ttest_ind(xa[~np.isnan(xa)], xc[~np.isnan(xc)], equal_var=False).pvalue
        rows.append((name, ms(xa), ms(xc), fmt_p(p)))
    for name, j, lab in [("Laki-laki / perempuan", 1, ("laki-laki", "perempuan")),
                         ("Tangan dominan kanan / non-kanan", 2, ("kanan", "non-kanan")),
                         ("Mata terbuka / tertutup", 7, ("terbuka", "tertutup"))]:
        ka, kc = clin[a, j][~np.isnan(clin[a, j])], clin[c, j][~np.isnan(clin[c, j])]
        tab = np.array([[(ka == 0).sum(), (ka == 1).sum()], [(kc == 0).sum(), (kc == 1).sum()]])
        p = stats.chi2_contingency(tab)[1]
        rows.append((name, f"{tab[0,0]} / {tab[0,1]}", f"{tab[1,0]} / {tab[1,1]}", fmt_p(p)))
    return pd.DataFrame(rows, columns=["Variabel", "ASD", "Kontrol", "p-value"])

def tabel_hilang(clin, y):
    rows = []
    for j, f in enumerate(FEATURES):
        m = np.isnan(clin[:, j])
        rows.append((f, f"{m.mean() * 100:.1f}%", f"{m[y == 1].mean() * 100:.1f}%", f"{m[y == 0].mean() * 100:.1f}%"))
    return pd.DataFrame(rows, columns=["Fitur", "Hilang (semua)", "Hilang (ASD)", "Hilang (kontrol)"])

def tabel_situs(site, y):
    d = pd.crosstab(pd.Series(site, name="Situs"), pd.Series(y, name="y")).rename(columns={0: "Kontrol", 1: "ASD"})
    d["Total"] = d.sum(axis=1); d["% ASD"] = (d["ASD"] / d["Total"] * 100).round(1)
    return d.sort_values("Total", ascending=False).reset_index()
