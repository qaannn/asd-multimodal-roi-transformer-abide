"""Data: pembersihan fenotipik, penggabungan dengan konektivitas ABIDE PCP, fold, cache."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CACHE_DIR, CSV_PATH, ABIDE_DIR, SEED, FEATURES

def fix_age(a):
    """CSV: umur USM & STANFORD tersimpan x10 (mis. 502.2 = 50.2 tahun). Umur sah maks 64 -> nilai > 64 dibagi 10.
    Aman dipanggil berulang (setelah koreksi semua nilai <= 64)."""
    a = np.asarray(a, dtype=np.float64).copy()
    m = a > 64
    a[m] = a[m] / 10.0
    return a

def load_clinical_safe(csv_path):
    """Memuat dan membersihkan fitur klinis dari CSV lokal (NaN dipertahankan)."""
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["DX_GROUP"].isin([1, 2])].copy()
    for c in FEATURES:
        if c != "HANDEDNESS_CATEGORY":
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(-9999, np.nan)
    df["AGE_AT_SCAN"] = fix_age(df["AGE_AT_SCAN"].to_numpy())          # koreksi satuan umur
    h = df["HANDEDNESS_CATEGORY"].astype("string").str.strip()
    miss_h = h.isna().to_numpy() | (h.fillna("") == "-9999").to_numpy()
    df["HANDEDNESS_CATEGORY"] = np.where(miss_h, np.nan, (h.fillna("") != "R").astype(float).to_numpy())
    df["SEX"] = np.where(df["SEX"].isna(), np.nan, (df["SEX"] == 2).astype(float))
    df["EYE_STATUS_AT_SCAN"] = np.where(df["EYE_STATUS_AT_SCAN"].isna(), np.nan,
                                        (df["EYE_STATUS_AT_SCAN"] == 2).astype(float))
    return df.reset_index(drop=True)

def prepare_dataset(cache_file=CACHE_DIR + "data_v2.npz", force=False):
    """Jika cache ada -> langsung dimuat. Jika tidak -> unduh ABIDE PCP, gabungkan dengan klinis, simpan."""
    if os.path.exists(cache_file) and not force:
        z = np.load(cache_file)
        D = {k: z[k] for k in z.files}
        n_fix = int((D["clin"][:, 0] > 64).sum())
        D["clin"][:, 0] = fix_age(D["clin"][:, 0])                      # patch cache lama (umur x10)
        print(f"Umur dikoreksi pada {n_fix} subjek (nilai > 64 dibagi 10)")
        print(f"Cache dimuat: N={len(D['y'])} | ASD={int(D['y'].sum())} TD={int((1 - D['y']).sum())}")
        return D

    from nilearn import datasets                         # diimpor di sini agar tidak perlu jika cache sudah ada
    from nilearn.connectome import ConnectivityMeasure
    print("Mengunduh konektivitas ABIDE PCP (CC200)... (bisa beberapa menit)")
    abide = datasets.fetch_abide_pcp(derivatives=['rois_cc200'], pipeline='cpac',
                                     band_pass_filtering=True, global_signal_regression=False,
                                     quality_checked=True, data_dir=ABIDE_DIR)
    M_all = ConnectivityMeasure(kind='correlation').fit_transform(abide.rois_cc200).astype(np.float32)
    ph = abide.phenotypic
    sub_abide = np.array(ph['SUB_ID']).astype(int)
    site_abide = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in ph['SITE_ID']], dtype="U20")
    y_abide = (np.array(ph['DX_GROUP']) == 1).astype(int)

    df_clin = load_clinical_safe(CSV_PATH).set_index("SUB_ID")
    ok = np.array([s in df_clin.index for s in sub_abide])
    M, sub, site = M_all[ok], sub_abide[ok], site_abide[ok]
    d = df_clin.loc[sub]
    y = (d["DX_GROUP"].to_numpy() == 1).astype(int)
    print(f"Cocok dengan CSV klinis: {ok.sum()} dari {len(ok)} | kecocokan label: {(y == y_abide[ok]).mean():.3f} (harus 1.000)")
    clin = d[FEATURES].to_numpy(dtype=np.float32)

    strat = pd.Series(y.astype(str)) + "_" + pd.Series(site.astype(str))
    rare = strat.map(strat.value_counts()) < 5
    strat[rare] = pd.Series(y.astype(str))[rare] + "_RARE"
    fold = np.full(len(sub), -1)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(sub)), strat)):
        fold[te] = k
    D = dict(M=M, y=y, sub_id=sub, site=site, clin=clin, fold=fold)
    np.savez_compressed(cache_file, **D)
    print(f"Data siap & tersimpan di Drive! N={len(sub)} | ASD={y.sum()} | TD={(1 - y).sum()}")
    return D
