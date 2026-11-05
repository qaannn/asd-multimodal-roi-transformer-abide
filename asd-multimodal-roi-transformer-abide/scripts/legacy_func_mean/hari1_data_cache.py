# =====================================================================
# HARI 1 - Bangun cache data yang BERSIH & TERCATAT (jalankan sekali)
# Output di OUT_DIR:
#   cache_3slice.npz   : images (N,3,224,224) float16 [0,1], y, sub_id, site,
#                        file_id, clin (N,8; NaN dipertahankan), miss (N,8)
#   inclusion_log.csv  : status & alasan setiap subjek (bahan diagram 1112 -> N)
#   folds.csv          : sub_id, fold (0-4), tetap dipakai SEMUA eksperimen
# Catatan: belum saya jalankan pada data asli Anda (hanya uji dengan data
# sintetis + nibabel tiruan). Jalankan, lalu baca ringkasan yang dicetak.
# =====================================================================
# !pip -q install "pandas<3" nibabel
import os, re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold

# ---------- KONFIGURASI (ubah path sesuai Drive Anda) ----------
CSV_PATH = "/content/drive/MyDrive/Proposal_Topik_Autism/DATASET/Phenotypic_V1_0b_preprocessed1.csv"
IMG_ROOT = "/content/dataset_fmri/"
OUT_DIR = "/content/drive/MyDrive/Proposal_Topik_Autism/DATASET/cache/"
SEED = 42
FEATURES = ["AGE_AT_SCAN", "SEX", "HANDEDNESS_CATEGORY",
            "FIQ", "VIQ", "PIQ", "BMI", "EYE_STATUS_AT_SCAN"]
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. Data klinis: bersih, NaN dipertahankan ----------
def fix_age(a):
    """CSV: umur USM & STANFORD tersimpan x10; nilai > 64 dibagi 10 (aman dipanggil berulang)."""
    a = np.asarray(a, dtype=np.float64).copy(); m = a > 64; a[m] = a[m] / 10.0; return a

def load_clinical(csv_path):
    df = pd.read_csv(csv_path, sep=";")           # tanpa on_bad_lines: gagal keras bila CSV rusak
    df = df[df["DX_GROUP"].isin([1, 2])].copy()
    df["LABEL"] = (df["DX_GROUP"] == 1).astype(int)   # ASD=1, TD=0

    for c in FEATURES:                              # -9999 = kode missing ABIDE
        if c != "HANDEDNESS_CATEGORY":
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(-9999, np.nan)

    df["AGE_AT_SCAN"] = fix_age(df["AGE_AT_SCAN"].to_numpy())          # koreksi satuan umur
    # Handedness: teks -> biner (0=kanan, 1=non-kanan), NaN dipertahankan
    h = df["HANDEDNESS_CATEGORY"].astype("string").str.strip()
    miss_h = h.isna().to_numpy() | (h.fillna("") == "-9999").to_numpy()
    df["HANDEDNESS_CATEGORY"] = np.where(miss_h, np.nan,
                                         (h.fillna("") != "R").astype(float).to_numpy())
    # SEX: 1=laki-laki, 2=perempuan -> 0/1 ; EYE: 1=terbuka, 2=tertutup -> 0/1
    df["SEX"] = np.where(df["SEX"].isna(), np.nan, (df["SEX"] == 2).astype(float))
    df["EYE_STATUS_AT_SCAN"] = np.where(df["EYE_STATUS_AT_SCAN"].isna(), np.nan,
                                        (df["EYE_STATUS_AT_SCAN"] == 2).astype(float))
    return df.reset_index(drop=True)


# ---------- 2. Indeks file eksak berdasarkan FILE_ID ----------
def build_file_index(root):
    idx = {}
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(".nii.gz"):
                key = re.sub(r"_func_mean\.nii\.gz$|\.nii\.gz$", "", f)
                idx[key] = os.path.join(r, f)
    return idx


# ---------- 3. NIfTI -> (3,224,224) ----------
def volume_to_3slice(vol, fracs=(0.40, 0.50, 0.60), size=224, p_lo=1, p_hi=99):
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol.mean(axis=-1)
    vol = np.nan_to_num(vol)
    brain = vol > 0
    if brain.sum() < 100:
        raise ValueError("volume kosong")
    xs = np.where(brain.any(axis=2).any(axis=1))[0]
    ys = np.where(brain.any(axis=2).any(axis=0))[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    lo, hi = np.percentile(vol[brain], [p_lo, p_hi])
    vol = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1) * brain
    D = vol.shape[2]
    chans = []
    for f in fracs:
        s = vol[x0:x1, y0:y1, int(D * f)]
        s = zoom(s, (size / s.shape[0], size / s.shape[1]), order=1)[:size, :size]
        if s.shape != (size, size):
            s = np.pad(s, ((0, size - s.shape[0]), (0, size - s.shape[1])))
        chans.append(s)
    return np.clip(np.stack(chans, 0), 0, 1).astype(np.float32)


# ---------- 4. Jalankan ----------
def main():
    df = load_clinical(CSV_PATH)
    index = build_file_index(IMG_ROOT)
    print(f"Subjek ASD/TD di CSV: {len(df)} | file .nii.gz ditemukan: {len(index)}")

    imgs, keep, status, reason = [], [], [], []
    for i, row in df.iterrows():
        fid = str(row["FILE_ID"]).strip()
        if fid.lower() == "no_filename":
            status.append("excluded"); reason.append("FILE_ID=no_filename (tidak ada citra di ABIDE)"); continue
        path = index.get(fid)
        if path is None:
            status.append("excluded"); reason.append("file tidak ada di zip/folder"); continue
        try:
            imgs.append(volume_to_3slice(nib.load(path).get_fdata()))
            keep.append(i); status.append("included"); reason.append("")
        except Exception as e:
            status.append("excluded"); reason.append(f"gagal diproses: {type(e).__name__}: {e}")

    log = df[["SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP"]].copy()
    log["status"], log["alasan"] = status, reason
    log.to_csv(OUT_DIR + "inclusion_log.csv", index=False)

    d = df.loc[keep].reset_index(drop=True)
    X_img = np.stack(imgs).astype(np.float16)
    clin = d[FEATURES].to_numpy(dtype=np.float32)
    miss = np.isnan(clin).astype(np.float32)
    y = d["LABEL"].to_numpy()

    # Fold berstrata label+situs (strata langka <5 digabung agar StratifiedKFold tidak error)
    strat = d["LABEL"].astype(str) + "_" + d["SITE_ID"].astype(str)
    rare = strat.map(strat.value_counts()) < 5
    strat[rare] = d["LABEL"].astype(str)[rare] + "_RARE"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = np.full(len(d), -1)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(d)), strat)):
        fold[te] = k
    pd.DataFrame({"sub_id": d["SUB_ID"], "fold": fold}).to_csv(OUT_DIR + "folds.csv", index=False)

    np.savez_compressed(OUT_DIR + "cache_3slice.npz", images=X_img, y=y,
                        sub_id=d["SUB_ID"].to_numpy(), site=np.array(d["SITE_ID"].astype(str).tolist(), dtype="U20"),
                        file_id=np.array(d["FILE_ID"].astype(str).tolist(), dtype="U40"), clin=clin, miss=miss)

    # ---------- Ringkasan: salin ke Bab 3 & kirim ke saya ----------
    print("\n=== RINGKASAN ===")
    print(f"Termasuk: {len(d)} | Dikeluarkan: {len(log) - len(d)}")
    print(log[log.status == "excluded"]["alasan"].str.slice(0, 60).value_counts().to_string())
    print("Distribusi kelas (ASD=1):", dict(zip(*np.unique(y, return_counts=True))))
    print("Jumlah per situs:\n", d["SITE_ID"].value_counts().to_string())
    print("Persen missing per fitur:\n", dict(zip(FEATURES, (miss.mean(0) * 100).round(1))))
    print("Citra: shape", X_img.shape, "| min/max", float(X_img.min()), float(X_img.max()),
          "| mean", round(float(X_img.astype(np.float32).mean()), 3))
    print("Ukuran per fold:", np.bincount(fold))

if __name__ == "__main__":
    main()
