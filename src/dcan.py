"""ABIDE I + II (turunan DCAN / pipeline ABCD-HCP) dari S3 publik fcp-indi:
unduh deret waktu ROI (CIFTI .ptseries.nii) + masker gerakan, QC gerakan, matriks korelasi, fenotipik ABIDE II,
lalu gabungkan menjadi satu cache (data_abide12_dcan.npz) bersekema sama dengan cache PCP.
CATATAN: jalur S3 dan pola nama berkas berasal dari struktur publik bucket; verifikasi dengan
`python scripts/00_build_dcan_dataset.py --inspect` sebelum unduhan massal."""
import os, re, time, shutil, urllib.parse, urllib.request
import concurrent.futures as cf
import numpy as np
import pandas as pd

from .config import (prepare_environment, DCAN_RAW_DIR, DCAN_ATLAS, DCAN_MIN_SECONDS, DCAN_FD, CSV_PATH, CACHE_DIR, TABEL_DIR, FEATURES)
from .data import load_clinical_safe, clean_clinical, make_folds

BASE = "https://s3.amazonaws.com/fcp-indi"
PREFIX = {"abide1": "data/Projects/ABIDE/Derivatives/DCAN/", "abide2": "data/Projects/ABIDE2/Derivatives/DCAN/"}
RAW_ABIDE2 = "data/Projects/ABIDE2/RawData/"


# ------------------------------------------------------------------ S3 publik
def _http_get(url, timeout=60, tries=3):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:                                   # noqa: BLE001
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def parse_listing(xml):
    prefixes = re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix>", xml)
    keys = list(zip(re.findall(r"<Key>([^<]+)</Key>", xml), map(int, re.findall(r"<Size>([^<]+)</Size>", xml))))
    nxt = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
    return prefixes, keys, (nxt[0] if nxt else None)


def s3_list(prefix, delimiter=True):
    """(daftar prefix, [(key, ukuran)]) dengan paginasi ListObjectsV2."""
    prefixes, keys, token = [], [], None
    while True:
        url = f"{BASE}?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000" + ("&delimiter=/" if delimiter else "")
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        p, k, token = parse_listing(_http_get(url).decode())
        prefixes += p; keys += k
        if not token:
            return prefixes, keys


def list_subject_prefixes(dataset):
    """{SUB_ID (int): prefix S3} untuk seluruh subjek DCAN pada dataset ('abide1' | 'abide2')."""
    prefixes, _ = s3_list(PREFIX[dataset])
    out = {}
    for p in prefixes:
        m = re.search(r"sub-(\d+)/?$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def _tokens(name):
    return tuple(re.findall(r"(?:ses|run|task|acq)-[A-Za-z0-9]+", name))


def pick_files(keys, atlas):
    """Pilih satu time series (prioritas task-rest; urutan leksikografis = sesi/run paling awal) + masker gerakan yang sepadan."""
    ts = [(k, s) for k, s in keys if "/func/" in k and "ptseries.nii" in k and f"atlas-{atlas}_" in k]
    if not ts:
        return None, "tanpa time series untuk atlas ini"
    rest = [x for x in ts if "task-rest" in x[0]] or ts
    k_ts, s_ts = sorted(rest)[0]
    masks = [(k, s) for k, s in keys if "/func/" in k and k.endswith("motion_mask.mat")]
    if not masks:
        return None, "tanpa motion mask"
    tok = _tokens(k_ts.split("/")[-1])
    same = [m for m in masks if _tokens(m[0].split("/")[-1]) == tok]
    k_m, s_m = sorted(same or masks)[0]
    return dict(key_ts=k_ts, size_ts=s_ts, key_mask=k_m, size_mask=s_m), ""


def download_file(key, dest, size, tries=3):
    """Unduh ke .part lalu pindahkan; berkas dengan ukuran sama dilewati."""
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "sudah ada"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for t in range(1, tries + 1):
        try:
            with urllib.request.urlopen(f"{BASE}/{urllib.parse.quote(key)}", timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(part) != size:
                raise IOError("ukuran tidak cocok")
            os.replace(part, dest)
            return "diunduh"
        except Exception:                                   # noqa: BLE001
            time.sleep(2 * t)
    raise IOError(f"gagal mengunduh {key}")


# ------------------------------------------------------------------ pembacaan & QC
def read_ptseries(path):
    import nibabel as nib
    x = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("ptseries bukan matriks waktu x ROI")
    return x


def motion_info(mat_path, fd=DCAN_FD):
    import scipy.io
    md = np.atleast_1d(scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["motion_data"])
    by = {round(float(e.FD_threshold), 2): e for e in md}
    e = by.get(round(fd, 2)) or by[min(by, key=lambda t: abs(t - fd))]
    return dict(remaining_seconds=float(e.remaining_seconds), total_frames=int(e.total_frame_count), tr=float(e.epi_TR))


def connectivity(ts):
    """(T,R) -> korelasi Pearson (R,R) float32; ROI datar (std=0) berkorelasi 0; diagonal 0."""
    ts = np.asarray(ts, np.float64); ts = ts - ts.mean(0); sd = ts.std(0); ok = sd > 1e-8
    C = np.zeros((ts.shape[1], ts.shape[1]))
    z = ts[:, ok] / sd[ok]
    C[np.ix_(ok, ok)] = (z.T @ z) / len(z)
    np.fill_diagonal(C, 0.0)
    return np.clip(C, -1, 1).astype(np.float32)


# ------------------------------------------------------------------ fenotipik ABIDE II
_CAND = {
    "SUB_ID": ["SUB_ID", "participant_id", "sub_id", "subject", "subject_id"],
    "DX_GROUP": ["DX_GROUP", "dx_group", "group", "diagnosis", "dx"],
    "AGE_AT_SCAN": ["AGE_AT_SCAN", "age_at_scan", "age"],
    "SEX": ["SEX", "sex", "gender"],
    "HANDEDNESS_CATEGORY": ["HANDEDNESS_CATEGORY", "handedness_category", "handedness"],
    "FIQ": ["FIQ", "fiq", "full_iq"], "VIQ": ["VIQ", "viq", "verbal_iq"], "PIQ": ["PIQ", "piq", "performance_iq"],
    "EYE_STATUS_AT_SCAN": ["EYE_STATUS_AT_SCAN", "eyes_status_at_scan", "eye_status_at_scan", "eye_status"],
}


def _first_col(df, names):
    low = {c.lower(): c for c in df.columns}
    return next((low[n.lower()] for n in names if n.lower() in low), None)


def _dx(v):
    s = str(v).strip().lower()
    if s in {"1", "1.0"} or "autis" in s or "asd" in s:
        return 1
    if s in {"2", "2.0", "td", "tc", "hc"} or "control" in s or "typical" in s or "healthy" in s:
        return 2
    return np.nan


def _sex(v):
    s = str(v).strip().lower()
    return 1 if s in {"1", "1.0", "m", "male"} else (2 if s in {"2", "2.0", "f", "female"} else np.nan)


def _eye(v):
    s = str(v).strip().lower()
    return 1 if s in {"1", "1.0", "open", "opened"} else (2 if s in {"2", "2.0", "closed"} else np.nan)


def _hand(v):
    s = str(v).strip().lower()
    if s.startswith("r"):
        return "R"
    return "L" if (s.startswith("l") or s.startswith("a") or s.startswith("m")) else np.nan


def standardize_abide2(df, site):
    """Tabel participants.tsv satu situs -> kolom bergaya ABIDE I (sebelum clean_clinical). Mengembalikan (df, laporan)."""
    out, laporan = pd.DataFrame(index=df.index), {}
    for col, names in _CAND.items():
        src = _first_col(df, names)
        laporan[col] = src
        out[col] = df[src] if src is not None else np.nan
    out["SUB_ID"] = out["SUB_ID"].map(lambda v: int(re.sub(r"\D", "", str(v))) if re.sub(r"\D", "", str(v)) else np.nan)
    out["DX_GROUP"] = out["DX_GROUP"].map(_dx)
    out["SEX"] = out["SEX"].map(_sex)
    out["EYE_STATUS_AT_SCAN"] = out["EYE_STATUS_AT_SCAN"].map(_eye)
    out["HANDEDNESS_CATEGORY"] = out["HANDEDNESS_CATEGORY"].map(_hand)
    out["BMI"] = np.nan
    out["SITE_ID"] = site
    return out, laporan


def load_abide2_phenotypic(raw_dir=DCAN_RAW_DIR):
    """Unduh participants.tsv tiap situs ABIDE II, seragamkan kolom, lalu bersihkan dengan aturan yang sama dengan ABIDE I."""
    d = os.path.join(raw_dir, "abide2_participants"); os.makedirs(d, exist_ok=True)
    sites, _ = s3_list(RAW_ABIDE2)
    frames, lap = [], {}
    for s in sites:
        name = s.rstrip("/").split("/")[-1]
        if not name.startswith("ABIDEII-"):
            continue
        f = os.path.join(d, name + ".tsv")
        if not os.path.exists(f):
            try:
                open(f, "wb").write(_http_get(f"{BASE}/{s}participants.tsv"))
            except Exception as e:                           # noqa: BLE001
                print(f"  participants.tsv {name} gagal: {e}"); continue
        std, lp = standardize_abide2(pd.read_csv(f, sep="\t", encoding="latin1"), name.replace("ABIDEII-", ""))
        frames.append(std); lap[name] = lp
    if not frames:
        raise RuntimeError("Tidak ada participants.tsv ABIDE II yang berhasil diunduh")
    allp = pd.concat(frames, ignore_index=True)
    missing = [c for c in ("SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX") if allp[c].isna().all()]
    if missing:
        raise RuntimeError(f"Kolom wajib ABIDE II tidak terpetakan: {missing}. Jalankan --inspect dan sesuaikan _CAND di src/dcan.py.")
    return clean_clinical(allp, fix_ages=False)


# ------------------------------------------------------------------ pembangun dataset
def _proses_subjek(item, atlas, raw_dir, min_seconds, fd):
    src, ds, sub_id, prefix = item
    try:
        _, keys = s3_list(prefix, delimiter=False)
        files, why = pick_files(keys, atlas)
        if files is None:
            return dict(src=src, sub_id=sub_id, status="dikeluarkan", alasan=why)
        d = os.path.join(raw_dir, ds, f"sub-{sub_id}")
        f_ts, f_m = os.path.join(d, files["key_ts"].split("/")[-1]), os.path.join(d, files["key_mask"].split("/")[-1])
        download_file(files["key_mask"], f_m, files["size_mask"])
        info = motion_info(f_m, fd)
        if info["remaining_seconds"] < min_seconds:
            return dict(src=src, sub_id=sub_id, status="dikeluarkan",
                        alasan=f"gerakan: sisa {info['remaining_seconds']:.0f} dtk < {min_seconds:.0f}")
        download_file(files["key_ts"], f_ts, files["size_ts"])
        ts = read_ptseries(f_ts)
        if ts.shape[0] < 50:
            return dict(src=src, sub_id=sub_id, status="dikeluarkan", alasan="time series terlalu pendek")
        return dict(src=src, sub_id=sub_id, status="ok", alasan="", M=connectivity(ts), sisa_detik=info["remaining_seconds"])
    except Exception as e:                                   # noqa: BLE001
        return dict(src=src, sub_id=sub_id, status="dikeluarkan", alasan=f"galat: {type(e).__name__}: {e}"[:120])


def build_dcan_dataset(atlas=DCAN_ATLAS, min_seconds=DCAN_MIN_SECONDS, fd=DCAN_FD, limit=None, workers=8,
                       raw_dir=DCAN_RAW_DIR, out=None):
    """Bangun cache ABIDE I + II (DCAN). limit = uji coba N subjek per dataset."""
    prepare_environment()                                   # menolak bila path Drive dipakai tetapi Drive belum ter-mount
    out = out or CACHE_DIR + "data_abide12_dcan.npz"
    os.makedirs(raw_dir, exist_ok=True)
    p1 = load_clinical_safe(CSV_PATH)
    p1 = p1[["SUB_ID", "SITE_ID", "DX_GROUP"] + FEATURES].assign(src=1)
    p2 = load_abide2_phenotypic(raw_dir)
    p2 = p2[["SUB_ID", "SITE_ID", "DX_GROUP"] + FEATURES].assign(src=2)
    pheno = pd.concat([p1, p2], ignore_index=True).dropna(subset=["SUB_ID"])
    pheno["SUB_ID"] = pheno["SUB_ID"].astype(int)
    print(f"Fenotipik ASD/TD: ABIDE I {len(p1)} | ABIDE II {len(p2)}")

    todo, log = [], []
    for src, ds in ((1, "abide1"), (2, "abide2")):
        pref = list_subject_prefixes(ds)
        sub = pheno[pheno.src == src]
        sub = sub.head(limit) if limit else sub
        for sid in sub["SUB_ID"]:
            if sid in pref:
                todo.append((src, ds, sid, pref[sid]))
            else:
                log.append(dict(src=src, sub_id=sid, status="dikeluarkan", alasan="tidak ada di DCAN"))
    print(f"Subjek dengan turunan DCAN: {len(todo)}")
    with cf.ThreadPoolExecutor(workers) as ex:
        hasil = list(ex.map(lambda it: _proses_subjek(it, atlas, raw_dir, min_seconds, fd), todo))
    ok = [h for h in hasil if h["status"] == "ok"]
    log += [{k: v for k, v in h.items() if k not in ("M",)} for h in hasil]
    log_df = pd.DataFrame(log)
    if limit is None:
        log_df.to_csv(TABEL_DIR + "dcan_inclusion_log.csv", index=False)
    print(log_df.assign(alasan=log_df["alasan"].str.slice(0, 40)).groupby(["src", "status", "alasan"]).size().to_string())
    if not ok:
        raise RuntimeError("Tidak ada subjek yang lolos. Jalankan --inspect untuk memeriksa nama berkas dan jalur S3.")

    idx = pd.DataFrame({"src": [h["src"] for h in ok], "SUB_ID": [h["sub_id"] for h in ok]})
    P = idx.merge(pheno, on=["src", "SUB_ID"], how="left")
    y = (P["DX_GROUP"].to_numpy() == 1).astype(int)
    site = np.array([f"A{s}_{x}" for s, x in zip(P["src"], P["SITE_ID"].astype(str))], dtype="U24")
    D = dict(M=np.stack([h["M"] for h in ok]), y=y, sub_id=P["SUB_ID"].to_numpy().astype(int), site=site,
             clin=P[FEATURES].to_numpy(dtype=np.float32), src=P["src"].to_numpy().astype(int),
             sisa_detik=np.array([h["sisa_detik"] for h in ok], np.float32))
    D["fold"] = make_folds(D["y"], D["site"])
    np.savez_compressed(out, **D)
    print(f"Tersimpan: {out} | N={len(y)} (ABIDE I {(D['src'] == 1).sum()}, ABIDE II {(D['src'] == 2).sum()}) | "
          f"ASD={y.sum()} TD={(1 - y).sum()} | ROI={D['M'].shape[-1]} | situs={len(set(site))}")
    return D


def inspect_dcan(atlas=DCAN_ATLAS, raw_dir=DCAN_RAW_DIR):
    """Pemeriksaan cepat sebelum unduhan massal: kolom ABIDE II dan contoh nama berkas DCAN."""
    sites, _ = s3_list(RAW_ABIDE2)
    nama = next(s.rstrip("/").split("/")[-1] for s in sites if s.rstrip("/").split("/")[-1].startswith("ABIDEII-"))
    df = pd.read_csv(__import__("io").StringIO(_http_get(f"{BASE}/{RAW_ABIDE2}{nama}/participants.tsv").decode()), sep="\t")
    std, lap = standardize_abide2(df, nama)
    print(f"[ABIDE II / {nama}] kolom asli: {list(df.columns)}\nPemetaan: {lap}\nNaN per kolom hasil:\n{std.isna().sum().to_string()}")
    for ds in ("abide1", "abide2"):
        pref = list_subject_prefixes(ds)
        sid, p = sorted(pref.items())[0]
        _, keys = s3_list(p, delimiter=False)
        files, why = pick_files(keys, atlas)
        print(f"\n[{ds}] {len(pref)} subjek di DCAN; contoh sub-{sid}: {files if files else why}")
        print("  contoh key:", [k for k, _ in keys if '/func/' in k][:4])
