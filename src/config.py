"""Konfigurasi terpusat. Semua path dapat diubah lewat variabel lingkungan."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lokasi data (TIDAK disertakan di repo). Default = struktur Google Drive penulis di Colab.
DATA_DIR = os.environ.get("ASD_DATA_DIR", "/content/drive/MyDrive/Proposal_Topik_Autism/DATASET/")
CACHE_DIR = os.path.join(DATA_DIR, "cache") + os.sep            # data_v2.npz dibuat otomatis di sini
CSV_PATH = os.path.join(DATA_DIR, "Phenotypic_V1_0b_preprocessed1.csv")
ABIDE_DIR = os.environ.get("ASD_ABIDE_DIR", "/content/abide_pcp")  # cache unduhan nilearn

# Lokasi keluaran (tabel kecil, prediksi out-of-fold, gambar)
RESULT_DIR = os.environ.get("ASD_RESULT_DIR", os.path.join(ROOT, "result"))
TABEL_DIR = os.path.join(RESULT_DIR, "tabel") + os.sep
METRICS_DIR = os.path.join(RESULT_DIR, "metrics") + os.sep
FIGURE_DIR = os.path.join(RESULT_DIR, "figure") + os.sep


def prepare_environment():
    """Membuat folder keluaran. Menolak bekerja bila path Google Drive dipakai padahal Drive belum ter-mount
    (kalau tidak, folder 'hantu' terbentuk di /content/drive dan drive.mount() berikutnya gagal)."""
    for d in (CACHE_DIR, TABEL_DIR, METRICS_DIR, FIGURE_DIR):
        if d.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
            raise RuntimeError("Google Drive belum ter-mount di sesi ini. Jalankan sel drive.mount('/content/drive') "
                               "lebih dulu (setelah mengganti runtime, semua sel harus dijalankan ulang dari atas).")
        os.makedirs(d, exist_ok=True)


SEED = 42
FEATURES = ["AGE_AT_SCAN", "SEX", "HANDEDNESS_CATEGORY", "FIQ", "VIQ", "PIQ", "BMI", "EYE_STATUS_AT_SCAN"]
CLIN7 = [0, 1, 2, 3, 4, 5, 7]            # tanpa BMI (hilang ~90%)
# Penanda "nilai hilang" sebagai fitur. DEFAULT MATI: di ABIDE, EYE_STATUS_AT_SCAN hilang pada 85 dari 86 kasus
# justru di kelompok ASD (di 14 situs), sehingga penanda hilang membocorkan diagnosis lewat artefak data.
# Nyalakan hanya untuk analisis sensitivitas: ASD_MISSING_INDICATORS=1
USE_MISSING_INDICATORS = os.environ.get("ASD_MISSING_INDICATORS", "0") == "1"

N_ROI = 200                               # atlas Craddock CC200
IU = np.triu_indices(N_ROI, k=1)

# Hiperparameter dasar model dalam dan pelatihan
MODEL_KW = dict(d_model=64, n_heads=4, n_enc_layers=2, n_cross_layers=1, dropout=0.2, head_dropout=0.3)
CFG = dict(epochs=80, patience=15, min_epochs=10, lr=5e-4, wd=0.05, bs=32, noise=0.02)

# Pengaturan eksperimen model dalam
DO_TUNE = True
N_TRIALS = 20            # konfigurasi acak per varian (turunkan bila waktu mepet)
SEARCH_SEED = 123        # sama untuk semua varian -> perbandingan adil
SEEDS_FINAL = (0, 1, 2)  # ensemble seed pada run akhir
MIN_RECALL = None        # mis. 0.8 untuk memprioritaskan recall saat memilih ambang
