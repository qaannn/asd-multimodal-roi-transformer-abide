# Deteksi ASD Multimodal: Konektivitas rs-fMRI (CC200) + Fenotipik pada ABIDE I

Repositori Tugas Akhir (S1 Informatika): membandingkan strategi fusi dua modalitas — konektivitas fungsional
rs-fMRI (atlas Craddock CC200) dan 7 fitur fenotipik — untuk klasifikasi *Autism Spectrum Disorder* (ASD) vs kontrol
pada ABIDE I, dengan baseline linear dan ROI-Transformer (varian tanpa fusi, concat, dan cross-attention).

> **Status.** Eksperimen inti selesai (baseline, model dalam, uji signifikansi). Belum dikerjakan:
> visualisasi atensi (ROI mana yang diperhatikan token klinis) dan evaluasi *leave-one-site-out*.
> Lisensi belum ditentukan; tambahkan sebelum publikasi. Data **tidak** disertakan (lihat bagian Data).

## Temuan utama (kohort 871 subjek, 5-fold berstrata label + situs)

1. **Fusi dua modalitas menambah informasi.** Late fusion (LogReg pada konektivitas + SVM pada klinis, rata-rata
   probabilitas) mencapai AUC pooled 0,810 [0,781; 0,839] vs 0,759 (konektivitas saja) dan 0,737 (klinis saja);
   selisih signifikan (bootstrap berpasangan dan McNemar + Holm).
2. **Model linear mengungguli ROI-Transformer** pada kohort ini: konektivitas saja 0,759 vs 0,671 (signifikan);
   late fusion vs ROI-Transformer + concat 0,810 vs 0,736 (signifikan).
3. **Cross-attention tidak lebih baik dari concat**: AUC pooled 0,684 vs 0,736; akurasi tidak berbeda (McNemar p = 0,81).
   Hipotesis bahwa token klinis memandu pembacaan jaringan otak lewat atensi tidak didukung data ini.
4. **Early fusion fitur** (PCA-50 konektivitas + klinis) AUC 0,750-0,756 tidak melampaui konektivitas saja (0,762).
5. **Backbone citra ImageNet beku** (ResNet-50, Swin-T) tidak dapat membaca matriks konektivitas
   (AUC 0,52-0,58) dan citra `func_mean` hampir tidak informatif (AUC 0,50-0,56); lihat `scripts/05` dan `scripts/legacy_func_mean`.
6. Recall ASD terbaik berasal dari baseline linear (0,66); model dalam pada ambang 0,5 tidak stabil antar-fold.

Semua klaim berlaku untuk kohort dan protokol ini, bukan generalisasi untuk deteksi ASD secara umum.

## Kohort

ABIDE I fenotipe 1.112 (539 ASD / 573 kontrol) -> 1.035 (FILE_ID tersedia) -> **871** (403 ASD / 468 kontrol; 20 situs)
tersedia di ABIDE PCP (C-PAC, `quality_checked=True`). Alasan pengeluaran tahap 2 disimpulkan dari parameter unduhan;
verifikasi sebelum dikutip. Tabel demografi dan per situs dibuat oleh `scripts/01`.

## Struktur repositori

```
.
├── README.md  requirements.txt  .gitignore
├── src/
│   ├── config.py       # path (via variabel lingkungan), konstanta, hiperparameter dasar
│   ├── data.py         # pembersihan fenotipik, koreksi umur, penggabungan, fold, cache
│   ├── models.py       # ROITokenizer, ClinicalTokenizer, CrossAttentionBlock, ROIFusionNet
│   ├── training.py     # 5-fold, early stopping, ensemble seed, ambang dari validasi, random search
│   ├── baselines.py    # baseline linear + prediksi out-of-fold
│   ├── stats.py        # AUC + CI bootstrap, McNemar exact + Holm, selisih AUC berpasangan
│   └── tables.py       # tabel kohort / demografi / data hilang / per situs
├── scripts/
│   ├── 01_prepare_data_and_tables.py
│   ├── 02_baseline_linear.py
│   ├── 03_train_deep.py                  # butuh GPU
│   ├── 04_significance_tests.py
│   ├── 05_diagnostic_frozen_backbones.py # diagnostik (arsip)
│   └── legacy_func_mean/                 # eksplorasi awal citra func_mean (arsip)
├── notebooks/00_colab_runner.ipynb       # menjalankan 01-04 di Colab
└── result/{tabel,metrics,figure}/
```

## Data

Data tidak didistribusikan di repo ini (ukuran dan syarat penggunaan ABIDE). Dibutuhkan:
1. **CSV fenotipik** `Phenotypic_V1_0b_preprocessed1.csv` (pemisah `;`) di `ASD_DATA_DIR`.
2. **Konektivitas CC200** diunduh otomatis oleh `scripts/01` lewat `nilearn.datasets.fetch_abide_pcp`
   (`pipeline='cpac'`, `band_pass_filtering=True`, `global_signal_regression=False`, `quality_checked=True`)
   lalu di-cache ke `ASD_DATA_DIR/cache/data_v2.npz`.

Variabel lingkungan: `ASD_DATA_DIR`, `ASD_RESULT_DIR`, `ASD_ABIDE_DIR` (lihat `src/config.py`).

## Cara menjalankan (Colab)

Gunakan `notebooks/00_colab_runner.ipynb`, atau berurutan:
```
pip install -r requirements.txt
python scripts/01_prepare_data_and_tables.py   # unduh sekali, cache, tabel kohort/demografi
python scripts/02_baseline_linear.py           # CPU
python scripts/03_train_deep.py                # GPU (T4); tuning + 5-fold x 3 seed
python scripts/04_significance_tests.py
```
Skrip 03 berhenti bila tidak ada GPU. Waktu: unduhan beberapa menit, baseline beberapa menit, 03 sekitar satu jam
(perkiraan pada T4; tergantung `N_TRIALS`).

## Protokol evaluasi (anti-kebocoran)

- 5-fold berstrata label + situs (seed 42); fold sama untuk semua model.
- Dalam tiap training fold, 15% disisihkan untuk early stopping (val AUC); imputasi median dan scaler klinis
  di-fit hanya pada sisa data latih; penanda nilai hilang ikut sebagai fitur.
- Konfigurasi model dalam dipilih lewat random search (20 konfigurasi per varian, konfigurasi sama untuk semua varian)
  berdasarkan **val AUC**, bukan AUC tes. Baseline linear tidak dituning.
- Ensemble 3 seed per fold; ambang keputusan dipilih dari validasi (Youden).
- Perbandingan model memakai prediksi out-of-fold: McNemar exact + koreksi Holm, dan selisih AUC bootstrap berpasangan
  (1000 resampel; interval bootstrap tidak dikoreksi untuk perbandingan ganda).

## Hasil

Angka di bawah disalin dari keluaran eksekusi penulis (lihat `result/README.md`).

**Baseline linear (rata-rata ± SD antar-fold)**

| Model | Akurasi | Recall | Presisi | F1 | AUC |
|---|---|---|---|---|---|
| Mayoritas | 0,537 | 0,000 | 0,000 | 0,000 | 0,500 |
| Situs saja | 0,564 | 0,469 | 0,534 | 0,498 | 0,582 ± 0,011 |
| Klinis 7 fitur (SVM) | 0,669 | 0,531 | 0,690 | 0,597 | 0,738 ± 0,021 |
| Konektivitas CC200 (LogReg) | 0,697 | 0,657 | 0,679 | 0,667 | 0,762 ± 0,031 |
| Konektivitas PCA-50 (SVM) | 0,654 | 0,630 | 0,626 | 0,628 | 0,708 ± 0,017 |
| Early fusion PCA-50 + klinis (w=1) | 0,685 | 0,563 | 0,700 | 0,623 | 0,750 ± 0,021 |
| Early fusion PCA-50 + klinis (w=3) | 0,690 | 0,546 | 0,726 | 0,619 | 0,756 ± 0,012 |
| **Late fusion (LogReg + SVM)** | **0,741** | **0,662** | **0,750** | **0,703** | **0,811 ± 0,025** |

**ROI-Transformer (ensemble 3 seed; ± SD antar-fold)**

| Varian | Val AUC | Test AUC | Akurasi (0,5) | Recall (0,5) | Recall (ambang val) |
|---|---|---|---|---|---|
| none (konektivitas saja) | 69,0% ± 3,0 | 68,9% ± 2,4 | 64,5% ± 1,3 | 60,1% ± 10,1 | 63,0% ± 11,2 |
| concat | 78,9% ± 2,7 | 73,8% ± 1,8 | 66,6% ± 1,8 | 52,3% ± 15,8 | 60,8% ± 12,6 |
| cross-attention | 70,8% ± 3,0 | 69,9% ± 6,2 | 63,0% ± 5,7 | 45,0% ± 24,5 | 69,4% ± 9,0 |

**Perbandingan lintas model (prediksi out-of-fold digabung, N = 871)**

| A vs B | ΔAUC [CI 95%] | p Holm | Signifikan |
|---|---|---|---|
| Late fusion vs konektivitas saja | +0,050 [0,033; 0,068] | 0,0036 | Ya |
| Late fusion vs klinis saja | +0,073 [0,042; 0,099] | 0,0003 | Ya |
| Late fusion vs deep concat | +0,075 [0,044; 0,102] | <0,0001 | Ya |
| Late fusion vs deep cross-attention | +0,126 [0,095; 0,159] | <0,0001 | Ya |
| Konektivitas LogReg vs ROI-Transformer (konek. saja) | +0,089 [0,058; 0,120] | 0,0008 | Ya |
| ROI-Transformer (konek. saja) vs + concat | −0,065 [−0,101; −0,030] | 0,8646 | Tidak (akurasi) |
| Deep concat vs deep cross-attention | +0,051 [0,019; 0,084] | 0,8646 | Tidak (akurasi) |

p Holm berasal dari McNemar pada akurasi di ambang masing-masing model; ΔAUC memakai bootstrap berpasangan.
Dua baris terakhir menunjukkan ketidaksesuaian antara AUC dan akurasi: fusi memperbaiki pemeringkatan subjek
tetapi belum terbukti memperbaiki keputusan pada satu ambang.

## Catatan data

- **Koreksi umur.** `AGE_AT_SCAN` pada USM dan STANFORD tersimpan terkalikan 10 di CSV yang dipakai
  (mis. 502,2 untuk 50,2 tahun). `fix_age()` membagi nilai > 64 dengan 10; rentang hasilnya (USM 8,8-50,2;
  STANFORD 7,5-12,9) cocok dengan rentang yang dipublikasikan. Verifikasi terhadap CSV ABIDE asli.
- **BMI dibuang** (hilang ±91%); handedness dibuat biner (kanan / non-kanan); nilai hilang diimputasi median per fold
  dengan penanda hilang.
- Pengeluaran subjek dicatat per tahap pada `tabel_alur_kohort.csv`.

## Keterbatasan

Satu kali pembagian fold (tanpa validasi silang berulang); tanpa evaluasi *leave-one-site-out* sehingga generalisasi
antar-situs belum diuji; satu atlas (CC200); baseline linear tidak dituning sedangkan model dalam dituning;
konektivitas masih dapat mengandung efek situs; atensi belum diinterpretasi; kohort ±870 subjek sehingga model dalam
dari nol rentan overfitting.

## Rujukan (periksa ulang detail sebelum dikutip)

- Di Martino dkk., 2014 — ABIDE (*Molecular Psychiatry*).
- Craddock dkk., 2013 — Preprocessed Connectomes Project; Craddock dkk., 2012 — atlas CC200 (*Human Brain Mapping*).
- Abraham dkk., 2014 — nilearn (*Frontiers in Neuroinformatics*).
- Kan dkk., 2022 — Brain Network Transformer (NeurIPS), sebagai rujukan pendekatan ROI sebagai token.
