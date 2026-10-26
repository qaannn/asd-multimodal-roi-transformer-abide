# Deteksi ASD Multimodal: Konektivitas rs-fMRI (CC200) + Fenotipik pada ABIDE I

Repositori Tugas Akhir (S1 Informatika): membandingkan strategi fusi dua modalitas — konektivitas fungsional rs-fMRI
(atlas Craddock CC200) dan fitur fenotipik — untuk klasifikasi *Autism Spectrum Disorder* (ASD) vs kontrol pada ABIDE I,
dengan baseline linear dan ROI-Transformer (tanpa fusi, concat, cross-attention).

> **Status.** Run 1 selesai (semua langkah 01-04 berjalan dari repo ini). Setelah itu ditemukan **artefak data hilang**
> (di bawah) sehingga hasil yang melibatkan fitur klinis harus dijalankan ulang tanpa penanda hilang (pengaturan default
> sekarang). Angka pada README ini berasal dari run 1 dan **belum final** untuk klaim yang memakai fitur klinis.
> Belum dikerjakan: visualisasi atensi dan evaluasi *leave-one-site-out*. Lisensi belum ditentukan. Data tidak disertakan.

## Peringatan validitas: penanda nilai hilang membocorkan diagnosis

`EYE_STATUS_AT_SCAN` hilang pada 86 subjek di CSV (1.112 subjek), dan **85 di antaranya ASD** (hanya 1 kontrol), tersebar
di 14 situs — jadi bukan sekadar efek situs. Di kohort 871 subjek: hilang 14,1% pada ASD vs 0,2% pada kontrol.
Penanda "nilai hilang" sebagai fitur (dipakai di run 1) karenanya memberi sinyal diagnosis lewat artefak pengumpulan data,
bukan lewat biologi. Uji indikatif pada 1.112 subjek CSV (SVM klinis, 5-fold x 3 ulangan): AUC 0,729 dengan penanda vs
0,671 tanpa penanda (0,678 tanpa EYE dan tanpa penanda; IQ saja 0,665). Karena itu:

- pengaturan default kini **tanpa penanda hilang** (`ASD_MISSING_INDICATORS=0`; nyalakan dengan `=1` untuk sensitivitas);
- `scripts/07_clinical_ablation.py` mengukur dampaknya pada kohort 871 untuk klinis saja dan late fusion;
- semua angka klinis dan fusi di bawah (run 1) diperlakukan sebagai **analisis sensitivitas** sampai run ulang selesai.

Hasil yang tidak memakai fitur klinis tidak terpengaruh: konektivitas LogReg (AUC 0,759) dan ROI-Transformer
konektivitas saja (0,671).

## Temuan sementara (run 1; kohort 871 subjek, 5-fold berstrata label + situs)

1. Konektivitas CC200 informatif (AUC 0,759); ROI-Transformer pada konektivitas saja lebih rendah (0,671), selisih signifikan.
2. Fitur klinis memberi sinyal besar terutama lewat IQ (FIQ ASD 105,8 vs kontrol 111,1; p < 0,001), tetapi sebagian AUC klinis
   run 1 berasal dari artefak data hilang (lihat atas).
3. Late fusion pada run 1 (AUC 0,810) unggul signifikan atas konektivitas saja dan klinis saja; **besarnya keunggulan perlu
   dihitung ulang tanpa penanda hilang**.
4. Cross-attention tidak terbukti lebih baik dari concat (akurasi tidak berbeda, McNemar p = 0,55; AUC concat lebih tinggi).
5. Backbone citra ImageNet beku (ResNet-50, Swin-T) tidak membaca matriks konektivitas (AUC 0,52-0,58); citra `func_mean`
   hampir tidak informatif (AUC 0,50-0,56): lihat `scripts/05` dan `scripts/legacy_func_mean`.

Klaim berlaku untuk kohort dan protokol ini, bukan generalisasi deteksi ASD.

## Kohort

ABIDE I fenotipe 1.112 (539 ASD / 573 kontrol) -> 1.035 (FILE_ID tersedia) -> **871** (403 ASD / 468 kontrol; 20 situs)
di ABIDE PCP (C-PAC, `quality_checked=True`). Alasan pengeluaran tahap 2 disimpulkan dari parameter unduhan; verifikasi
sebelum dikutip. Tabel demografi, data hilang, dan per situs: `result/tabel/`.

## Struktur repositori

```
.
├── README.md  requirements.txt  .gitignore
├── src/
│   ├── config.py       # path, konstanta, hiperparameter dasar, flag penanda hilang
│   ├── data.py         # pembersihan fenotipik, koreksi umur, penggabungan, fold, cache
│   ├── models.py       # ROITokenizer, ClinicalTokenizer, CrossAttentionBlock, ROIFusionNet
│   ├── training.py     # 5-fold, early stopping, ensemble seed, ambang dari validasi, random search
│   ├── baselines.py    # baseline linear + prediksi out-of-fold
│   ├── stats.py        # AUC + CI bootstrap, McNemar exact + Holm, selisih AUC berpasangan
│   ├── site_eval.py    # evaluasi per situs dari prediksi out-of-fold
│   └── tables.py       # tabel kohort / demografi / data hilang / per situs
├── scripts/
│   ├── 01_prepare_data_and_tables.py
│   ├── 02_baseline_linear.py
│   ├── 03_train_deep.py                  # butuh GPU
│   ├── 04_significance_tests.py
│   ├── 05_diagnostic_frozen_backbones.py # diagnostik (arsip)
│   ├── 06_per_site_evaluation.py         # AUC dalam-situs (tanpa pelatihan baru)
│   ├── 07_clinical_ablation.py           # ablasi fitur klinis / penanda hilang
│   └── legacy_func_mean/                 # eksplorasi awal citra func_mean (arsip)
├── notebooks/00_colab_runner.ipynb
└── result/{tabel,metrics,figure}/
```

## Data

Data tidak didistribusikan di repo ini (ukuran dan syarat penggunaan ABIDE). Dibutuhkan:
1. **CSV fenotipik** `Phenotypic_V1_0b_preprocessed1.csv` (pemisah `;`) di `ASD_DATA_DIR`.
2. **Konektivitas CC200** diunduh otomatis oleh `scripts/01` lewat `nilearn.datasets.fetch_abide_pcp`
   (`pipeline='cpac'`, `band_pass_filtering=True`, `global_signal_regression=False`, `quality_checked=True`) lalu di-cache ke
   `ASD_DATA_DIR/cache/data_v2.npz`.

Variabel lingkungan: `ASD_DATA_DIR`, `ASD_RESULT_DIR`, `ASD_ABIDE_DIR`, `ASD_MISSING_INDICATORS` (lihat `src/config.py`).

## Cara menjalankan (Colab)

Gunakan `notebooks/00_colab_runner.ipynb`. Setelah mengganti runtime, jalankan ulang semua sel dari atas (mount Drive dulu).
```
pip install -r requirements.txt
python scripts/01_prepare_data_and_tables.py
python scripts/02_baseline_linear.py
python scripts/03_train_deep.py                # GPU (T4); ±35-60 menit
python scripts/04_significance_tests.py
python scripts/06_per_site_evaluation.py       # opsional
python scripts/07_clinical_ablation.py         # ablasi klinis / penanda hilang
```
Skrip 03 berhenti bila tidak ada GPU. Untuk membandingkan dengan run 1, jalankan dengan `ASD_RESULT_DIR` yang berbeda.

## Protokol evaluasi (anti-kebocoran)

- 5-fold berstrata label + situs (seed 42); fold sama untuk semua model.
- Dalam tiap training fold, 15% untuk early stopping (val AUC); imputasi median dan scaler klinis di-fit hanya pada data latih.
  Penanda nilai hilang: **mati** pada pengaturan default (lihat peringatan validitas).
- Konfigurasi model dalam dipilih lewat random search (20 konfigurasi per varian, konfigurasi sama untuk semua varian)
  berdasarkan **val AUC**, bukan AUC tes. Baseline linear tidak dituning. Run 1 memilih: none = rs05, concat = rs13, cross = rs14.
- Ensemble 3 seed per fold; ambang keputusan dari validasi (Youden).
- Perbandingan model memakai prediksi out-of-fold: McNemar exact + Holm, dan selisih AUC bootstrap berpasangan (1000 resampel;
  interval bootstrap tidak dikoreksi untuk perbandingan ganda).

## Hasil run 1 (dengan penanda hilang; analisis sensitivitas)

Sumber: `result/tabel/run1_dengan_penanda_hilang/`.

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
| Late fusion (LogReg + SVM) | 0,741 | 0,662 | 0,750 | 0,703 | 0,811 ± 0,025 |

**ROI-Transformer (ensemble 3 seed; ± SD antar-fold)**

| Varian | Val AUC | Test AUC | Akurasi (0,5) | Recall (0,5) | Recall (ambang val) |
|---|---|---|---|---|---|
| none | 69,1% ± 3,1 | 68,8% ± 2,1 | 64,4% ± 1,4 | 58,3% ± 11,4 | 62,2% ± 11,9 |
| concat | 78,4% ± 3,1 | 74,0% ± 2,1 | 66,9% ± 2,4 | 54,1% ± 18,3 | 61,3% ± 11,8 |
| cross-attention | 70,9% ± 3,7 | 69,4% ± 6,0 | 63,8% ± 6,1 | 50,8% ± 22,9 | 61,7% ± 18,7 |

**Perbandingan lintas model (prediksi out-of-fold digabung, N = 871)**

| A vs B | ΔAUC [CI 95%] | p Holm (McNemar) |
|---|---|---|
| Late fusion vs konektivitas saja | +0,050 [0,033; 0,068] | 0,0036 |
| Late fusion vs klinis saja | +0,073 [0,042; 0,099] | 0,0003 |
| Late fusion vs deep concat | +0,080 [0,050; 0,110] | <0,0001 |
| Late fusion vs deep cross-attention | +0,124 [0,092; 0,155] | <0,0001 |
| Konektivitas LogReg vs ROI-Transformer (konek. saja) | +0,088 [0,058; 0,119] | 0,0006 |
| ROI-Transformer (konek. saja) vs + concat | −0,058 [−0,094; −0,022] | 0,669 |
| Deep concat vs deep cross-attention | +0,043 [0,006; 0,079] | 0,669 |

p Holm berasal dari McNemar pada akurasi di ambang masing-masing model; ΔAUC memakai bootstrap berpasangan. Dua baris
terakhir menunjukkan ketidaksesuaian AUC vs akurasi: fusi memperbaiki pemeringkatan tetapi belum terbukti memperbaiki
keputusan pada satu ambang.

## Catatan data

- **Koreksi umur.** `AGE_AT_SCAN` pada USM dan STANFORD terkalikan 10 di CSV yang dipakai (mis. 502,2 untuk 50,2 tahun);
  `fix_age()` membagi nilai > 64 dengan 10 (hasil: USM 8,8-50,2; STANFORD 7,5-12,9, sesuai rentang terpublikasi).
  Verifikasi terhadap CSV ABIDE asli.
- **BMI dibuang** (hilang ±91%); handedness dibuat biner (kanan / non-kanan).
- Pengeluaran subjek dicatat pada `tabel_alur_kohort.csv`.

## Keterbatasan

Satu pembagian fold (tanpa validasi silang berulang); tanpa evaluasi *leave-one-site-out*; satu atlas (CC200); baseline linear
tidak dituning sedangkan model dalam dituning; konektivitas masih dapat mengandung efek situs; atensi belum diinterpretasi;
kohort ±870 subjek sehingga model dalam dari nol rentan overfitting; hasil klinis run 1 terpengaruh artefak data hilang.

## Rujukan (periksa ulang detail sebelum dikutip)

- Di Martino dkk., 2014 — ABIDE (*Molecular Psychiatry*).
- Craddock dkk., 2013 — Preprocessed Connectomes Project; Craddock dkk., 2012 — atlas CC200 (*Human Brain Mapping*).
- Abraham dkk., 2014 — nilearn (*Frontiers in Neuroinformatics*).
- Kan dkk., 2022 — Brain Network Transformer (NeurIPS), sebagai rujukan pendekatan ROI sebagai token.
