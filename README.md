# Deteksi ASD Multimodal: Konektivitas rs-fMRI (CC200) + Fenotipik pada ABIDE I

Repositori Tugas Akhir (S1 Informatika): membandingkan strategi fusi dua modalitas — konektivitas fungsional rs-fMRI
(atlas Craddock CC200) dan fitur fenotipik — untuk klasifikasi *Autism Spectrum Disorder* (ASD) vs kontrol pada ABIDE I,
dengan baseline linear dan ROI-Transformer (tanpa fusi, concat, cross-attention).

> **Status.** Eksperimen inti selesai (langkah 01-07 dijalankan dari repo ini; hasil utama = tanpa penanda nilai hilang).
> Belum dikerjakan: visualisasi atensi dan evaluasi *leave-one-site-out*. Lisensi belum ditentukan. Data tidak disertakan.

## Temuan utama (kohort 871 subjek, 5-fold berstrata label + situs)

1. **Konektivitas CC200 adalah sumber sinyal utama**: LogReg AUC 0,759 [0,725; 0,791].
2. **Sinyal klinis yang sah hampir seluruhnya berasal dari IQ.** Tanpa penanda nilai hilang, klinis saja AUC 0,659; IQ saja 0,641;
   umur + jenis kelamin saja ≈ acak (0,467).
3. **Late fusion** (LogReg konektivitas + SVM klinis, rata-rata probabilitas) terbaik: AUC 0,779 [0,748; 0,810]. Keunggulannya atas
   konektivitas saja **kecil tetapi signifikan**: ΔAUC +0,020 [0,007; 0,033], p Holm = 0,018, dan konsisten di dalam situs
   (lebih tinggi pada 8 dari 10 situs yang dihitung). Atas klinis saja: +0,120 [0,085; 0,153].
4. **ROI-Transformer belum menyamai baseline linear**: pada konektivitas saja 0,672 vs 0,759 (signifikan); late fusion vs deep
   concat +0,079 [0,049; 0,110] dan vs deep cross-attention +0,094 [0,062; 0,125] (keduanya signifikan).
5. **Fusi klinis pada model dalam belum terbukti membantu**, dan cross-attention tidak terbukti lebih baik dari concat:
   concat − konektivitas saja +0,027 [−0,009; 0,062]; concat − cross-attention +0,014 [−0,021; 0,050] (keduanya mencakup nol).
6. **Early fusion fitur** (PCA-50 + klinis) AUC 0,707-0,724 tidak melampaui konektivitas saja (0,762); late fusion lebih baik (0,781).
7. **Backbone citra ImageNet beku** (ResNet-50, Swin-T) tidak membaca matriks konektivitas (AUC 0,52-0,58), dan citra `func_mean`
   hampir tidak informatif (0,50-0,56): lihat `scripts/05` dan `scripts/legacy_func_mean`.
8. **Recall**: baseline linear 0,65-0,66; model dalam pada ambang dari validasi sangat bervariasi antar-fold
   (mis. cross-attention 68,0% ± 14,4); hipotesis peningkatan recall oleh fusi tidak didukung jelas.

Klaim berlaku untuk kohort dan protokol ini, bukan generalisasi deteksi ASD.

## Peringatan validitas: penanda nilai hilang membocorkan diagnosis

`EYE_STATUS_AT_SCAN` hilang pada 86 subjek di CSV (1.112 subjek), dan **85 di antaranya ASD** (satu kontrol), tersebar di 14 situs
(di kohort 871: 14,1% pada ASD vs 0,2% pada kontrol). Penanda "nilai hilang" sebagai fitur karenanya memberi sinyal diagnosis lewat
artefak pengumpulan data. Dampak pada kohort 871 (AUC pooled, `scripts/07`):

| Varian klinis | AUC klinis saja | AUC late fusion |
|---|---|---|
| (konektivitas saja, pembanding) | — | 0,759 |
| A. 7 fitur + penanda hilang (run 1; **tidak dipakai**) | 0,737 | 0,810 |
| **B. 7 fitur tanpa penanda hilang (hasil utama)** | **0,659** | **0,779** |
| C. 6 fitur tanpa EYE + penanda hilang | 0,681 | 0,787 |
| D. 6 fitur tanpa EYE, tanpa penanda hilang | 0,660 | 0,779 |
| E. IQ saja | 0,641 | 0,776 |
| F. umur + jenis kelamin saja | 0,467 | 0,754 |

Keunggulan late fusion atas konektivitas saja turun dari +0,050 (run 1) menjadi +0,020 setelah artefak dihilangkan. Pengaturan
default kini **tanpa penanda hilang** (`ASD_MISSING_INDICATORS=0`; `=1` hanya untuk analisis sensitivitas). Hasil run 1 disimpan di
`result/tabel/run1_dengan_penanda_hilang/`. Penambahan umur + jenis kelamin saja pada konektivitas menurunkan AUC (0,754 < 0,759):
fitur tak informatif menambah derau.

## Kohort

ABIDE I fenotipe 1.112 (539 ASD / 573 kontrol) -> 1.035 (FILE_ID tersedia) -> **871** (403 ASD / 468 kontrol; 20 situs) di ABIDE PCP
(C-PAC, `quality_checked=True`). Alasan pengeluaran tahap 2 disimpulkan dari parameter unduhan; verifikasi sebelum dikutip.

| Variabel | ASD | Kontrol | p |
|---|---|---|---|
| Umur (tahun) | 17,1 ± 8,0 | 16,8 ± 7,2 | 0,663 |
| Full-scale IQ | 105,8 ± 17,1 | 111,1 ± 12,1 | <0,001 |
| Verbal IQ | 104,9 ± 18,1 | 111,6 ± 13,1 | <0,001 |
| Performance IQ | 105,7 ± 17,3 | 107,9 ± 13,0 | 0,052 |
| Laki-laki / perempuan | 349 / 54 | 378 / 90 | 0,027 |
| Tangan kanan / non-kanan | 231 / 39 | 300 / 28 | 0,032 |
| Mata terbuka / tertutup | 237 / 109 | 320 / 147 | 1,000 |

Data hilang: handedness 31,3%, VIQ 17,2%, PIQ 15,6%, FIQ 6,9%, eye status 6,7%, BMI 90,7% (BMI dibuang). Tabel lengkap dan per situs:
`result/tabel/`.

## Struktur repositori

```
.
├── README.md  requirements.txt  .gitignore
├── src/
│   ├── config.py       # path, konstanta, hiperparameter dasar, flag penanda hilang
│   ├── data.py         # pembersihan fenotipik, koreksi umur, penggabungan, fold, cache
│   ├── models.py       # ROITokenizer, ClinicalTokenizer, CrossAttentionBlock, ROIFusionNet
│   ├── training.py     # 5-fold, early stopping, ensemble seed, ambang dari validasi, random search
│   ├── baselines.py    # baseline linear + prediksi out-of-fold (ukuran ROI dinamis, ComBat opsional)
│   ├── dcan.py         # ABIDE I+II (DCAN): unduh S3, QC gerakan, korelasi, fenotipik ABIDE II
│   ├── harmonize.py    # ComBat anti-kebocoran (dipelajari dari data latih)
│   ├── resnet_baseline.py  # ResNet-50 beku + SVM (baseline DL lama)
│   ├── scenarios.py    # definisi & runner 5 skenario + analisis tambahan
│   ├── report.py       # kompilasi ringkasan (akurasi diutamakan) + uji akurasi berpasangan
│   ├── stats.py        # AUC + CI bootstrap, McNemar exact + Holm, selisih AUC berpasangan
│   ├── site_eval.py    # evaluasi per situs dari prediksi out-of-fold
│   └── tables.py       # tabel kohort / demografi / data hilang / per situs
├── scripts/
│   ├── 00_build_dcan_dataset.py          # (skenario S1-S5) bangun dataset ABIDE I+II DCAN
│   ├── 01_prepare_data_and_tables.py
│   ├── 02_baseline_linear.py
│   ├── 03_train_deep.py                  # butuh GPU
│   ├── 04_significance_tests.py
│   ├── 05_diagnostic_frozen_backbones.py # diagnostik (arsip)
│   ├── 06_per_site_evaluation.py         # AUC dalam-situs (tanpa pelatihan baru)
│   ├── 07_clinical_ablation.py           # ablasi fitur klinis / penanda hilang
│   ├── 08_run_scenarios.py               # jalankan skenario S1-S5 / analisis tambahan
│   ├── 09_compile_scenarios.py           # ringkasan CSV + uji akurasi untuk BAB 4
│   └── legacy_func_mean/                 # eksplorasi awal citra func_mean (arsip)
├── notebooks/00_colab_runner.ipynb
└── result/{tabel,metrics,figure}/
```

## Rencana lima skenario resmi (ABIDE I + II, DCAN) — belum dijalankan

Batasan dosen: maksimal 5 skenario utama; **akurasi** metrik utama (ambang 0,5 untuk semua skenario; AUC sekunder). Variasi lain
(klinis saja, late fusion linear, ablasi, ComBat) = analisis tambahan, bukan skenario.

| # | Dataset | Model |
|---|---|---|
| S1 | ABIDE I (DCAN) | Konektivitas fMRI + LogReg (baseline ML monomodal) |
| S2 | ABIDE I (DCAN) | ResNet-50 beku + SVM pada konektivitas (baseline DL lama) |
| S3 | ABIDE I (DCAN) | ROI-Transformer + klinis, cross-attention (model baru) |
| S4 | ABIDE I + II (DCAN) | Konektivitas fMRI + LogReg (baseline pada data gabungan) |
| S5 | ABIDE I + II (DCAN) | ROI-Transformer + klinis, cross-attention (final; checkpoint `.pth` disimpan) |

```
python scripts/00_build_dcan_dataset.py --inspect   # periksa kolom ABIDE II & nama berkas S3 DULU
python scripts/00_build_dcan_dataset.py --limit 5   # uji coba kecil
python scripts/00_build_dcan_dataset.py             # penuh
python scripts/08_run_scenarios.py --id S1 S2 S3
python scripts/08_run_scenarios.py --id S4 S5
python scripts/08_run_scenarios.py --supp T1 T2 T3 T4 --supp-dataset abide1_dcan     # analisis tambahan
python scripts/08_run_scenarios.py --supp T2 T5 T6 --supp-dataset abide12_dcan       # termasuk ComBat
python scripts/09_compile_scenarios.py             # -> result/tabel/skenario_ringkasan.csv (+ _bab4.md, uji akurasi)
```
Catatan: pipeline DCAN berbeda dari C-PAC (hasil utama di atas), sehingga angka skenario ini tidak sebanding langsung dengan tabel hasil
di atas. ComBat dipelajari hanya dari data latih tiap fold dan tanpa label. Jalur S3/nama berkas DCAN dan kolom fenotipik ABIDE II
belum diuji terhadap server sesungguhnya (verifikasi dengan `--inspect`).

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
python scripts/06_per_site_evaluation.py
python scripts/07_clinical_ablation.py
```
Skrip 03 berhenti bila tidak ada GPU. Langkah 06 membutuhkan keluaran 02, 03 (prediksi out-of-fold) di `ASD_RESULT_DIR/metrics`.

## Protokol evaluasi (anti-kebocoran)

- 5-fold berstrata label + situs (seed 42); fold sama untuk semua model.
- Dalam tiap training fold, 15% untuk early stopping (val AUC); imputasi median dan scaler klinis di-fit hanya pada data latih.
  Penanda nilai hilang **mati** (lihat peringatan validitas).
- Konfigurasi model dalam dipilih lewat random search (20 konfigurasi per varian, konfigurasi sama untuk semua varian) berdasarkan
  **val AUC**, bukan AUC tes. Terpilih: none = rs05, concat = rs13, cross = rs14. Baseline linear tidak dituning.
- Ensemble 3 seed per fold; ambang keputusan dari validasi (Youden).
- Perbandingan model memakai prediksi out-of-fold: McNemar exact + Holm, dan selisih AUC bootstrap berpasangan (1000 resampel;
  interval bootstrap tidak dikoreksi untuk perbandingan ganda).

## Hasil

**Baseline linear (rata-rata ± SD antar-fold)**

| Model | Akurasi | Recall | Presisi | F1 | AUC |
|---|---|---|---|---|---|
| Mayoritas | 0,537 | 0,000 | 0,000 | 0,000 | 0,500 |
| Situs saja | 0,564 | 0,469 | 0,534 | 0,498 | 0,582 ± 0,011 |
| Klinis 7 fitur (SVM) | 0,621 | 0,518 | 0,608 | 0,559 | 0,659 ± 0,029 |
| Konektivitas CC200 (LogReg) | 0,697 | 0,657 | 0,679 | 0,667 | 0,762 ± 0,031 |
| Konektivitas PCA-50 (SVM) | 0,654 | 0,630 | 0,626 | 0,628 | 0,708 ± 0,017 |
| Early fusion PCA-50 + klinis (w=1) | 0,658 | 0,560 | 0,653 | 0,602 | 0,724 ± 0,036 |
| Early fusion PCA-50 + klinis (w=3) | 0,643 | 0,540 | 0,636 | 0,583 | 0,707 ± 0,027 |
| **Late fusion (LogReg + SVM)** | **0,729** | **0,652** | **0,733** | **0,689** | **0,781 ± 0,029** |

**ROI-Transformer (ensemble 3 seed; rata-rata ± SD antar-fold)**

| Varian | Val AUC | Test AUC | Akurasi (0,5) | Recall (0,5) | Recall (ambang val) |
|---|---|---|---|---|---|
| none (konektivitas saja) | 69,7% ± 3,7 | 68,9% ± 2,5 | 64,4% ± 1,5 | 58,1% ± 11,0 | 54,9% ± 18,0 |
| concat | 76,8% ± 2,9 | 70,9% ± 1,4 | 66,1% ± 2,8 | 49,5% ± 14,8 | 55,7% ± 10,7 |
| cross-attention | 72,2% ± 3,2 | 69,9% ± 3,4 | 65,2% ± 2,3 | 66,6% ± 15,7 | 68,0% ± 14,4 |

**Prediksi out-of-fold digabung (N = 871)**

| Model | AUC [CI 95%] | Akurasi | Recall | Presisi |
|---|---|---|---|---|
| Klinis saja (SVM) | 0,659 [0,622; 0,696] | 0,622 | 0,506 | 0,611 |
| Konektivitas saja (LogReg) | 0,759 [0,725; 0,791] | 0,697 | 0,658 | 0,678 |
| Late fusion (LogReg + SVM) | 0,779 [0,748; 0,810] | 0,729 | 0,653 | 0,733 |
| ROI-Transformer (konektivitas saja) | 0,672 [0,636; 0,707] | 0,618 | 0,551 | 0,594 |
| ROI-Transformer + klinis (concat) | 0,699 [0,666; 0,733] | 0,646 | 0,558 | 0,634 |
| ROI-Transformer + klinis (cross-attention) | 0,685 [0,650; 0,722] | 0,629 | 0,677 | 0,586 |

**Perbandingan berpasangan**

| A vs B | ΔAUC (A − B) [CI 95%] | p Holm (McNemar) | Signifikan |
|---|---|---|---|
| Late fusion vs konektivitas saja | +0,020 [0,007; 0,033] | 0,018 | Ya |
| Late fusion vs klinis saja | +0,120 [0,085; 0,153] | <0,0001 | Ya |
| Late fusion vs deep concat | +0,079 [0,049; 0,110] | <0,0001 | Ya |
| Late fusion vs deep cross-attention | +0,094 [0,062; 0,125] | <0,0001 | Ya |
| Konektivitas LogReg vs ROI-Transformer (konek. saja) | +0,087 [0,058; 0,119] | <0,0001 | Ya |
| ROI-Transformer (konek. saja) vs + concat | −0,027 [−0,062; 0,009] | 0,284 | Tidak |
| Deep concat vs deep cross-attention | +0,014 [−0,021; 0,050] | 0,390 | Tidak |

p Holm berasal dari McNemar pada akurasi di ambang masing-masing model; ΔAUC memakai bootstrap berpasangan.

**AUC dalam-situs** (prediksi out-of-fold; hanya situs dengan N ≥ 30 dan kelas minoritas ≥ 8: 10 dari 20 situs)

| Model | AUC pooled | AUC dalam-situs (berbobot N) | rata-rata ± SD antar-situs |
|---|---|---|---|
| Klinis saja | 0,659 | 0,647 | 0,634 ± 0,076 |
| Konektivitas saja | 0,759 | 0,756 | 0,739 ± 0,073 |
| **Late fusion** | **0,779** | **0,780** | **0,766 ± 0,067** |
| ROI-Transformer (konek. saja) | 0,672 | 0,668 | 0,663 ± 0,085 |
| ROI-Transformer + concat | 0,699 | 0,702 | 0,684 ± 0,076 |
| ROI-Transformer + cross-attention | 0,685 | 0,690 | 0,684 ± 0,067 |

AUC dalam-situs hampir sama dengan AUC pooled, sehingga hasil tidak tampak berasal dari perbedaan proporsi ASD antar-situs.
Late fusion lebih tinggi dari konektivitas saja pada 8 dari 10 situs (Wilcoxon indikatif p = 0,0098; daya rendah dan tidak dikoreksi).
Tabel per situs: `result/tabel/situs_auc.csv`.

## Catatan data

- **Koreksi umur.** `AGE_AT_SCAN` pada USM dan STANFORD terkalikan 10 di CSV yang dipakai (mis. 502,2 untuk 50,2 tahun); `fix_age()`
  membagi nilai > 64 dengan 10 (hasil: USM 8,8-50,2; STANFORD 7,5-12,9, sesuai rentang terpublikasi). Verifikasi terhadap CSV ABIDE asli.
- **BMI dibuang** (hilang ±91%); handedness dibuat biner (kanan / non-kanan).
- Pengeluaran subjek dicatat pada `tabel_alur_kohort.csv`.

## Keterbatasan

Satu pembagian fold (tanpa validasi silang berulang); tanpa evaluasi *leave-one-site-out* (evaluasi per situs memakai model yang dilatih
dengan semua situs); hanya 10 dari 20 situs yang memenuhi syarat AUC dalam-situs; satu atlas (CC200); baseline linear tidak dituning
sedangkan model dalam dituning; keunggulan late fusion atas konektivitas saja kecil (+0,020); sinyal klinis hampir seluruhnya IQ, yang
sendiri berbeda antar-kelompok dan dapat dipengaruhi cara rekrutmen; atensi belum diinterpretasi; kohort ±870 subjek sehingga model
dalam dari nol rentan overfitting; hasil per-fold model dalam bervariasi.

## Rujukan (periksa ulang detail sebelum dikutip)

- Di Martino dkk., 2014 — ABIDE (*Molecular Psychiatry*).
- Craddock dkk., 2013 — Preprocessed Connectomes Project; Craddock dkk., 2012 — atlas CC200 (*Human Brain Mapping*).
- Abraham dkk., 2014 — nilearn (*Frontiers in Neuroinformatics*).
- Kan dkk., 2022 — Brain Network Transformer (NeurIPS), sebagai rujukan pendekatan ROI sebagai token.
