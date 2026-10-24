# Eksplorasi awal: citra `func_mean` (arsip)

Dua skrip ini adalah tahap awal sebelum beralih ke matriks konektivitas:
`hari1_data_cache.py` membuat cache 3 irisan otak (224x224) dari `*_func_mean.nii.gz` (883 subjek) dan
`hari2_baseline.py` mengekstrak fitur ResNet-50 / Swin-T beku lalu menjalankan baseline sklearn.

Temuan: citra `func_mean` (rata-rata temporal) hampir tidak informatif untuk ASD
(AUC citra saja: ResNet-50 0,557; Swin-T 0,504; setara tebakan). Karena itu pipeline utama memakai
deret waktu ROI -> matriks konektivitas (lihat folder `scripts/` di atas).
Angka *klinis* dari tahap ini dihitung sebelum koreksi umur dan telah digantikan oleh hasil utama;
skrip di folder ini sudah diberi koreksi umur, tetapi tidak dipakai dalam hasil utama.
