# Folder hasil

Isi awal folder `tabel/` (`baseline_linear_ringkas.csv`, `deep_ablation_ringkas.csv`, `uji_ringkas_oof.csv`,
`uji_berpasangan.csv`, `tabel_alur_kohort.csv`) **disalin dari keluaran eksekusi di Colab** yang dikirim penulis,
bukan dihasilkan ulang saat repo ini disusun. Menjalankan skrip `scripts/01`-`04` akan menimpa file-file ini
dengan keluaran asli beserta berkas per-fold, tuning, dan prediksi out-of-fold (`metrics/*.npz`).
`tabel_demografi.csv` dan `tabel_per_situs.csv` belum disertakan karena baru dihasilkan oleh `scripts/01`.
Angka deep learning pada `deep_ablation_ringkas.csv` berasal dari ringkasan eksekusi yang disalin ke dokumentasi;
verifikasi dengan `tabel/deep_ablation_per_fold.csv` setelah menjalankan ulang.
