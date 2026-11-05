# Folder hasil

Semua angka berasal dari keluaran eksekusi Colab (repo ini, pengaturan default: tanpa penanda nilai hilang) yang disalin ke sini;
jalankan ulang skrip untuk menimpa dengan berkas asli (per-fold, tuning, prediksi out-of-fold di `metrics/`).

- `tabel/` (hasil utama, run 2): `baseline_linear_ringkas`, `deep_ablation_ringkas`, `uji_ringkas_oof`, `uji_berpasangan`,
  `situs_auc`, `situs_ringkasan`, `situs_late_fusion_vs_lain`, `ablasi_fitur_klinis`, serta `tabel_alur_kohort`,
  `tabel_demografi`, `tabel_data_hilang`, `tabel_per_situs`.
- `tabel/run1_dengan_penanda_hilang/`: run 1 (dengan penanda nilai hilang), disimpan sebagai analisis sensitivitas.
- `figure/situs_auc.png` dibuat oleh `scripts/06`; salin dari folder hasil di Drive bila ingin disertakan.
