# Folder hasil

- `tabel/tabel_alur_kohort.csv`, `tabel_demografi.csv`, `tabel_data_hilang.csv`, `tabel_per_situs.csv`: keluaran
  `scripts/01` (tidak bergantung pada pengaturan penanda hilang). Disalin dari keluaran eksekusi Colab.
- `tabel/run1_dengan_penanda_hilang/`: **run 1** (semua fitur klinis + penanda nilai hilang). Disalin dari keluaran
  `scripts/02`, `03`, `04` pada satu kali eksekusi. Dipertahankan sebagai **analisis sensitivitas**, bukan hasil utama
  (lihat README: penanda `EYE_STATUS_AT_SCAN` hilang terkait diagnosis).
- Hasil utama (tanpa penanda hilang, pengaturan default) akan dihasilkan oleh `scripts/02`-`07` dan ditaruh di
  `tabel/` dan `metrics/` setelah dijalankan ulang; jalankan dengan `ASD_RESULT_DIR` berbeda dari run 1.
