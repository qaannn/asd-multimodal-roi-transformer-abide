"""Langkah 9: kompilasi hasil skenario ke result/tabel/ (akurasi diutamakan, AUC di sebelahnya):
skenario_ringkasan.csv, skenario_ringkasan_bab4.md, skenario_uji_akurasi.csv, analisis_tambahan_ringkasan.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.config import prepare_environment, TABEL_DIR, METRICS_DIR
from src.report import summarize_scenarios, summarize_supplement, paired_tests


def main():
    prepare_environment()
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print("=== 5 SKENARIO (rata-rata ± SD antar-fold; akurasi pada ambang 0,5) ===")
    print(summarize_scenarios(METRICS_DIR, TABEL_DIR).to_string(index=False))
    t = paired_tests(METRICS_DIR, TABEL_DIR)
    if len(t):
        print("\n=== UJI BERPASANGAN PADA AKURASI (subjek sama) ===\n" + t.to_string(index=False))
    s = summarize_supplement(METRICS_DIR, TABEL_DIR)
    if len(s):
        print("\n=== ANALISIS TAMBAHAN ===\n" + s.to_string(index=False))
    print("\nTersimpan di", TABEL_DIR)


if __name__ == "__main__":
    main()
