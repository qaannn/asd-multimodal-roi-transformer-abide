"""Langkah 1: siapkan data (unduh ABIDE PCP sekali, cache) lalu buat tabel kohort/demografi."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.config import CSV_PATH, TABEL_DIR
from src.data import prepare_dataset
from src.tables import tabel_alur_kohort, tabel_demografi, tabel_hilang, tabel_situs


def main():
    D = prepare_dataset()                               # umur sudah dikoreksi (fix_age)
    raw = pd.read_csv(CSV_PATH, sep=";")
    out = {"tabel_alur_kohort": tabel_alur_kohort(raw, D["sub_id"], D["y"]),
           "tabel_demografi": tabel_demografi(D["clin"], D["y"]),
           "tabel_data_hilang": tabel_hilang(D["clin"], D["y"]),
           "tabel_per_situs": tabel_situs(D["site"], D["y"])}
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
    for name, df in out.items():
        print(f"\n=== {name} ===\n{df.to_string(index=False)}")
        df.to_csv(TABEL_DIR + name + ".csv", index=False)


if __name__ == "__main__":
    main()
