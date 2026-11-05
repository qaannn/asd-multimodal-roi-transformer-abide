"""Langkah 0 (hanya untuk skenario S1-S5): bangun dataset ABIDE I + II (turunan DCAN) dan simpan ke cache.
  python scripts/00_build_dcan_dataset.py --inspect     # PERIKSA DULU: kolom fenotipik ABIDE II & contoh nama berkas S3
  python scripts/00_build_dcan_dataset.py --limit 5     # uji coba 5 subjek per dataset
  python scripts/00_build_dcan_dataset.py               # penuh (unduh ±2 GB per atlas; bisa dilanjutkan bila terputus)"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import prepare_environment, DCAN_ATLAS, DCAN_MIN_SECONDS
from src.dcan import build_dcan_dataset, inspect_dcan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--atlas", default=DCAN_ATLAS); ap.add_argument("--min-seconds", type=float, default=DCAN_MIN_SECONDS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    prepare_environment()
    if a.inspect:
        inspect_dcan(a.atlas); return
    out = None
    if a.limit:                                              # uji coba: jangan menimpa cache penuh
        from src.config import CACHE_DIR
        out = CACHE_DIR + f"data_abide12_dcan_uji{a.limit}.npz"
    build_dcan_dataset(atlas=a.atlas, min_seconds=a.min_seconds, limit=a.limit, workers=a.workers, out=out)


if __name__ == "__main__":
    main()
