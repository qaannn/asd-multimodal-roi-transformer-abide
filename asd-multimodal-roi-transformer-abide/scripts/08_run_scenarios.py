"""Langkah 8: jalankan skenario resmi S1-S5 (dan analisis tambahan) secara berurutan; tiap skenario menyimpan hasilnya sendiri,
sehingga aman bila sesi putus (skenario yang sudah selesai tidak perlu diulang).
  python scripts/08_run_scenarios.py --id S1 S2 S3           # ABIDE I
  python scripts/08_run_scenarios.py --id S4 S5              # ABIDE I + II (butuh scripts/00 selesai)
  python scripts/08_run_scenarios.py --supp T1 T2 T3 T4 --supp-dataset abide1_dcan
  python scripts/08_run_scenarios.py --supp T5 T6 --supp-dataset abide12_dcan   # ComBat (analisis tambahan)"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import prepare_environment, SEEDS_FINAL
from src.scenarios import SCENARIOS, SUPPLEMENT, run_scenario, run_supplement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", nargs="*", default=[], choices=list(SCENARIOS) + ["all"])
    ap.add_argument("--supp", nargs="*", default=[], choices=list(SUPPLEMENT)); ap.add_argument("--supp-dataset", default="abide12_dcan")
    ap.add_argument("--abide1-pipeline", choices=["dcan", "pcp"], default="dcan",
                    help="pcp = pakai ABIDE I versi C-PAC (871 subjek) untuk S1-S3 (tidak sepipeline dengan ABIDE II)")
    ap.add_argument("--trials", type=int, default=12, help="konfigurasi acak untuk skenario model dalam (S3; S5 memakai ulang)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_FINAL)); ap.add_argument("--retune-s5", action="store_true")
    a = ap.parse_args()
    prepare_environment()
    ids = list(SCENARIOS) if "all" in a.id else a.id
    for sid in ids:
        run_scenario(sid, a.abide1_pipeline, a.trials, tuple(a.seeds), a.retune_s5)
    for tid in a.supp:
        run_supplement(tid, a.supp_dataset, a.abide1_pipeline, a.trials, tuple(a.seeds))


if __name__ == "__main__":
    main()
