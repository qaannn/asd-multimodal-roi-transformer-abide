"""Langkah 6: ablasi fitur klinis - seberapa besar sinyal klinis berasal dari artefak data hilang (EYE_STATUS)?
Memakai fold yang sama (data_v2.npz). Mengukur AUC klinis saja dan late fusion dengan konektivitas (LogReg)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.config import prepare_environment, TABEL_DIR, IU, SEED
from src.data import prepare_dataset

# indeks kolom clin: 0 AGE, 1 SEX, 2 HAND, 3 FIQ, 4 VIQ, 5 PIQ, 6 BMI, 7 EYE
VARIANTS = [
    ("A. 7 fitur + penanda hilang (konfigurasi lama)", [0, 1, 2, 3, 4, 5, 7], True),
    ("B. 7 fitur, tanpa penanda hilang (konfigurasi baru)", [0, 1, 2, 3, 4, 5, 7], False),
    ("C. 6 fitur tanpa EYE, + penanda hilang", [0, 1, 2, 3, 4, 5], True),
    ("D. 6 fitur tanpa EYE, tanpa penanda hilang", [0, 1, 2, 3, 4, 5], False),
    ("E. IQ saja (FIQ, VIQ, PIQ)", [3, 4, 5], False),
    ("F. umur + jenis kelamin saja", [0, 1], False),
]


def oof_conn(D):
    y, fold = D["y"], D["fold"]; X = np.stack([m[IU] for m in D["M"]]); p = np.zeros(len(y))
    for k in sorted(np.unique(fold)):
        te, tr = fold == k, fold != k
        sc = StandardScaler(); a, b = sc.fit_transform(X[tr]), sc.transform(X[te])
        p[te] = LogisticRegression(C=0.001, class_weight="balanced", max_iter=3000).fit(a, y[tr]).predict_proba(b)[:, 1]
    return p


def oof_clin(D, cols, indicator):
    y, fold, X = D["y"], D["fold"], D["clin"][:, cols]; p = np.zeros(len(y))
    for k in sorted(np.unique(fold)):
        te, tr = fold == k, fold != k
        imp = SimpleImputer(strategy="median", add_indicator=indicator); sc = StandardScaler()
        a = sc.fit_transform(imp.fit_transform(X[tr])); b = sc.transform(imp.transform(X[te]))
        p[te] = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True,
                    random_state=SEED).fit(a, y[tr]).predict_proba(b)[:, 1]
    return p


def main():
    prepare_environment()
    D = prepare_dataset(); y = D["y"]
    pc = oof_conn(D)
    auc_conn = roc_auc_score(y, pc)
    rows = [{"Varian klinis": "(konektivitas saja, pembanding)", "AUC klinis saja": np.nan,
             "AUC late fusion + konektivitas": round(auc_conn, 3)}]
    for name, cols, ind in VARIANTS:
        p = oof_clin(D, cols, ind)
        rows.append({"Varian klinis": name, "AUC klinis saja": round(roc_auc_score(y, p), 3),
                     "AUC late fusion + konektivitas": round(roc_auc_score(y, (pc + p) / 2), 3)})
        print(rows[-1])
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
    print("\n(AUC pooled dari prediksi out-of-fold, N=%d)\n%s" % (len(y), out.to_string(index=False)))
    out.to_csv(TABEL_DIR + "ablasi_fitur_klinis.csv", index=False)


if __name__ == "__main__":
    main()
