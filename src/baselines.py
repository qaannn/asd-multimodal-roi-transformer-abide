"""Baseline linear (scikit-learn): situs saja, klinis, konektivitas, early fusion (PCA), late fusion."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from .config import SEED, CLIN7, IU


def clin_block(D, cols=CLIN7):
    def f(tr, te):
        X = D["clin"][:, cols]
        imp = SimpleImputer(strategy="median", add_indicator=True); sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
        return Xtr, sc.transform(imp.transform(X[te]))
    return f

def site_block(D):
    def f(tr, te):
        S = D["site"].reshape(-1, 1); oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return oh.fit_transform(S[tr]), oh.transform(S[te])
    return f

def conn_raw_block(D):                      # segitiga atas, di-standardisasi pada train
    X = np.stack([m[IU] for m in D["M"]])
    def f(tr, te):
        sc = StandardScaler(); return sc.fit_transform(X[tr]), sc.transform(X[te])
    return f

def conn_pca_block(D, n_pca=50):
    X = np.stack([m[IU] for m in D["M"]])
    def f(tr, te):
        sc = StandardScaler(); pca = PCA(n_components=n_pca, random_state=SEED)
        Xtr = pca.fit_transform(sc.fit_transform(X[tr])); Xte = pca.transform(sc.transform(X[te]))
        s = Xtr.std(0).mean(); return Xtr / s, Xte / s
    return f

def early_fusion_block(img_f, clin_f, w=1.0):
    def f(tr, te):
        a, b = img_f(tr, te), clin_f(tr, te)
        return np.hstack([a[0], w * b[0]]), np.hstack([a[1], w * b[1]])
    return f

SVM = lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED)
LR_STRONG = lambda: LogisticRegression(C=0.001, class_weight="balanced", max_iter=3000)   # untuk 19.900 fitur
LR = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
DUMMY = lambda: DummyClassifier(strategy="most_frequent")

def _metrics(name, k, y, pred, score):
    return dict(model=name, fold=k, acc=accuracy_score(y, pred), recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, score) if len(np.unique(score)) > 1 else 0.5)

def run_cv(D, name, block, clf_factory):
    rows = []
    for k in sorted(np.unique(D["fold"])):
        te, tr = D["fold"] == k, D["fold"] != k
        Xtr, Xte = block(tr, te)
        clf = clf_factory().fit(Xtr, D["y"][tr])
        score = clf.decision_function(Xte) if hasattr(clf, "decision_function") else np.zeros(te.sum())
        rows.append(_metrics(name, k, D["y"][te], clf.predict(Xte), score))
    return pd.DataFrame(rows)

def run_cv_late(D, name, img_f, img_clf, clin_f, clin_clf):
    """Late fusion: rata-rata probabilitas dua model terpisah (pembanding untuk early fusion)."""
    rows = []
    for k in sorted(np.unique(D["fold"])):
        te, tr = D["fold"] == k, D["fold"] != k
        a_tr, a_te = img_f(tr, te); b_tr, b_te = clin_f(tr, te)
        p1 = img_clf().fit(a_tr, D["y"][tr]).predict_proba(a_te)[:, 1]
        p2 = clin_clf().fit(b_tr, D["y"][tr]).predict_proba(b_te)[:, 1]
        p = (p1 + p2) / 2
        rows.append(_metrics(name, k, D["y"][te], (p > 0.5).astype(int), p))
    return pd.DataFrame(rows)

SVM_P = lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED)

def run_all(D):
    runs = [
        run_cv(D, "0 Mayoritas (batas tebakan)", lambda tr, te: (np.zeros((tr.sum(), 1)), np.zeros((te.sum(), 1))), DUMMY),
        run_cv(D, "1 Situs saja (LogReg)", site_block(D), LR),
        run_cv(D, "2 Klinis 7 fitur (SVM)", clin_block(D), SVM),
        run_cv(D, "3 Konektivitas CC200 (LogReg)", conn_raw_block(D), LR_STRONG),
        run_cv(D, "4 Konektivitas PCA50 (SVM)", conn_pca_block(D), SVM),
        run_cv(D, "5 Early fusion PCA50+klinis (w=1)", early_fusion_block(conn_pca_block(D), clin_block(D), 1.0), SVM),
        run_cv(D, "6 Early fusion PCA50+klinis (w=3)", early_fusion_block(conn_pca_block(D), clin_block(D), 3.0), SVM),
        run_cv_late(D, "7 Late fusion (LogReg konek. + SVM klinis)", conn_raw_block(D), LR_STRONG, clin_block(D), SVM_P),
    ]
    per_fold = pd.concat(runs, ignore_index=True)
    agg = per_fold.groupby("model", sort=False)[["acc", "recall", "precision", "f1", "auc"]].agg(["mean", "std"])
    table = pd.DataFrame({m: agg[m]["mean"].round(3).astype(str) + " ± " + agg[m]["std"].round(3).astype(str)
                          for m in ["acc", "recall", "precision", "f1", "auc"]})
    return per_fold, table


def oof_linear(D):
    y, fold, N = D["y"], D["fold"], len(D["y"])
    X_conn = np.stack([m[IU] for m in D["M"]])
    X_clin = D["clin"][:, CLIN7]
    p_conn, p_clin = np.zeros(N), np.zeros(N)
    for k in sorted(np.unique(fold)):
        te, tr = fold == k, fold != k
        sc = StandardScaler(); a_tr = sc.fit_transform(X_conn[tr]); a_te = sc.transform(X_conn[te])
        imp = SimpleImputer(strategy="median", add_indicator=True); sc2 = StandardScaler()
        b_tr = sc2.fit_transform(imp.fit_transform(X_clin[tr])); b_te = sc2.transform(imp.transform(X_clin[te]))
        p_conn[te] = LogisticRegression(C=0.001, class_weight="balanced", max_iter=3000).fit(a_tr, y[tr]).predict_proba(a_te)[:, 1]
        p_clin[te] = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True,
                         random_state=SEED).fit(b_tr, y[tr]).predict_proba(b_te)[:, 1]
        print(f"  baseline linear fold {k} selesai")
    return {"Klinis saja (SVM)": (p_clin, 0.5),
            "Konektivitas saja (LogReg)": (p_conn, 0.5),
            "Late fusion (LogReg + SVM)": ((p_conn + p_clin) / 2, 0.5)}
