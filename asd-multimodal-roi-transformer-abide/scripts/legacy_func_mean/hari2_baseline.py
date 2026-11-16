# =====================================================================
# HARI 2 - Baseline jujur di atas cache hari 1 (tanpa melatih jaringan)
# Tempel per sel "# %%" di Colab. Runtime: GPU (T4) disarankan untuk SEL A.
#  SEL A : ekstrak fitur beku ResNet-50 & Swin-T (sekali, disimpan ke Drive)
#  SEL B : baseline sklearn, 5-fold memakai folds.csv, hasil mean +- SD
# SEL B sudah diuji dengan data sintetis; SEL A (torch) belum bisa saya jalankan.
# =====================================================================

# %% [SEL 0] Setup
# !pip -q install timm
import os, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = "/content/drive/MyDrive/Proposal_Topik_Autism/DATASET/cache/"
SEED = 42

# %% [SEL A] Ekstraksi fitur beku (jalankan sekali)
def extract_features(out_dir=OUT_DIR, batch=32):
    import torch, torch.nn as nn, torchvision, timm
    f_res, f_swin = out_dir + "feats_resnet50.npy", out_dir + "feats_swin_t.npy"
    if os.path.exists(f_res) and os.path.exists(f_swin):
        print("Fitur sudah ada, dilewati."); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = np.load(out_dir + "cache_3slice.npz")["images"]            # (N,3,224,224) float16 [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    resnet.fc = nn.Identity()                                          # keluaran 2048-dim
    swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)  # 768-dim
    for name, model, path in [("ResNet-50", resnet, f_res), ("Swin-T", swin, f_swin)]:
        model = model.to(dev).eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                x = torch.from_numpy(imgs[i:i + batch].astype(np.float32)).to(dev)
                x = (x - mean) / std
                outs.append(model(x).float().cpu().numpy())
        feats = np.concatenate(outs)
        np.save(path, feats)
        print(f"{name}: {feats.shape} | mean {feats.mean():.3f} std {feats.std():.3f}")
# extract_features()        # <- jalankan baris ini sekali di Colab


# %% [SEL B] Baseline sklearn
def load_all(out_dir=OUT_DIR, use_feats=True):
    z = np.load(out_dir + "cache_3slice.npz")
    folds = pd.read_csv(out_dir + "folds.csv").set_index("sub_id")["fold"]
    D = dict(y=z["y"], clin=z["clin"].copy(), site=z["site"], fold=folds.loc[z["sub_id"]].to_numpy())
    _a = D["clin"][:, 0].astype(np.float64); _a[_a > 64] /= 10.0; D["clin"][:, 0] = _a   # koreksi umur (x10)
    if use_feats:
        D["resnet"] = np.load(out_dir + "feats_resnet50.npy")
        D["swin"] = np.load(out_dir + "feats_swin_t.npy")
    return D

CLIN8 = list(range(8))                  # urutan: AGE,SEX,HAND,FIQ,VIQ,PIQ,BMI,EYE
CLIN7 = [0, 1, 2, 3, 4, 5, 7]           # tanpa BMI (missing ~91%)

# --- pembangun fitur: SEMUA statistik di-fit hanya pada train fold ---
def clin_block(D, cols):
    def f(tr, te):
        X = D["clin"][:, cols]
        imp = SimpleImputer(strategy="median", add_indicator=True)     # indikator missing (tanpa label)
        sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
        return Xtr, sc.transform(imp.transform(X[te]))
    return f

def site_block(D):
    def f(tr, te):
        S = D["site"].reshape(-1, 1)
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return oh.fit_transform(S[tr]), oh.transform(S[te])
    return f

def img_block(F, n_pca=32):
    def f(tr, te):
        sc = StandardScaler(); pca = PCA(n_components=n_pca, random_state=SEED)
        Xtr = pca.fit_transform(sc.fit_transform(F[tr]))
        Xte = pca.transform(sc.transform(F[te]))
        s = Xtr.std(0).mean()                                          # samakan skala dgn fitur klinis
        return Xtr / s, Xte / s
    return f

def fusion_block(img_f, clin_f, w_clin=1.0):
    def f(tr, te):
        a_tr, a_te = img_f(tr, te); c_tr, c_te = clin_f(tr, te)
        return np.hstack([a_tr, w_clin * c_tr]), np.hstack([a_te, w_clin * c_te])
    return f

SVM = lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED)
LR = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
DUMMY = lambda: DummyClassifier(strategy="most_frequent")

def run_cv(D, name, block, clf_factory):
    y, fold, rows = D["y"], D["fold"], []
    for k in sorted(np.unique(fold)):
        te, tr = fold == k, fold != k
        Xtr, Xte = block(tr, te)
        clf = clf_factory().fit(Xtr, y[tr])
        score = clf.decision_function(Xte) if hasattr(clf, "decision_function") else np.zeros(te.sum())
        pred = clf.predict(Xte)
        rows.append(dict(model=name, fold=k,
                         acc=accuracy_score(y[te], pred),
                         recall=recall_score(y[te], pred, zero_division=0),
                         precision=precision_score(y[te], pred, zero_division=0),
                         f1=f1_score(y[te], pred, zero_division=0),
                         auc=roc_auc_score(y[te], score) if len(np.unique(score)) > 1 else 0.5))
    return pd.DataFrame(rows)

def run_all(D):
    exps = [
        ("0 Mayoritas (batas tebakan)",        lambda tr, te: (np.zeros((tr.sum(), 1)), np.zeros((te.sum(), 1))), DUMMY),
        ("1 Situs saja (LogReg)",              site_block(D), LR),
        ("2 Klinis 7 fitur (LogReg)",          clin_block(D, CLIN7), LR),
        ("3 Klinis 7 fitur (SVM)",             clin_block(D, CLIN7), SVM),
        ("4 Klinis 8 fitur +BMI (LogReg)",     clin_block(D, CLIN8), LR),
        ("5 Citra ResNet-50 (SVM)",            img_block(D["resnet"]), SVM),
        ("6 Citra Swin-T (SVM)",               img_block(D["swin"]), SVM),
        ("7 Fusi ResNet-50 + klinis7 (w=1)",   fusion_block(img_block(D["resnet"]), clin_block(D, CLIN7), 1.0), SVM),
        ("8 Fusi ResNet-50 + klinis7 (w=3)",   fusion_block(img_block(D["resnet"]), clin_block(D, CLIN7), 3.0), SVM),
        ("9 Fusi Swin-T + klinis7 (w=1)",      fusion_block(img_block(D["swin"]), clin_block(D, CLIN7), 1.0), SVM),
        ("10 Fusi Swin-T + klinis7 (w=3)",     fusion_block(img_block(D["swin"]), clin_block(D, CLIN7), 3.0), SVM),
    ]
    per_fold = pd.concat([run_cv(D, n, b, c) for n, b, c in exps], ignore_index=True)
    agg = per_fold.groupby("model", sort=False)[["acc", "recall", "precision", "f1", "auc"]].agg(["mean", "std"])
    table = pd.DataFrame({m: agg[m]["mean"].round(3).astype(str) + " ± " + agg[m]["std"].round(3).astype(str)
                          for m in ["acc", "recall", "precision", "f1", "auc"]})
    return per_fold, table

# --- Jalankan di Colab ---
# D = load_all()
# print("Batas tebakan (selalu 'TD'):", round(1 - D["y"].mean(), 3))
# per_fold, table = run_all(D)
# pd.set_option("display.width", 200); print(table.to_string())
# per_fold.to_csv(OUT_DIR + "hasil_hari2_per_fold.csv", index=False)
# table.to_csv(OUT_DIR + "hasil_hari2_ringkas.csv")
