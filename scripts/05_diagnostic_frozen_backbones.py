"""Diagnostik (arsip): fitur ResNet-50 / Swin-T beku (ImageNet) pada matriks konektivitas. Hasil: tidak informatif."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.config import CACHE_DIR, TABEL_DIR
from src.data import prepare_dataset
from src.baselines import run_cv, run_cv_late, clin_block, SVM_P


def extract_conn_feats(D, out=CACHE_DIR, batch=32, clip=0.8):
    import torch, torch.nn as nn, torch.nn.functional as Fn, torchvision, timm
    f_res, f_swin = out + "feats_conn_resnet50.npy", out + "feats_conn_swin_t.npy"
    if os.path.exists(f_res) and os.path.exists(f_swin):
        print("Fitur konektivitas sudah ada, dilewati."); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2"); resnet.fc = nn.Identity()
    swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
    M = D["M"]
    for name, model, path in [("ResNet-50", resnet, f_res), ("Swin-T", swin, f_swin)]:
        model = model.to(dev).eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(M), batch):
                m = torch.from_numpy(M[i:i + batch]).to(dev).clone()          # (B,200,200), korelasi [-1,1]
                m.diagonal(dim1=-2, dim2=-1).zero_()                           # diagonal = 1 dibuang
                x = (m.clamp(-clip, clip) / clip + 1) / 2                      # -> [0,1], 0.5 = korelasi nol
                x = Fn.interpolate(x.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
                x = (x.repeat(1, 3, 1, 1) - mean) / std
                outs.append(model(x).float().cpu().numpy())
        feats = np.concatenate(outs); np.save(path, feats)
        print(f"{name}: {feats.shape} | mean {feats.mean():.3f} std {feats.std():.3f}")


# %% [SEL 2] Evaluasi: fitur beku saja, dan late fusion dengan klinis
def feat_block(F):
    def f(tr, te):
        sc = StandardScaler(); return sc.fit_transform(F[tr]), sc.transform(F[te])
    return f

LR_MID = lambda: LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000)

def run_conn_deep(D, out=CACHE_DIR):
    runs = []
    for name, fn in [("ResNet-50", "feats_conn_resnet50.npy"), ("Swin-T", "feats_conn_swin_t.npy")]:
        fb = feat_block(np.load(out + fn))
        runs.append(run_cv(D, f"A {name} beku pada konektivitas (LogReg)", fb, LR_MID))
        runs.append(run_cv_late(D, f"B Late fusion {name} + klinis", fb, LR_MID, clin_block(D), SVM_P))
    per_fold = pd.concat(runs, ignore_index=True)
    agg = per_fold.groupby("model", sort=False)[["acc", "recall", "precision", "f1", "auc"]].agg(["mean", "std"])
    return pd.DataFrame({m: agg[m]["mean"].round(3).astype(str) + " ± " + agg[m]["std"].round(3).astype(str)
                         for m in ["acc", "recall", "precision", "f1", "auc"]})


def main():
    D = prepare_dataset()
    extract_conn_feats(D)                                   # butuh torch, torchvision, timm (GPU disarankan)
    table = run_conn_deep(D)
    print(table.to_string())
    table.to_csv(TABEL_DIR + "diagnostik_backbone_beku.csv")


if __name__ == "__main__":
    main()
