"""Baseline DL lama (proposal): ResNet-50 ImageNet BEKU sebagai ekstraktor fitur dari matriks konektivitas + SVM.
Matriks korelasi (R,R) -> gambar 224x224 tiga kanal (diagonal 0, korelasi dipotong ±0,8 lalu diskalakan ke [0,1])."""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import CACHE_DIR


def extract_resnet_features(M, batch=32, clip=0.8):
    import torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2"); model.fc = nn.Identity()
    model = model.to(dev).eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(M), batch):
            m = torch.from_numpy(M[i:i + batch]).to(dev).clone()
            m.diagonal(dim1=-2, dim2=-1).zero_()
            x = (m.clamp(-clip, clip) / clip + 1) / 2
            x = Fn.interpolate(x.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
            outs.append(model((x.repeat(1, 3, 1, 1) - mean) / std).float().cpu().numpy())
    return np.concatenate(outs)


def get_features(D, tag):
    """Fitur beku di-cache per dataset (tidak bergantung pada fold, karena tidak ada pelatihan)."""
    path = CACHE_DIR + f"feats_resnet_{tag}_{D['M'].shape[0]}x{D['M'].shape[-1]}.npy"
    if os.path.exists(path):
        return np.load(path)
    F = extract_resnet_features(D["M"])
    np.save(path, F)
    return F


def resnet_block(F):
    def f(tr, te):
        sc = StandardScaler(); return sc.fit_transform(F[tr]), sc.transform(F[te])
    return f
