"""Arsitektur: ROI-Transformer (1 token per ROI) + fusi klinis (none | concat | cross-attention)."""
import torch
import torch.nn as nn

class ROITokenizer(nn.Module):
    """200 wilayah otak -> 200 token. Token ROI ke-i = proyeksi baris ke-i matriks + embedding identitas ROI."""
    def __init__(self, n_roi=200, d_model=64, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_roi, d_model), nn.LayerNorm(d_model))
        self.roi_id = nn.Parameter(torch.zeros(1, n_roi, d_model))
        nn.init.trunc_normal_(self.roi_id, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, M):
        eye = torch.eye(M.shape[-1], device=M.device, dtype=torch.bool)
        M = M.masked_fill(eye, 0.0)                       # buang korelasi dengan diri sendiri
        return self.drop(self.proj(M) + self.roi_id)

class ClinicalTokenizer(nn.Module):
    """7 fitur klinis -> 7 token (plus penanda 'nilai hilang')."""
    def __init__(self, n_features=7, d_model=64):
        super().__init__()
        bound = d_model ** -0.5
        self.weight = nn.Parameter(torch.empty(n_features, d_model).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_features, d_model).uniform_(-bound, bound))
        self.miss_embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)

    def forward(self, x, miss=None):
        tok = x.unsqueeze(-1) * self.weight + self.bias
        if miss is not None:
            tok = tok + miss.unsqueeze(-1) * self.miss_embed
        return tok

class CrossAttentionBlock(nn.Module):
    """Token klinis (query) 'menyorot' token ROI (key/value) yang relevan."""
    def __init__(self, d_model=64, n_heads=4, mlp_ratio=2.0, dropout=0.2):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        h = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(h, d_model), nn.Dropout(dropout))

    def forward(self, q, kv, return_attn=False):
        kvn = self.norm_kv(kv)
        a, w = self.attn(self.norm_q(q), kvn, kvn, need_weights=return_attn, average_attn_weights=True)
        q = q + a
        q = q + self.ff(self.norm_ff(q))
        return q, w                                        # w: (B, 7, 200) bila return_attn

class ROIFusionNet(nn.Module):
    """fusion: 'none' (otak saja) | 'concat' (gabung biasa) | 'cross' (cross-attention, usulan utama)."""
    def __init__(self, fusion="cross", n_roi=200, n_clin=7, d_model=64, n_heads=4,
                 n_enc_layers=2, n_cross_layers=1, dropout=0.2, head_dropout=0.3, n_classes=2):
        super().__init__()
        assert fusion in ("none", "concat", "cross")
        self.fusion = fusion
        self.tok = ROITokenizer(n_roi, d_model, dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        if fusion == "cross":
            self.clin_tok = ClinicalTokenizer(n_clin, d_model)
            self.cross = nn.ModuleList([CrossAttentionBlock(d_model, n_heads, dropout=dropout)
                                        for _ in range(n_cross_layers)])
            self.norm_out = nn.LayerNorm(d_model)
            head_in = 2 * d_model
        elif fusion == "concat":
            self.clin_mlp = nn.Sequential(nn.Linear(2 * n_clin, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            head_in = 2 * d_model
        else:
            head_in = d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(head_dropout),
                                  nn.Linear(d_model, n_classes))

    def forward(self, M, clin=None, miss=None, return_attn=False):
        h = self.norm(self.encoder(self.tok(M)))           # (B,200,d)
        z_roi = h.mean(1)
        attn = None
        if self.fusion == "cross":
            q = self.clin_tok(clin, miss)
            for blk in self.cross:
                q, attn = blk(q, h, return_attn)
            z = torch.cat([self.norm_out(q).mean(1), z_roi], dim=1)
        elif self.fusion == "concat":
            z = torch.cat([self.clin_mlp(torch.cat([clin, miss], dim=1)), z_roi], dim=1)
        else:
            z = z_roi
        logits = self.head(z)
        return (logits, attn) if return_attn else logits
