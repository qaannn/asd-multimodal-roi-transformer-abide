"""Harmonisasi efek situs (ComBat parametrik, Johnson dkk. 2007) yang aman dari kebocoran:
dipelajari HANYA dari data latih, lalu diterapkan ke data lain. Label diagnosis TIDAK dipakai sebagai kovariat."""
import numpy as np


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def matrices_to_features(M, iu=None):
    """(n,R,R) korelasi -> (n,p) fitur segitiga atas dalam ruang Fisher-z."""
    iu = iu if iu is not None else np.triu_indices(M.shape[-1], k=1)
    return fisher_z(M[:, iu[0], iu[1]].astype(np.float32)).astype(np.float32)


def features_to_matrices(F, R, iu=None):
    """(n,p) Fisher-z -> (n,R,R) korelasi simetris (diagonal 0; diagonal memang dibuang oleh model)."""
    iu = iu if iu is not None else np.triu_indices(R, k=1)
    r = np.tanh(F).astype(np.float32)
    M = np.zeros((len(F), R, R), np.float32)
    M[:, iu[0], iu[1]] = r
    M[:, iu[1], iu[0]] = r
    return M


class ComBat:
    """X: (n,p); batch: label situs (n,); covars: (n,k) kovariat yang DIPERTAHANKAN (mis. umur, jenis kelamin)."""

    def __init__(self, eb=True, max_iter=100, tol=1e-4):
        self.eb, self.max_iter, self.tol = eb, max_iter, tol

    def _design(self, batch, covars):
        idx = np.array([self.index_.get(b, -1) for b in batch])
        B = np.zeros((len(batch), len(self.batches_)), np.float64)
        ok = idx >= 0
        B[np.arange(len(batch))[ok], idx[ok]] = 1.0
        return idx, B, (np.zeros((len(batch), 0)) if covars is None else (np.asarray(covars, np.float64) - self.cmu_) / self.csd_)

    def fit(self, X, batch, covars=None):
        X = np.asarray(X, np.float32); batch = np.asarray(batch).astype(str)
        self.batches_ = sorted(set(batch)); self.index_ = {b: i for i, b in enumerate(self.batches_)}
        if covars is not None:
            c = np.asarray(covars, np.float64); self.cmu_, self.csd_ = c.mean(0), c.std(0) + 1e-8
        idx, B, C = self._design(batch, covars)
        n, p = X.shape; m = len(self.batches_); k = C.shape[1]
        D = np.hstack([B, C])
        beta = np.linalg.lstsq(D, X.astype(np.float64), rcond=None)[0]                  # (m+k, p)
        n_i = B.sum(0)
        self.alpha_ = (n_i / n) @ beta[:m]
        self.beta_cov_ = beta[m:]
        resid = X - (D @ beta).astype(np.float32)
        self.sigma_ = np.sqrt((resid.astype(np.float64) ** 2).mean(0)).clip(1e-6)
        Z = ((X - (self.alpha_ + C @ self.beta_cov_)) / self.sigma_).astype(np.float32)
        gam = np.stack([Z[idx == i].mean(0, dtype=np.float64) for i in range(m)])
        dl2 = np.stack([Z[idx == i].var(0, ddof=1, dtype=np.float64) if (idx == i).sum() > 1 else np.ones(p) for i in range(m)])
        if self.eb:
            gam, dl2 = self._empirical_bayes(Z, idx, gam, dl2)
        self.gamma_, self.delta_ = gam, np.sqrt(dl2.clip(1e-8))
        return self

    def _empirical_bayes(self, Z, idx, gam_hat, dl2_hat):
        m = gam_hat.shape[0]
        with np.errstate(all="ignore"):
            gbar, tau2 = gam_hat.mean(1), gam_hat.var(1, ddof=1)
            md, vd = dl2_hat.mean(1), dl2_hat.var(1, ddof=1)
            lam = (2 * vd + md ** 2) / vd
            theta = (md * vd + md ** 3) / vd
        gam_s, dl2_s = np.empty_like(gam_hat), np.empty_like(dl2_hat)
        for i in range(m):
            Zi = Z[idx == i].astype(np.float64); n_i = len(Zi)
            # situs sangat kecil / prior degenerat (mis. n_i=1 -> varians seragam 1): pakai estimasi tanpa EB agar tidak NaN
            if (n_i < 3 or not np.isfinite([lam[i], theta[i], tau2[i], gbar[i]]).all() or tau2[i] <= 1e-12 or vd[i] <= 1e-12):
                gam_s[i], dl2_s[i] = gam_hat[i], dl2_hat[i]
                continue
            g_old, d_old = gam_hat[i].copy(), dl2_hat[i].copy()
            for _ in range(self.max_iter):
                g_new = (n_i * tau2[i] * gam_hat[i] + d_old * gbar[i]) / (n_i * tau2[i] + d_old)
                d_new = (theta[i] + 0.5 * ((Zi - g_new) ** 2).sum(0)) / (n_i / 2 + lam[i] - 1)
                change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                             np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)))
                g_old, d_old = g_new, d_new
                if change < self.tol:
                    break
            gam_s[i], dl2_s[i] = g_old, d_old
        return gam_s, dl2_s

    def transform(self, X, batch, covars=None):
        X = np.asarray(X, np.float32); batch = np.asarray(batch).astype(str)
        idx, B, C = self._design(batch, covars)
        m = len(self.batches_)
        if (idx < 0).any():
            print(f"  [ComBat] {int((idx < 0).sum())} subjek dari situs tak dikenal: hanya dinormalkan umum (tanpa koreksi situs)")
        gam = np.vstack([self.gamma_, np.zeros((1, X.shape[1]))])
        dlt = np.vstack([self.delta_, np.ones((1, X.shape[1]))])
        idx = np.where(idx < 0, m, idx)
        mean = self.alpha_ + C @ self.beta_cov_
        Z = ((X - mean) / self.sigma_ - gam[idx]) / dlt[idx]
        return (Z * self.sigma_ + mean).astype(np.float32)

    def fit_transform(self, X, batch, covars=None):
        return self.fit(X, batch, covars).transform(X, batch, covars)


def site_labels_for_combat(site, fit_idx, min_n=5):
    """Situs dengan < min_n subjek di data latih (atau yang tidak ada sama sekali di data latih) digabung ke batch 'RARE'.
    Aturan dihitung dari data latih saja, lalu diterapkan sama ke seluruh subjek."""
    site = np.asarray(site).astype(str)
    s_fit, c = np.unique(site[fit_idx], return_counts=True)
    kecil = set(s_fit[c < min_n]) | (set(site) - set(s_fit))
    return np.where(np.isin(site, list(kecil)), "RARE", site)


def harmonize_matrices(D, fit_idx, cov_cols=(0, 1)):
    """Matriks korelasi seluruh subjek setelah ComBat yang DIPELAJARI hanya dari subjek fit_idx.
    Kovariat dipertahankan: umur & jenis kelamin (kolom clin 0 dan 1). Tanpa label diagnosis."""
    M = D["M"]; R = M.shape[-1]; iu = np.triu_indices(R, k=1)
    F = matrices_to_features(M, iu)
    cov = np.asarray(D["clin"][:, list(cov_cols)], np.float64)
    med = np.nanmedian(cov[fit_idx], axis=0)
    cov = np.where(np.isnan(cov), med, cov)                                    # imputasi kovariat dari data latih
    batch = site_labels_for_combat(D["site"], fit_idx)
    cb = ComBat().fit(F[fit_idx], batch[fit_idx], cov[fit_idx])
    return features_to_matrices(cb.transform(F, batch, cov), R, iu)
