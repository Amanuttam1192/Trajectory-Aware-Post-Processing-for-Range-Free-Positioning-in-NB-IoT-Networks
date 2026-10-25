"""
Phase 1 - Reproduce Strategy I (their WkNN baseline, the accuracy leader).

Their setup: SINR features, fNull = -25 dB, no PCA, k = 2,
Weighted-Coverage metric  d(i,j) = sqrt(L2 over ALL features) / L_ij
where L_ij = number of valid features the two points share.
Estimate = weighted average of the k nearest RPs, weights = 1/d.
Evaluated over their pre-generated 70/30 splits (values: RP vs TP).

Target from Table III: Oslo 15.67 m mean / 21.35 m std,
                        Rome 13.12 m mean / 18.67 m std.
"""
import numpy as np
from loader import load_city, load_splits, haversine_m

REPO = r'C:\Users\amanu\Desktop\Positioning Research\Positioning-NBIoT-Beyond-WkNN-main'
K = 2
N_RUNS = 100


def wknn_predict(M, valid, latlon, rp_idx, tp_idx, k=K):
    A = M[tp_idx].astype(np.float32)          # (T, F)
    B = M[rp_idx].astype(np.float32)          # (R, F)
    # squared L2 over all features (missing = fNull, as in their code)
    D2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    np.maximum(D2, 0, out=D2)
    L = valid[tp_idx].astype(np.float32) @ valid[rp_idx].astype(np.float32).T
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.sqrt(D2) / L                    # weighted-coverage metric
    D[L == 0] = np.inf
    D += 1e-12                                 # avoid /0 in weights

    nn = np.argpartition(D, k, axis=1)[:, :k]
    rows = np.arange(D.shape[0])[:, None]
    dnn = D[rows, nn]                          # (T, k)
    w = 1.0 / dnn
    # fallback: a TP sharing no features with any RP -> uniform weights
    bad = ~np.isfinite(w)
    w[bad] = 0.0
    zero_rows = w.sum(1) == 0
    w[zero_rows] = 1.0
    P = latlon[rp_idx][nn]                     # (T, k, 2)
    est = (w[..., None] * P).sum(1) / w.sum(1, keepdims=True)
    return est


def run_city(city):
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    alloc = load_splits(f'{REPO}/{city}/TP_RP_allocation_{city}.mat')
    A = list(alloc.values())[0]                # (N, n_splits) with values {1,2}
    # figure out which label means RP (the ~70% one)
    frac1 = (A[:, 0] == 1).mean()
    rp_label = 1 if frac1 > 0.5 else 2

    means, stds, meds, p95s = [], [], [], []
    all_err = []
    for r in range(N_RUNS):
        col = A[:, r]
        rp_idx = np.where(col == rp_label)[0]
        tp_idx = np.where(col != rp_label)[0]
        est = wknn_predict(M, valid, latlon, rp_idx, tp_idx)
        gt = latlon[tp_idx]
        err = haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1])
        means.append(err.mean()); stds.append(err.std())
        meds.append(np.median(err)); p95s.append(np.percentile(err, 95))
        all_err.append(err)
    all_err = np.concatenate(all_err)
    print(f'{city}:  mean {np.mean(means):.2f} m  (paper Table III target)  '
          f'std {np.mean(stds):.2f} m')
    print(f'        median {np.mean(meds):.2f} m   p95 {np.mean(p95s):.2f} m   '
          f'runs {N_RUNS}, RP share {(A[:,0]==rp_label).mean():.2f}')
    np.save(f'phase1_err_{city}.npy', all_err)
    return np.mean(means), np.mean(stds)


if __name__ == '__main__':
    for city in ['Oslo', 'Rome']:
        run_city(city)
