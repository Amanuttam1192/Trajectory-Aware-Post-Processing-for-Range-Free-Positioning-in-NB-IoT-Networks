"""
Does the trajectory-smoothing layer help estimators OTHER than WkNN?
Tests XGBoost with and without the same rolling-median correction,
Protocol A, Oslo and Rome.
"""
import numpy as np
from loader import load_city, load_splits, haversine_m
from phase1_baseline import wknn_predict
from phase3_xgboost import build_features, xgb_predict, to_local_xy, to_latlon
from phase4_trajectory import rolling_median

REPO = '/home/claude/Positioning-NBIoT-Beyond-WkNN'
W = 5
N_RUNS = 3


def smooth(est, camp_te, lat0, lon0, w=W):
    exy = to_local_xy(est, lat0, lon0)
    sxy = exy.copy()
    for c in np.unique(camp_te):
        m = np.where(camp_te == c)[0]
        sxy[m, 0] = rolling_median(exy[m, 0], w)
        sxy[m, 1] = rolling_median(exy[m, 1], w)
    return to_latlon(sxy, lat0, lon0)


def st(e):
    return (e.mean(), e.std(), np.median(e), np.percentile(e, 95))


for city in ['Oslo', 'Rome']:
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    X = build_features(M, valid)
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()
    alloc = list(load_splits(f'{REPO}/{city}/TP_RP_allocation_{city}.mat').values())[0]
    rp = 1 if (alloc[:, 0] == 1).mean() > 0.5 else 2

    acc = {k: [] for k in ['w_raw', 'w_smo', 'x_raw', 'x_smo']}
    for r in range(N_RUNS):
        col = alloc[:, r]
        tr = np.where(col == rp)[0]
        te = np.where(col != rp)[0]
        gt = latlon[te]
        ct = camp[te]

        ew = wknn_predict(M, valid, latlon, tr, te)
        ex = xgb_predict(X, latlon, tr, te, lat0, lon0)
        for tag, est in [('w', ew), ('x', ex)]:
            sm = smooth(est, ct, lat0, lon0)
            acc[f'{tag}_raw'].append(haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1]))
            acc[f'{tag}_smo'].append(haversine_m(gt[:, 0], gt[:, 1], sm[:, 0], sm[:, 1]))

    print(f'===== {city} (Protocol A, {N_RUNS} runs, W={W}) =====')
    for name, k in [('WkNN            ', 'w_raw'), ('WkNN + smoothing', 'w_smo'),
                    ('XGBoost         ', 'x_raw'), ('XGBoost + smoothing', 'x_smo')]:
        e = np.concatenate(acc[k])
        print('  %-20s mean %7.2f  std %7.2f  med %6.2f  p95 %7.2f' % ((name,) + st(e)))
    wr = np.concatenate(acc['w_raw']).mean(); ws = np.concatenate(acc['w_smo']).mean()
    xr = np.concatenate(acc['x_raw']).mean(); xs = np.concatenate(acc['x_smo']).mean()
    print('  -> smoothing gain:  WkNN %.1f%%   XGBoost %.1f%%' % (
        100 * (1 - ws / wr), 100 * (1 - xs / xr)))
