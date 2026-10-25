"""
Window-size sensitivity of the trajectory smoothing layer.

Reproduces:
  - Table II of the paper (mean error vs W under Protocol A, 20 splits)
  - the Protocol B gains quoted in Section V-D (W = 5 and W = 15)

Requires loader.py, phase1_baseline.py, phase3_xgboost.py and
phase4_trajectory.py in the same folder.
"""
import numpy as np
from loader import load_city, load_splits, haversine_m
from phase1_baseline import wknn_predict
from phase4_trajectory import smooth_by_campaign

REPO = 'Positioning-NBIoT-Beyond-WkNN-main'
N_RUNS_A = 20
WINDOWS_A = [3, 5, 7]          # columns of Table II (W = 1 is the baseline)
WINDOWS_B = [5, 15]            # values quoted in Section V-D


def mean_err(gt, est):
    return haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1])


def run_city(city):
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()

    # ---- Protocol A: same 20 allocations as phase4_trajectory.py ----
    alloc = list(load_splits(f'{REPO}/{city}/TP_RP_allocation_{city}.mat').values())[0]
    rp_label = 1 if (alloc[:, 0] == 1).mean() > 0.5 else 2
    raw, smo = [], {w: [] for w in WINDOWS_A}
    for r in range(N_RUNS_A):
        col = alloc[:, r]
        tr, te = np.where(col == rp_label)[0], np.where(col != rp_label)[0]
        est = wknn_predict(M, valid, latlon, tr, te)
        raw.append(mean_err(latlon[te], est))
        for w in WINDOWS_A:
            smo[w].append(mean_err(latlon[te], smooth_by_campaign(est, camp[te], lat0, lon0, w)))
    base = np.concatenate(raw).mean()
    cells = [f'W=1: {base:.2f}'] + [f'W={w}: {np.concatenate(smo[w]).mean():.2f}'
                                    for w in WINDOWS_A]
    print(f'{city} | Protocol A (Table II): ' + '   '.join(cells))

    # ---- Protocol B: leave-one-campaign-out ----
    raw, smo = [], {w: [] for w in WINDOWS_B}
    for c in np.unique(camp):
        te, tr = np.where(camp == c)[0], np.where(camp != c)[0]
        est = wknn_predict(M, valid, latlon, tr, te)
        raw.append(mean_err(latlon[te], est))
        for w in WINDOWS_B:
            smo[w].append(mean_err(latlon[te], smooth_by_campaign(est, camp[te], lat0, lon0, w)))
    base = np.concatenate(raw).mean()
    gains = [f'W={w}: {100 * (1 - np.concatenate(smo[w]).mean() / base):.1f}%'
             for w in WINDOWS_B]
    print(f'{city} | Protocol B gain (Sec. V-D): ' + '   '.join(gains))


if __name__ == '__main__':
    for city in ['Oslo', 'Rome']:
        run_city(city)
