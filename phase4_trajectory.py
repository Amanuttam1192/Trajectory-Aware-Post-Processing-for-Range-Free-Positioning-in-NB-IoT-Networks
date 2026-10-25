"""
Phase 4 (final) - Trajectory-aware post-processing on top of WkNN.

Method: run their Strategy I WkNN, then apply a centered rolling MEDIAN
(window W) to the estimated positions along each campaign's drive order,
in local metric coordinates. Uses ONLY estimated positions and the
measurement sequence order -- no ground truth, so it is deployable.

Rows in the dataset are stored in drive order within each campaign
(verified: consecutive-row ground-truth distance is 2-5 m).

Evaluated under BOTH protocols:
  A) their pre-generated random 70/30 splits  -> directly comparable to
     Table III of the J-ISPIN 2025 paper (WkNN: Oslo 15.67 m, Rome 13.12 m)
  B) leave-one-campaign-out (honest, leakage-free)

Requires loader.py, phase1_baseline.py, phase3_xgboost.py in the same folder.
Set REPO to your extracted repo path.
"""
import numpy as np
from loader import load_city, load_splits, haversine_m
from phase1_baseline import wknn_predict
from phase3_xgboost import to_local_xy, to_latlon

REPO = r'C:\Users\amanu\Desktop\Positioning Research\Positioning-NBIoT-Beyond-WkNN-main'
WINDOW = 5
N_RUNS_A = 20


def rolling_median(a, w):
    n = len(a)
    out = np.empty_like(a, dtype=float)
    h = w // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = np.median(a[lo:hi])
    return out


def smooth_by_campaign(est_latlon, camp_of_points, lat0, lon0, w=WINDOW):
    exy = to_local_xy(est_latlon, lat0, lon0)
    sxy = exy.copy()
    for c in np.unique(camp_of_points):
        m = np.where(camp_of_points == c)[0]      # drive order within campaign
        sxy[m, 0] = rolling_median(exy[m, 0], w)
        sxy[m, 1] = rolling_median(exy[m, 1], w)
    return to_latlon(sxy, lat0, lon0)


def stats_line(tag, e):
    print('  %-14s mean %6.2f  std %6.2f  med %5.2f  p95 %6.2f' %
          (tag, e.mean(), e.std(), np.median(e), np.percentile(e, 95)))


def run_city(city):
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()

    # ---- Protocol A: their random splits (Table III comparison) ----
    alloc = list(load_splits(f'{REPO}/{city}/TP_RP_allocation_{city}.mat').values())[0]
    rp_label = 1 if (alloc[:, 0] == 1).mean() > 0.5 else 2
    raw_all, smo_all = [], []
    for r in range(N_RUNS_A):
        col = alloc[:, r]
        tr = np.where(col == rp_label)[0]
        te = np.where(col != rp_label)[0]
        est = wknn_predict(M, valid, latlon, tr, te)
        smo = smooth_by_campaign(est, camp[te], lat0, lon0)
        gt = latlon[te]
        raw_all.append(haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1]))
        smo_all.append(haversine_m(gt[:, 0], gt[:, 1], smo[:, 0], smo[:, 1]))
    raw, smo = np.concatenate(raw_all), np.concatenate(smo_all)
    print(f'===== {city} | A: random split ({N_RUNS_A} runs, W={WINDOW}) =====')
    stats_line('WkNN (paper)', raw)
    stats_line('+ smoothing', smo)
    print('  mean improvement: %.1f%%   p95: %.1f%%' % (
        100 * (1 - smo.mean() / raw.mean()),
        100 * (1 - np.percentile(smo, 95) / np.percentile(raw, 95))))
    np.save(f'phase4_random_raw_{city}.npy', raw)
    np.save(f'phase4_random_smoothed_{city}.npy', smo)

    # ---- Protocol B: LOCO ----
    raw_all, smo_all = [], []
    for c in np.unique(camp):
        te = np.where(camp == c)[0]
        tr = np.where(camp != c)[0]
        est = wknn_predict(M, valid, latlon, tr, te)
        smo = smooth_by_campaign(est, camp[te], lat0, lon0)
        gt = latlon[te]
        raw_all.append(haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1]))
        smo_all.append(haversine_m(gt[:, 0], gt[:, 1], smo[:, 0], smo[:, 1]))
    raw, smo = np.concatenate(raw_all), np.concatenate(smo_all)
    print(f'===== {city} | B: leave-one-campaign-out =====')
    stats_line('WkNN', raw)
    stats_line('+ smoothing', smo)
    np.save(f'phase4_loco_smoothed_{city}.npy', smo)


if __name__ == '__main__':
    for city in ['Oslo', 'Rome']:
        run_city(city)
