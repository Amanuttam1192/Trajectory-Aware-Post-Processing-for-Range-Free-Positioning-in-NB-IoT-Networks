"""
Phase 3 - Gradient-boosted regression (XGBoost) for positioning.

Model: two XGBRegressors predicting local metric coordinates (x_east, y_north)
from the fingerprint. Features = SINR vector (fNull = -25) + engineered
extras: number of detected cells, max SINR, mean/std of valid SINRs.

Evaluated under BOTH protocols:
  A) their random 70/30 splits (first N_RANDOM of the pre-generated 100)
  B) leave-one-campaign-out (LOCO)
and compared against the Phase 1/2 WkNN results.

Set REPO to your extracted repo folder before running.
"""
import numpy as np
from xgboost import XGBRegressor
from loader import load_city, load_splits, haversine_m
from phase1_baseline import wknn_predict

REPO = r'C:\Users\amanu\Desktop\Positioning Research\Positioning-NBIoT-Beyond-WkNN-main'   # <-- EDIT ON YOUR PC
N_RANDOM = 5          # random-split runs for XGB (training is costly)
F_NULL = -25.0

XGB_PARAMS = dict(
    n_estimators=600, max_depth=8, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
    tree_method='hist', n_jobs=-1, random_state=0)


def to_local_xy(latlon, lat0, lon0):
    """Equirectangular projection to meters around a city-centre anchor."""
    R = 6371000.0
    x = np.radians(latlon[:, 1] - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(latlon[:, 0] - lat0) * R
    return np.column_stack([x, y])


def to_latlon(xy, lat0, lon0):
    R = 6371000.0
    lon = lon0 + np.degrees(xy[:, 0] / (R * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(xy[:, 1] / R)
    return np.column_stack([lat, lon])


def build_features(M, valid):
    n_det = valid.sum(1, keepdims=True).astype(float)
    Mv = np.where(valid, M, np.nan)
    with np.errstate(all='ignore'):
        mx = np.nanmax(Mv, axis=1, keepdims=True)
        mn = np.nanmean(Mv, axis=1, keepdims=True)
        sd = np.nanstd(Mv, axis=1, keepdims=True)
    extra = np.nan_to_num(np.hstack([n_det, mx, mn, sd]), nan=F_NULL)
    return np.hstack([M, extra]).astype(np.float32)


def xgb_predict(X, latlon, tr_idx, te_idx, lat0, lon0):
    Y = to_local_xy(latlon, lat0, lon0)
    est_xy = np.zeros((len(te_idx), 2))
    for c in range(2):
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X[tr_idx], Y[tr_idx, c])
        est_xy[:, c] = model.predict(X[te_idx])
    return to_latlon(est_xy, lat0, lon0)


def stats(err):
    return (err.mean(), err.std(), np.median(err), np.percentile(err, 95))


def run_city(city):
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    X = build_features(M, valid)
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()

    print(f'===== {city} =====')
    # ---- Protocol A: their random splits ----
    alloc = list(load_splits(f'{REPO}/{city}/TP_RP_allocation_{city}.mat').values())[0]
    rp_label = 1 if (alloc[:, 0] == 1).mean() > 0.5 else 2
    ew_all, ex_all = [], []
    for r in range(N_RANDOM):
        col = alloc[:, r]
        tr = np.where(col == rp_label)[0]
        te = np.where(col != rp_label)[0]
        gt = latlon[te]
        est_w = wknn_predict(M, valid, latlon, tr, te)
        est_x = xgb_predict(X, latlon, tr, te, lat0, lon0)
        ew_all.append(haversine_m(gt[:, 0], gt[:, 1], est_w[:, 0], est_w[:, 1]))
        ex_all.append(haversine_m(gt[:, 0], gt[:, 1], est_x[:, 0], est_x[:, 1]))
    ew, ex = np.concatenate(ew_all), np.concatenate(ex_all)
    print(f'A random-split ({N_RANDOM} runs):')
    print('  WkNN    mean %6.2f  std %6.2f  med %6.2f  p95 %7.2f' % stats(ew))
    print('  XGBoost mean %6.2f  std %6.2f  med %6.2f  p95 %7.2f' % stats(ex))

    # ---- Protocol B: LOCO ----
    ew_all, ex_all = [], []
    for c in np.unique(camp):
        te = np.where(camp == c)[0]
        tr = np.where(camp != c)[0]
        gt = latlon[te]
        est_w = wknn_predict(M, valid, latlon, tr, te)
        est_x = xgb_predict(X, latlon, tr, te, lat0, lon0)
        ew_all.append(haversine_m(gt[:, 0], gt[:, 1], est_w[:, 0], est_w[:, 1]))
        ex_all.append(haversine_m(gt[:, 0], gt[:, 1], est_x[:, 0], est_x[:, 1]))
    ew, ex = np.concatenate(ew_all), np.concatenate(ex_all)
    print('B leave-one-campaign-out:')
    print('  WkNN    mean %6.2f  std %6.2f  med %6.2f  p95 %7.2f' % stats(ew))
    print('  XGBoost mean %6.2f  std %6.2f  med %6.2f  p95 %7.2f' % stats(ex))
    np.save(f'phase3_xgb_loco_err_{city}.npy', ex)


if __name__ == '__main__':
    for city in ['Oslo', 'Rome']:
        run_city(city)
