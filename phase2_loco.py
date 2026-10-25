"""
Phase 2 - Leakage-free evaluation: Leave-One-Campaign-Out (LOCO).

Their random 70/30 split places points from the same drive run in both
train and test; they admit this gives a positive bias 'within 7%'.
Here the SAME WkNN algorithm is evaluated with entire campaigns held out:
RPs = all points from the other campaigns, TPs = the held-out campaign.
This quantifies the true bias and sets the honest baseline that Phases
3-4 must beat.
"""
import numpy as np
from loader import load_city, haversine_m
from phase1_baseline import wknn_predict

REPO = r'C:\Users\amanu\Desktop\Positioning Research\Positioning-NBIoT-Beyond-WkNN-main'


def run_city(city):
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    errs = []
    print(f'--- {city} ---')
    for c in np.unique(camp):
        tp_idx = np.where(camp == c)[0]
        rp_idx = np.where(camp != c)[0]
        est = wknn_predict(M, valid, latlon, rp_idx, tp_idx)
        gt = latlon[tp_idx]
        e = haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1])
        errs.append(e)
        print(f'  hold out campaign {c} ({len(tp_idx):4d} pts): '
              f'mean {e.mean():6.2f} m  median {np.median(e):6.2f} m')
    all_e = np.concatenate(errs)
    print(f'  LOCO overall: mean {all_e.mean():.2f} m  std {all_e.std():.2f} m  '
          f'median {np.median(all_e):.2f} m  p95 {np.percentile(all_e,95):.2f} m')
    np.save(f'phase2_err_{city}.npy', all_e)
    # also save per-point campaign order for the trajectory phase
    return all_e


if __name__ == '__main__':
    for city in ['Oslo', 'Rome']:
        run_city(city)
