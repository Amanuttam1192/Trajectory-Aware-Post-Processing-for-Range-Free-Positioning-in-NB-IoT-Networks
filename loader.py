"""
Phase 0 - Data loader for the De Nardis et al. NB-IoT positioning dataset.
Converts the MATLAB workspace (Oslo_NBIoT_data.mat / Rome_NBIoT_data.mat)
into numpy structures identical to those used by their Strategy I code:

  M      : (N_points, N_features) SINR matrix, missing entries = -25 dB
  valid  : (N_points, N_features) boolean mask of detected features
  latlon : (N_points, 2)
  camp   : (N_points,) campaign/run ID of each point (for leakage-free eval)

Feature identity = unique triplet (NPCI, eNodeB-ID, operatorID),
matching columns [1 2 8] of each per-point cell matrix in dataSet(:,3).
"""
import numpy as np
import scipy.io as sio

F_NULL = -25.0          # missing-feature value for SINR (their setting)
RF_COL = 3              # 0-based col in per-point matrix: 2=RSSI,3=SINR,4=RSRP,5=RSRQ
ID_COLS = (0, 1, 7)     # NPCI, eNodeB-ID, operator ID (0-based)


def load_city(mat_path):
    m = sio.loadmat(mat_path)
    ds = m['dataSet']
    n = ds.shape[0]

    latlon = np.array([[ds[i, 0].item(), ds[i, 1].item()] for i in range(n)])

    # Build the global feature vocabulary (uniqueNPCIs)
    triplets = []
    for i in range(n):
        cellmat = np.atleast_2d(np.array(ds[i, 2], dtype=float))
        triplets.append(cellmat[:, ID_COLS])
    allt = np.vstack(triplets)
    uniq = np.unique(allt, axis=0)
    index = {tuple(row): j for j, row in enumerate(uniq)}
    nf = len(uniq)

    M = np.full((n, nf), F_NULL)
    valid = np.zeros((n, nf), dtype=bool)
    for i in range(n):
        cellmat = np.atleast_2d(np.array(ds[i, 2], dtype=float))
        for r in range(cellmat.shape[0]):
            j = index[tuple(cellmat[r, ID_COLS])]
            M[i, j] = cellmat[r, RF_COL]
            valid[i, j] = True

    # campaign id: last column of dataSet (col 11 in Oslo, col 3 in Rome)
    last = ds.shape[1] - 1
    camp = np.array([int(np.array(ds[i, last]).item()) for i in range(n)])

    # per-point timestamp (col 1 of the cell matrix looks like seconds); take min
    tstamp = np.array([np.atleast_2d(np.array(ds[i, 2], dtype=float))[:, 1].min()
                       for i in range(n)])
    return M, valid, latlon, camp, tstamp


def load_splits(alloc_path):
    """Their pre-generated 100-run TP/RP allocations (1-based MATLAB indices)."""
    a = sio.loadmat(alloc_path)
    keys = [k for k in a.keys() if not k.startswith('__')]
    return {k: a[k] for k in keys}


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


if __name__ == '__main__':
    import sys
    REPO = r'C:\Users\amanu\Desktop\Positioning Research\Positioning-NBIoT-Beyond-WkNN-main'
    for city in ['Oslo', 'Rome']:
        M, valid, latlon, camp, ts = load_city(f'{repo}/{city}/{city}_NBIoT_data.mat')
        print(f'{city}: {M.shape[0]} points, {M.shape[1]} features '
              f'(paper: Oslo 5266/399ish, Rome 2670/237ish)')
        print(f'  campaigns: {np.unique(camp, return_counts=True)}')
        print(f'  mean detected cells/point: {valid.sum(1).mean():.1f}')
