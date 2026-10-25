"""
=============================================================================
 make_all_figures.py  -  runs every experiment and saves each result as a
                         separate PNG image in a  figures/  folder.
=============================================================================

WHAT IT PRODUCES
  fig1_baseline_reproduction.png   published vs reproduced bar chart
  fig2_main_improvement.png        WkNN vs WkNN+smoothing (mean/median/p95)
  fig3_cdf_<City>.png              CDF per city, with mean lines marked
  fig4_cdf_all.png                 both cities side by side
  fig5_window_sweep.png            error vs smoothing window W
  fig6_estimator_comparison.png    WkNN vs XGBoost, each with/without smoothing
  fig7_error_reduction.png         % improvement summary
  fig8_protocol_comparison.png     Protocol A vs Protocol B (log scale)
  fig9_percentile_bars.png         where in the distribution the gain comes from
  fig10_trajectory_example.png     a real drive: raw vs smoothed estimates
  results_summary.csv              every number in one table

REQUIREMENTS
  loader.py, phase1_baseline.py, phase3_xgboost.py, phase4_trajectory.py
  in the same folder, plus:  pip install numpy scipy matplotlib xgboost

EDIT THIS BEFORE RUNNING  ->  REPO path below
=============================================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # save files instead of opening windows
import matplotlib.pyplot as plt

from loader import load_city, load_splits, haversine_m
from phase1_baseline import wknn_predict
from phase3_xgboost import build_features, xgb_predict, to_local_xy, to_latlon
from phase4_trajectory import rolling_median

# ----------------------------------------------------------------- SETTINGS
REPO = r'C:\Users\amanu\Desktop\Positioning Research\Positioning-NBIoT-Beyond-WkNN-main'
OUT = 'figures'
CITIES = ['Oslo', 'Rome']
W = 5                    # smoothing window
N_RUNS_FAST = 20         # splits for WkNN experiments (cheap)
N_RUNS_XGB = 3           # splits for XGBoost experiments (slow to train)
RUN_XGBOOST = True       # set False to skip the slow parts

# published values from Table III of the reference paper
PUBLISHED = {'Oslo': (15.67, 21.35), 'Rome': (13.12, 18.67)}

C_BASE = '#22458C'       # baseline colour
C_PROP = '#C0392B'       # proposed method colour
C_ALT = '#E08E45'        # alternative estimator colour

plt.rcParams.update({
    'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130, 'savefig.dpi': 200, 'savefig.bbox': 'tight'
})
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print('  saved', path)


def stats(e):
    return dict(mean=e.mean(), std=e.std(), median=np.median(e),
                p67=np.percentile(e, 67), p95=np.percentile(e, 95))


def smooth_by_run(est, camp_te, lat0, lon0, w=W):
    """Apply the rolling-median trajectory correction, per drive run."""
    exy = to_local_xy(est, lat0, lon0)
    sxy = exy.copy()
    for c in np.unique(camp_te):
        m = np.where(camp_te == c)[0]
        sxy[m, 0] = rolling_median(exy[m, 0], w)
        sxy[m, 1] = rolling_median(exy[m, 1], w)
    return to_latlon(sxy, lat0, lon0)


# ============================================================ RUN EXPERIMENTS
print('Running experiments (this takes a few minutes)...')
R = {}          # R[city] = dict of error arrays

for city in CITIES:
    print(f'\n--- {city} ---')
    M, valid, latlon, camp, _ = load_city(f'{REPO}/{city}/{city}_NBIoT_data.mat')
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()
    alloc = list(load_splits(f'{REPO}/{city}/TP_RP_allocation_{city}.mat').values())[0]
    rp_label = 1 if (alloc[:, 0] == 1).mean() > 0.5 else 2

    d = {k: [] for k in ['wknn', 'wknn_s', 'xgb', 'xgb_s']}

    # ---- Protocol A, WkNN ----
    for r in range(N_RUNS_FAST):
        col = alloc[:, r]
        tr, te = np.where(col == rp_label)[0], np.where(col != rp_label)[0]
        gt, ct = latlon[te], camp[te]
        est = wknn_predict(M, valid, latlon, tr, te)
        sm = smooth_by_run(est, ct, lat0, lon0)
        d['wknn'].append(haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1]))
        d['wknn_s'].append(haversine_m(gt[:, 0], gt[:, 1], sm[:, 0], sm[:, 1]))
    print('  WkNN done')

    # ---- Protocol A, XGBoost ----
    if RUN_XGBOOST:
        X = build_features(M, valid)
        for r in range(N_RUNS_XGB):
            col = alloc[:, r]
            tr, te = np.where(col == rp_label)[0], np.where(col != rp_label)[0]
            gt, ct = latlon[te], camp[te]
            est = xgb_predict(X, latlon, tr, te, lat0, lon0)
            sm = smooth_by_run(est, ct, lat0, lon0)
            d['xgb'].append(haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1]))
            d['xgb_s'].append(haversine_m(gt[:, 0], gt[:, 1], sm[:, 0], sm[:, 1]))
        print('  XGBoost done')

    # ---- Protocol B (leave-one-campaign-out), WkNN ----
    lb, lbs = [], []
    for c in np.unique(camp):
        te, tr = np.where(camp == c)[0], np.where(camp != c)[0]
        gt = latlon[te]
        est = wknn_predict(M, valid, latlon, tr, te)
        sm = smooth_by_run(est, camp[te], lat0, lon0)
        lb.append(haversine_m(gt[:, 0], gt[:, 1], est[:, 0], est[:, 1]))
        lbs.append(haversine_m(gt[:, 0], gt[:, 1], sm[:, 0], sm[:, 1]))
    d['loco'] = [np.concatenate(lb)]
    d['loco_s'] = [np.concatenate(lbs)]
    print('  LOCO done')

    # ---- window sweep ----
    sweep = {}
    col = alloc[:, 0]
    tr, te = np.where(col == rp_label)[0], np.where(col != rp_label)[0]
    gt, ct = latlon[te], camp[te]
    base_est = wknn_predict(M, valid, latlon, tr, te)
    sweep[1] = haversine_m(gt[:, 0], gt[:, 1], base_est[:, 0], base_est[:, 1]).mean()
    for w in [3, 5, 7, 9, 11, 15]:
        sm = smooth_by_run(base_est, ct, lat0, lon0, w)
        sweep[w] = haversine_m(gt[:, 0], gt[:, 1], sm[:, 0], sm[:, 1]).mean()
    d['sweep'] = sweep
    print('  window sweep done')

    # ---- one example trajectory (for fig10), Protocol A so it is representative ----
    col = alloc[:, 0]
    tr_a, te_a = np.where(col == rp_label)[0], np.where(col != rp_label)[0]
    big = max(np.unique(camp[te_a]), key=lambda c: (camp[te_a] == c).sum())
    sel = te_a[camp[te_a] == big]
    est = wknn_predict(M, valid, latlon, tr_a, sel)
    sm = smooth_by_run(est, camp[sel], lat0, lon0)
    d['traj'] = (latlon[sel], est, sm)

    R[city] = {k: (np.concatenate(v) if isinstance(v, list) and len(v) else v)
               for k, v in d.items()}

print('\nMaking figures...')

# ================================================== FIG 1  reproduction check
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(CITIES)); bw = 0.35
pub = [PUBLISHED[c][0] for c in CITIES]
rep = [R[c]['wknn'].mean() for c in CITIES]
b1 = ax.bar(x - bw/2, pub, bw, label='Published (Table III)', color=C_BASE)
b2 = ax.bar(x + bw/2, rep, bw, label='This implementation', color='#7FA6D9')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f'{b.get_height():.2f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(CITIES)
ax.set_ylabel('Mean positioning error [m]')
ax.set_title('Baseline reproduction: published vs. our implementation')
ax.legend(); ax.set_ylim(0, max(pub + rep) * 1.25)
save(fig, 'fig1_baseline_reproduction.png')

# ================================================== FIG 2  main improvement
fig, axes = plt.subplots(1, len(CITIES), figsize=(11, 4.4))
for ax, city in zip(np.atleast_1d(axes), CITIES):
    a, b = stats(R[city]['wknn']), stats(R[city]['wknn_s'])
    keys = ['mean', 'median', 'p95']
    labels = ['Mean', 'Median', '95th pct']
    x = np.arange(3); bw = 0.36
    r1 = ax.bar(x - bw/2, [a[k] for k in keys], bw, label='WkNN (baseline)', color=C_BASE)
    r2 = ax.bar(x + bw/2, [b[k] for k in keys], bw, label='WkNN + smoothing', color=C_PROP)
    for r in list(r1) + list(r2):
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6,
                f'{r.get_height():.1f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Positioning error [m]')
    gain = 100 * (1 - b['mean'] / a['mean'])
    ax.set_title(f'{city}  —  mean error {gain:.1f}% lower')
    ax.legend(fontsize=9); ax.set_ylim(0, a['p95'] * 1.2)
fig.suptitle('Effect of trajectory smoothing (Protocol A)', y=1.02, fontsize=13)
save(fig, 'fig2_main_improvement.png')


# ================================================== FIG 3/4  CDFs with means
def draw_cdf(ax, raw, smo, city):
    xmax = np.percentile(raw, 99)
    # annotation offsets staggered so the two labels never overlap
    for e, lab, col, dx, dy in [(raw, 'WkNN (baseline)', C_BASE, 0.22, 0.13),
                                (smo, 'WkNN + smoothing', C_PROP, 0.10, -0.22)]:
        xs = np.sort(e)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=col, lw=2, label=f'{lab}  (mean {e.mean():.2f} m)')
        # vertical line marking the mean, and its CDF value
        ax.axvline(e.mean(), color=col, ls='--', lw=1.2, alpha=0.8)
        frac = (e <= e.mean()).mean()
        ax.plot([e.mean()], [frac], 'o', color=col, ms=7, zorder=5)
        ax.annotate(f'mean {e.mean():.2f} m\n({100*frac:.0f}% of points below)',
                    xy=(e.mean(), frac),
                    xytext=(e.mean() + xmax * dx, min(0.95, max(0.06, frac + dy))),
                    fontsize=8.5, color=col, ha='left',
                    arrowprops=dict(arrowstyle='->', color=col, lw=0.9))
    ax.set_xlim(0, np.percentile(raw, 99))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Positioning error [m]')
    ax.set_ylabel('CDF')
    ax.set_title(f'{city} — error distribution')
    ax.legend(loc='lower right', fontsize=9)


for city in CITIES:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    draw_cdf(ax, R[city]['wknn'], R[city]['wknn_s'], city)
    save(fig, f'fig3_cdf_{city}.png')

fig, axes = plt.subplots(1, len(CITIES), figsize=(12, 4.6))
for ax, city in zip(np.atleast_1d(axes), CITIES):
    draw_cdf(ax, R[city]['wknn'], R[city]['wknn_s'], city)
save(fig, 'fig4_cdf_all.png')

# ================================================== FIG 5  window sweep
fig, ax = plt.subplots(figsize=(7, 4.4))
for city, col in zip(CITIES, [C_BASE, C_PROP]):
    ws = sorted(R[city]['sweep'])
    vals = [R[city]['sweep'][w] for w in ws]
    ax.plot(ws, vals, 'o-', color=col, lw=2, label=city)
    for w, v in zip(ws, vals):
        ax.annotate(f'{v:.2f}', (w, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color=col)
ax.set_xlabel('Smoothing window W   (W = 1 means no smoothing)')
ax.set_ylabel('Mean positioning error [m]')
ax.set_title('Sensitivity to the window size')
ax.legend()
save(fig, 'fig5_window_sweep.png')

# ================================================== FIG 6  estimator comparison
if RUN_XGBOOST:
    fig, axes = plt.subplots(1, len(CITIES), figsize=(11.5, 4.5))
    for ax, city in zip(np.atleast_1d(axes), CITIES):
        names = ['WkNN', 'WkNN\n+smooth', 'XGBoost', 'XGBoost\n+smooth']
        vals = [R[city]['wknn'].mean(), R[city]['wknn_s'].mean(),
                R[city]['xgb'].mean(), R[city]['xgb_s'].mean()]
        cols = [C_BASE, C_PROP, C_ALT, '#B5651D']
        bars = ax.bar(names, vals, color=cols)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                    f'{b.get_height():.2f}', ha='center', fontsize=10)
        ax.set_ylabel('Mean positioning error [m]')
        ax.set_title(city)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle('The smoothing layer helps both estimators', y=1.02, fontsize=13)
    save(fig, 'fig6_estimator_comparison.png')

# ================================================== FIG 7  % reduction summary
fig, ax = plt.subplots(figsize=(7.5, 4.4))
labels, vals, cols = [], [], []
for city in CITIES:
    labels.append(f'{city}\nWkNN')
    vals.append(100 * (1 - R[city]['wknn_s'].mean() / R[city]['wknn'].mean()))
    cols.append(C_PROP)
    if RUN_XGBOOST:
        labels.append(f'{city}\nXGBoost')
        vals.append(100 * (1 - R[city]['xgb_s'].mean() / R[city]['xgb'].mean()))
        cols.append(C_ALT)
bars = ax.bar(labels, vals, color=cols)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.6,
            f'{b.get_height():.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean error reduction [%]')
ax.set_title('Improvement delivered by the trajectory-smoothing layer')
ax.set_ylim(0, max(vals) * 1.25)
save(fig, 'fig7_error_reduction.png')

# ================================================== FIG 8  protocol comparison
fig, ax = plt.subplots(figsize=(8, 4.4))
x = np.arange(len(CITIES) * 2); bw = 0.36
raw = [R[c]['wknn'].mean() for c in CITIES] + [R[c]['loco'].mean() for c in CITIES]
smo = [R[c]['wknn_s'].mean() for c in CITIES] + [R[c]['loco_s'].mean() for c in CITIES]
names = [f'{c}\nProtocol A' for c in CITIES] + [f'{c}\nProtocol B' for c in CITIES]
b1 = ax.bar(x - bw/2, raw, bw, label='WkNN', color=C_BASE)
b2 = ax.bar(x + bw/2, smo, bw, label='WkNN + smoothing', color=C_PROP)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.05,
            f'{b.get_height():.1f}', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_yscale('log')
ax.set_ylabel('Mean positioning error [m]  (log scale)')
ax.set_title('Random split (A) vs leave-one-drive-out (B)')
ax.legend()
save(fig, 'fig8_protocol_comparison.png')

# ================================================== FIG 9  percentile bars
fig, axes = plt.subplots(1, len(CITIES), figsize=(11.5, 4.4))
for ax, city in zip(np.atleast_1d(axes), CITIES):
    ps = [50, 67, 80, 90, 95, 99]
    a = [np.percentile(R[city]['wknn'], q) for q in ps]
    b = [np.percentile(R[city]['wknn_s'], q) for q in ps]
    red = [100 * (1 - bb / aa) for aa, bb in zip(a, b)]
    bars = ax.bar([str(q) for q in ps], red, color=C_PROP)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.0f}%', ha='center', fontsize=9)
    ax.set_xlabel('Percentile of the error distribution')
    ax.set_ylabel('Error reduction [%]')
    ax.set_title(f'{city} — where the gain comes from')
    ax.axhline(0, color='k', lw=0.8)
fig.suptitle('The improvement grows towards the tail of the distribution',
             y=1.02, fontsize=13)
save(fig, 'fig9_percentile_bars.png')

# ================================================== FIG 10  example trajectory
fig, axes = plt.subplots(1, len(CITIES), figsize=(12, 5))
for ax, city in zip(np.atleast_1d(axes), CITIES):
    gt, est, sm = R[city]['traj']
    n = min(120, len(gt))                     # first n points of the run
    ax.plot(gt[:n, 1], gt[:n, 0], '-o', color='0.35', lw=1.6, ms=2.5, label='True path (GNSS)')
    ax.plot(est[:n, 1], est[:n, 0], 'x', color=C_BASE, ms=5, mew=1.2, alpha=0.8,
            label='WkNN estimates')
    ax.plot(sm[:n, 1], sm[:n, 0], 'o', color=C_PROP, ms=4, alpha=0.85,
            label='After smoothing')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.tick_params(axis='x', labelrotation=20, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'{city} — segment of one drive run')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.suptitle('Raw estimates scatter off the road; smoothing pulls them back',
             y=1.0, fontsize=13)
save(fig, 'fig10_trajectory_example.png')

# ================================================== CSV SUMMARY
rows = [('City', 'Protocol', 'Method', 'Mean', 'Std', 'Median', 'P67', 'P95')]
for city in CITIES:
    combos = [('A', 'WkNN', 'wknn'), ('A', 'WkNN + smoothing', 'wknn_s')]
    if RUN_XGBOOST:
        combos += [('A', 'XGBoost', 'xgb'), ('A', 'XGBoost + smoothing', 'xgb_s')]
    combos += [('B', 'WkNN', 'loco'), ('B', 'WkNN + smoothing', 'loco_s')]
    for proto, name, key in combos:
        s = stats(R[city][key])
        rows.append((city, proto, name, f"{s['mean']:.2f}", f"{s['std']:.2f}",
                     f"{s['median']:.2f}", f"{s['p67']:.2f}", f"{s['p95']:.2f}"))
with open(os.path.join(OUT, 'results_summary.csv'), 'w') as f:
    for r in rows:
        f.write(','.join(r) + '\n')
print('  saved', os.path.join(OUT, 'results_summary.csv'))

print(f'\nDone. All images are in the "{OUT}" folder.')
