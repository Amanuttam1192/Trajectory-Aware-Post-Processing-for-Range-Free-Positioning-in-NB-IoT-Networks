"""
Phase 5 - Paper-ready outputs: results table + CDF figures.

Consumes the .npy error arrays saved by Phases 1, 2, and 4 (run those first,
in this folder). Produces:
  results_table.csv / results_table.md   - the main comparison table
  cdf_protocolA_<city>.png               - CDF: WkNN vs WkNN+smoothing (random split)
  cdf_protocolB_<city>.png               - CDF under leave-one-campaign-out
All positioning-error axes in meters.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CITIES = ['Oslo', 'Rome']
PAPER = {  # Table III, Strategy I (WkNN, PCA off) - the published baseline
    'Oslo': (15.67, 21.35),
    'Rome': (13.12, 18.67),
}


def stats(e):
    return dict(mean=e.mean(), std=e.std(), median=np.median(e),
                p67=np.percentile(e, 67), p95=np.percentile(e, 95))


def cdf(ax, e, label, **kw):
    x = np.sort(e)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, label=label, **kw)


rows = []
for city in CITIES:
    raw_A = np.load(f'phase4_random_raw_{city}.npy')
    smo_A = np.load(f'phase4_random_smoothed_{city}.npy')
    raw_B = np.load(f'phase2_err_{city}.npy')
    smo_B = np.load(f'phase4_loco_smoothed_{city}.npy')

    for proto, raw, smo in [('random split', raw_A, smo_A),
                            ('LOCO', raw_B, smo_B)]:
        for method, e in [('WkNN (Strategy I)', raw),
                          ('WkNN + trajectory smoothing', smo)]:
            s = stats(e)
            rows.append([city, proto, method,
                         f"{s['mean']:.2f}", f"{s['std']:.2f}",
                         f"{s['median']:.2f}", f"{s['p67']:.2f}",
                         f"{s['p95']:.2f}"])

    # ---- CDF figures ----
    for proto, raw, smo, tag in [('Random 70/30 split', raw_A, smo_A, 'A'),
                                 ('Leave-one-campaign-out', raw_B, smo_B, 'B')]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        cdf(ax, raw, 'WkNN (Strategy I)', color='#555', lw=1.8)
        cdf(ax, smo, 'WkNN + trajectory smoothing', color='#c0392b', lw=1.8)
        ax.set_xlabel('Positioning error [m]')
        ax.set_ylabel('CDF')
        ax.set_title(f'{city} - {proto}')
        ax.set_xlim(0, np.percentile(raw, 99))
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        fig.savefig(f'cdf_protocol{tag}_{city}.png', dpi=200)
        plt.close(fig)

header = ['City', 'Protocol', 'Method', 'Mean [m]', 'Std [m]',
          'Median [m]', 'P67 [m]', 'P95 [m]']
with open('results_table.csv', 'w') as f:
    f.write(','.join(header) + '\n')
    for r in rows:
        f.write(','.join(r) + '\n')
with open('results_table.md', 'w') as f:
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '---|' * len(header) + '\n')
    for r in rows:
        f.write('| ' + ' | '.join(r) + ' |\n')

print('Published baseline (Table III, Strategy I): '
      + ', '.join(f'{c} {PAPER[c][0]} m' for c in CITIES))
print('\nGenerated: results_table.csv, results_table.md, and 4 CDF figures.')
with open('results_table.md') as f:
    print(f.read())