import csv


def load_csv(fname):
    with open('evaluation/' + fname) as f:
        return [
            {'pv': float(r['primary_variance']), 'k': float(r['kurtosis'])}
            for r in csv.DictReader(f)
        ]


AI_FILES = [
    'stargan_results_1000_kurtosis.csv',
    'pggan_v1_results_1000_kurtosis.csv',
    'pggan_v2_results_1000_kurtosis.csv',
    'faceapp_results_1000_kurtosis.csv',
]
ai_rows = []
for fname in AI_FILES:
    ai_rows.extend(load_csv(fname))


def score(pv, k, w):
    s = (1 - w) * (1 - pv) + w * max(0.0, k)
    return max(0.0, min(1.0, s))


def grid_search(real_rows, real_name):
    weights = [0.40, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01,
               0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0]
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []
    for w in weights:
        for t in thresholds:
            real_ok = sum(1 for r in real_rows if score(r['pv'], r['k'], w) < t)
            ai_ok = sum(1 for r in ai_rows if score(r['pv'], r['k'], w) >= t)
            real_rate = real_ok / len(real_rows)
            ai_rate = ai_ok / len(ai_rows)
            balanced = (real_rate + ai_rate) / 2
            results.append((balanced, w, t, real_ok, ai_ok))
    results.sort(reverse=True)
    print(f"=== {real_name} (n={len(real_rows)}) as the REAL set ===")
    print(f"{'weight':>7} {'thresh':>7} {'real_ok':>14} {'ai_ok':>13} {'balanced_acc':>13}")
    for bal, w, t, real_ok, ai_ok in results[:5]:
        print(f"{w:>7.3f} {t:>7.2f} {real_ok:>9d}/{len(real_rows):<4} {ai_ok:>8d}/{len(ai_rows)} {bal:>13.4f}")
    print()


londondb_rows = load_csv('londondb_results_204_kurtosis.csv')
ffhq_rows = load_csv('ffhq_results_1000_kurtosis.csv')
celeba_rows = load_csv('celeba_results_1000_kurtosis.csv')

grid_search(londondb_rows, 'London-DB')
grid_search(ffhq_rows, 'FFHQ')
grid_search(celeba_rows, 'CelebA')

combined_rows = londondb_rows + ffhq_rows + celeba_rows
grid_search(combined_rows, 'All three real datasets combined')
