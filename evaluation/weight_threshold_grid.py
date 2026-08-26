import csv

with open('evaluation/londondb_results_204_kurtosis.csv') as f:
    rows = [
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
    with open('evaluation/' + fname) as f:
        for r in csv.DictReader(f):
            ai_rows.append({'pv': float(r['primary_variance']), 'k': float(r['kurtosis'])})


def score(pv, k, w):
    s = (1 - w) * (1 - pv) + w * max(0.0, k)
    return max(0.0, min(1.0, s))


weights = [0.40, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01,
           0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0]
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

results = []
for w in weights:
    for t in thresholds:
        real_ok = sum(1 for r in rows if score(r['pv'], r['k'], w) < t)
        ai_ok = sum(1 for r in ai_rows if score(r['pv'], r['k'], w) >= t)
        real_rate = real_ok / len(rows)
        ai_rate = ai_ok / len(ai_rows)
        balanced = (real_rate + ai_rate) / 2
        results.append((balanced, w, t, real_ok, ai_ok))

results.sort(reverse=True)

print("Top 10 (weight, threshold) combinations by balanced accuracy:")
print(f"{'weight':>7} {'thresh':>7} {'real_ok/204':>12} {'ai_ok/4000':>11} {'balanced_acc':>13}")
for bal, w, t, real_ok, ai_ok in results[:10]:
    print(f"{w:>7.3f} {t:>7.2f} {real_ok:>9d}/204 {ai_ok:>8d}/4000 {bal:>13.4f}")

print()
default = next(r for r in results if r[1] == 0.40 and r[2] == 0.50)
print(f"For comparison, current default (weight=0.40, threshold=0.50):")
print(f"  real={default[3]}/204  ai={default[4]}/4000  balanced_acc={default[0]:.4f}")
