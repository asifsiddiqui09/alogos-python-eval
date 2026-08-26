import csv

# London-DB: all 204 images are REAL. This is the cleanest test bed for
# trying new scoring weights, since there's no class-mixing to account for.
with open('evaluation/londondb_results_204_kurtosis.csv') as f:
    rows = [
        {
            'file': r['file'],
            'pv': float(r['primary_variance']),
            'k': float(r['kurtosis']),
        }
        for r in csv.DictReader(f)
    ]

# AI datasets, loaded too, purely to check the tradeoff: does fixing
# London-DB break correct AI classifications elsewhere?
AI_FILES = {
    'StarGAN':  'stargan_results_1000_kurtosis.csv',
    'PGGAN v1': 'pggan_v1_results_1000_kurtosis.csv',
    'PGGAN v2': 'pggan_v2_results_1000_kurtosis.csv',
    'FaceApp':  'faceapp_results_1000_kurtosis.csv',
}
ai_rows = []
for name, fname in AI_FILES.items():
    with open('evaluation/' + fname) as f:
        for r in csv.DictReader(f):
            ai_rows.append({
                'dataset': name,
                'pv': float(r['primary_variance']),
                'k': float(r['kurtosis']),
            })


def score(pv, k, kurt_weight):
    pv_weight = 1.0 - kurt_weight
    s = pv_weight * (1 - pv) + kurt_weight * max(0.0, k)
    return max(0.0, min(1.0, s))


# Rank London-DB images by how "easy" they'd be to classify correctly:
# lowest kurtosis first, highest primary_variance as tiebreaker.
ranked = sorted(rows, key=lambda r: (r['k'], -r['pv']))

print("Top 5 easiest London-DB images to correctly classify as REAL:")
for r in ranked[:5]:
    print(f"  {r['file']:<20} primary_variance={r['pv']:.4f}   kurtosis={r['k']:.4f}")
print()

best = ranked[0]
print(f"Highlighting: {best['file']}  (pv={best['pv']:.4f}, k={best['k']:.4f})")
print()

threshold = 0.5
print(f"Sweeping kurtosis weight on London-DB (threshold fixed at {threshold}):")
header = f"{'kurt_wt':>8} {'best score':>11} {'London-DB correct':>19} {'AI correct':>12}"
print(header)
print("-" * len(header))
for w in [0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.0]:
    s_best = score(best['pv'], best['k'], w)
    n_ld_ok = sum(1 for r in rows if score(r['pv'], r['k'], w) < threshold)
    n_ai_ok = sum(1 for r in ai_rows if score(r['pv'], r['k'], w) >= threshold)
    flag = "  <-- best case now REAL" if s_best < threshold else ""
    print(f"{w:>8.3f} {s_best:>11.4f} {n_ld_ok:>10d} / {len(rows)}   {n_ai_ok:>6d} / {len(ai_rows)}{flag}")

print()
print(f"(London-DB: {len(rows)} images, all ground-truth REAL)")
print(f"(AI comparison set: {len(ai_rows)} images across StarGAN/PGGAN v1/PGGAN v2/FaceApp)")
