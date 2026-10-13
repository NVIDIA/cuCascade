import csv
import hashlib
import json
import statistics
from pathlib import Path

base = Path('/tmp/cucascade-pinned-pr-20261012')
results = {}
for policy in ['sirius', 'none']:
    root = base / ('corrected-input-' + policy)
    m = json.loads((root / 'MANIFEST.json').read_text())
    assert len(m['runs']) == 11 and [r['candidate'] for r in m['runs'][3:]] == list('BCCBCBBC')
    noise = [r['FOM_ms'] for r in m['runs'][:3]]
    ab = m['runs'][3:]
    values = {t: [r['FOM_ms'] for r in ab if r['candidate'] == t] for t in ['B', 'C']}
    med = {t: statistics.median(v) for t, v in values.items()}
    pairs = []
    first = {t: [] for t in ['B', 'C']}
    fingerprints = {q: set() for q in range(1, 23)}
    for i in range(0, 8, 2):
        pair = {r['candidate']: r['FOM_ms'] for r in ab[i:i+2]}
        pairs.append(100 * (1 - pair['C'] / pair['B']))
    for r in m['runs']:
        rows = [v for v in csv.DictReader((Path(r['benchmark_directory']) / 'csv/runtimes.csv').open()) if v['query'] != 'TOTAL']
        assert len(rows) == 88
        check = [json.loads(s) for s in Path(r['iteration_validation']).read_text().splitlines()]
        assert len(check) == 88 and all(v['correct'] for v in check)
        for v in check:
            fingerprints[v['query']].add((v['rows'], v['digest']))
        if r['name'].startswith('ab-'):
            first[r['candidate']].append(sum(float(v['runtime_s']) * 1000 for v in rows if int(v['iteration']) == 0))
        assert all(r['GPU_path'][key] == 88 for key in ['starts', 'completed', 'engine_queries']) and not r['GPU_path']['errors']
    assert all(len(v) == 1 for v in fingerprints.values())
    per_query = {}
    for q in range(1, 23):
        local = {t: statistics.median(r['per_query_hot_median_ms'][str(q)] for r in ab if r['candidate'] == t) for t in ['B', 'C']}
        per_query[str(q)] = {'baseline_ms': local['B'], 'candidate_ms': local['C'], 'time_reduction_pct': 100 * (1 - local['C'] / local['B'])}
    first_med = {t: statistics.median(v) for t, v in first.items()}
    results[policy] = {'FOM': 'median of4process sums of22per-query hot medians', 'baseline_noise_ms': noise, 'baseline_noise_median_ms': statistics.median(noise), 'baseline_noise_range_pct': 100 * (max(noise)-min(noise)) / statistics.median(noise), 'process_FOM_ms': values, 'median_FOM_ms': med, 'time_reduction_pct': 100 * (1-med['C']/med['B']), 'adjacent_block_pair_time_reduction_pct': pairs, 'first_per_query_process_sums_ms': first, 'first_per_query_median_sum_ms': first_med, 'first_per_query_time_reduction_pct': 100 * (1-first_med['C']/first_med['B']), 'first_per_query_scope': 'shared process; not a uniformly cold system benchmark', 'checked_noise_and_timed_iterations': 968, 'per_query_medians': per_query}
    for tag, item in m['extensions'].items():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    assert hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest() == m['config_sha256']
    assert hashlib.sha256((root/'validated_runner.py').read_bytes()).hexdigest() == m['runner_sha256']
    (root / 'RESULTS.json').write_text(json.dumps(results[policy], indent=2) + '\n')
(base / 'CORRECTED-RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps({p: {k: v for k, v in r.items() if k != 'per_query_medians'} for p, r in results.items()}, indent=2))
