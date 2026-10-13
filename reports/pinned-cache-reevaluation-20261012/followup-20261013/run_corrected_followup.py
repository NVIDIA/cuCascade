"""Two input-cache policies, faithful consumer linking, fixed validation/paired schedule."""
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

base = Path('/tmp/cucascade-pinned-pr-20261012')
original = base / 'sirius-benchmark'
original_manifest = json.loads((original / 'MANIFEST.json').read_text())
roots = {policy: base / ('corrected-input-' + policy) for policy in ['sirius', 'none']}
artifacts = {tag: base / 'sirius-build-corrected' / variant / 'sirius.duckdb_extension' for tag, variant in [('B', 'baseline'), ('C', 'candidate')]}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for policy, root in roots.items():
    root.mkdir(exist_ok=True)
    for name in ['validated_runner.py', 'reference-path.txt', 'canonical-harness-hashes.json']:
        source = original / name
        target = root / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    config = (original / 'config.yaml').read_text()
    assert config.count('mode: sirius') == 1
    config = config.replace('mode: sirius', 'mode: ' + policy)
    target = root / 'config.yaml'
    if target.exists():
        assert target.read_text() == config
    else:
        target.write_text(config)
    (root / 'CONTRACT.md').write_text(f'''# Corrected consumer comparison: input-cache mode {policy}\n\nFixed before timings. Use faithful direct-core-object B/C links restoring normal exported GNU-unique RMM device resource registry; independent ELF/provenance gate precedes GPU smoke. Same production allocator B=main, C=merged8b2d229, same frozen consumer nonallocator objects/dependencies, and same SF50 data/query CPUreference. Only the allocator+constructing owner differ between arms. Original private-registry measurements are excluded from deployment inference and never pooled with these runs.\n\nFOM: sum22per-query medians of iterations1–3 of four grouped runs/query, then median of four process totals/arm. Independent processes; first iteration excluded. Fixed3baseline-noise blocks then B,C,C,B,C,B,B,C. Within-policy schedules are interleaved with the other policy after both noise series. Validate every88results againstCPUreference, disablefallback, require88GPUstarts/completions/engineentries and no errors. No optional extra sampling. First-per-query results are descriptive, not uniformly cold system timing.\n\nFixed48GBGPU/24GBhost,8pipeline/4creator/6scan threads,512MiBbatches, no explicit pinning, no host→GPU memoryprefetcher, no Quent. Mode sirius retains undecoded input withLRU and permits derived backend readahead; mode none disables that policy and compatible reads remainO_DIRECT. CUDA/context/metadata/resource pools survive within eachprocess; initial filefadvise only. This tests input-cache policy sensitivity, not a claim of fully cold physical media or independently priced readahead.\n\nGPUjobs serialized through existinggpu-run.shlease. No profilers during timings and no concurrent expensive CPUanalysis/builds. All source/config/binary hashes frozen beforetiming. Stop on correctness/path failure; environmental contamination excludesaffectedattempt for stated non-result-drivenreason.\n''')


def command(policy, label, tag, iterations):
    root = roots[policy]
    args = list(original_manifest['runs'][3]['command'])
    args = [s.replace('/sirius-runtime/B', '/sirius-runtime-corrected/' + tag) for s in args]
    args[args.index('python') + 1] = str(root / 'validated_runner.py')
    args[args.index('--iterations') + 1] = str(iterations)
    args[args.index('--config') + 1] = str(root / 'config.yaml')
    args[args.index('--output') + 1] = str(root / label)
    args[args.index('--name') + 1] = 'pr185-corrected-' + policy + '-' + label + '-' + tag
    args = ['PR185_ITERATION_VALIDATION=' + str(root / (label + '-iterations.jsonl')) if s.startswith('PR185_ITERATION_VALIDATION=') else s for s in args]
    return args


def run(policy, label, tag, iterations):
    root = roots[policy]
    args = command(policy, label, tag, iterations)
    assert not (root / label).exists() and not (root / (label + '-iterations.jsonl')).exists()
    start = time.time()
    with (root / (label + '.log')).open('w') as log:
        result = subprocess.run(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f'{policy}/{label}: exit {result.returncode}')
    benchmarks = list((root / label).glob('tpch_*'))
    assert len(benchmarks) == 1
    bench = benchmarks[0]
    validation = list(csv.DictReader((bench / 'validation.csv').open()))
    assert len(validation) == 22 and all(r['status'] == 'success' for r in validation)
    rows = [r for r in csv.DictReader((bench / 'csv/runtimes.csv').open()) if r['query'] != 'TOTAL']
    assert len(rows) == 22 * iterations and all(math.isfinite(float(r['runtime_s'])) for r in rows)
    assert {(r['query'], int(r['iteration'])) for r in rows} == {(f'q{q}', i) for q in range(1, 23) for i in range(iterations)}
    checks = [json.loads(s) for s in (root / (label + '-iterations.jsonl')).read_text().splitlines()]
    assert len(checks) == 22 * iterations and all(r['correct'] for r in checks)
    text = '\n'.join(f.read_text(errors='replace') for f in (bench / 'log_dir').rglob('*') if f.is_file())
    gpu = {'starts': text.count('Transparent GPU execution: executing query'), 'completed': text.count('Transparent GPU execution: query completed'), 'engine_queries': len(re.findall(r'query \d+ telemetry_query=', text)), 'errors': re.findall(r'fallback to|Error executing query|Unknown error executing|sweep error', text, re.I)}
    assert all(gpu[key] == 22 * iterations for key in ['starts', 'completed', 'engine_queries']) and not gpu['errors']
    medians = {str(q): statistics.median(float(r['runtime_s']) * 1000 for r in rows if r['query'] == f'q{q}' and (iterations == 1 or int(r['iteration']) > 0)) for q in range(1, 23)}
    rec = {'policy': policy, 'name': label, 'candidate': tag, 'command': args, 'start_epoch': start, 'elapsed_seconds': time.time() - start, 'exit_status': result.returncode, 'benchmark_directory': str(bench), 'iteration_validation': str(root / (label + '-iterations.jsonl')), 'GPU_path': gpu, 'per_query_hot_median_ms': medians, 'FOM_ms': sum(medians.values())}
    print(json.dumps({'finished': policy + '/' + label, 'candidate': tag, 'FOM_ms': rec['FOM_ms'], 'elapsed_seconds': rec['elapsed_seconds']}), flush=True)
    return rec


def verify():
    for name, digest in json.loads((original / 'canonical-harness-hashes.json').read_text()).items():
        assert sha(name) == digest, name
    assert (base / 'sirius-build-corrected/INDEPENDENT-GATE.json').exists(), 'independent link gate required'
    for tag, path in artifacts.items():
        runtime = base / 'sirius-runtime-corrected' / tag / 'extension/sirius'
        runtime.mkdir(parents=True, exist_ok=True)
        link = runtime / 'sirius.duckdb_extension'
        if not link.exists():
            link.symlink_to(path)
        assert link.resolve() == path


mode = sys.argv[1]
verify()
if mode == 'smoke':
    records = [run(policy, 'smoke-' + tag, tag, 1) for policy in roots for tag in ['B', 'C']]
    (base / 'CORRECTED-SMOKES.json').write_text(json.dumps(records, indent=2) + '\n')
elif mode == 'timing':
    assert len(json.loads((base / 'CORRECTED-SMOKES.json').read_text())) == 4
    schedule = [(policy, 'noise-' + str(i), 'B') for policy in roots for i in range(3)]
    schedule += [(policy, 'ab-' + str(i), tag) for i, tag in enumerate(['B', 'C', 'C', 'B', 'C', 'B', 'B', 'C']) for policy in roots]
    manifests = {policy: {'consumer': 'source-equivalent frozen current consumer objects; corrected original direct-object linking', 'config_sha256': sha(root / 'config.yaml'), 'runner_sha256': sha(root / 'validated_runner.py'), 'schedule': [[label, tag] for policy2, label, tag in schedule if policy2 == policy], 'runs': [], 'extensions': {tag: {'path': str(path), 'sha256': sha(path)} for tag, path in artifacts.items()}, 'independent_gate': str(base / 'sirius-build-corrected/INDEPENDENT-GATE.json')} for policy, root in roots.items()}
    for policy, label, tag in schedule:
        rec = run(policy, label, tag, 4)
        manifests[policy]['runs'].append(rec)
        (roots[policy] / 'MANIFEST.json').write_text(json.dumps(manifests[policy], indent=2) + '\n')
    verify()
    for policy, manifest in manifests.items():
        assert sha(roots[policy] / 'config.yaml') == manifest['config_sha256']
        assert sha(roots[policy] / 'validated_runner.py') == manifest['runner_sha256']
        for tag, item in manifest['extensions'].items():
            assert sha(item['path']) == item['sha256']
else:
    raise ValueError(mode)
