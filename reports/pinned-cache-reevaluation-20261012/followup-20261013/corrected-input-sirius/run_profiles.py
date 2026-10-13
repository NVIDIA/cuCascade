import hashlib
import json
import subprocess
import time
from pathlib import Path

p = Path('/tmp/cucascade-pinned-pr-20261012/corrected-input-sirius')
manifest = json.loads((p / 'MANIFEST.json').read_text())
assert len(manifest['runs']) == 11, 'finish fixed timings first'
records = []
# Q18 is the largest baseline-noise suite contributor; profile its first hot iteration.
for tag in ['B', 'C']:
    name = 'profile-' + tag
    args = list(manifest['runs'][3]['command'])
    args = [s.replace('/sirius-runtime-corrected/B', '/sirius-runtime-corrected/' + tag) for s in args]
    args[args.index('--output') + 1] = str(p / name)
    args[args.index('--name') + 1] = 'pr185-' + name
    idx = args.index('python')
    args[idx + 1] = str(p / 'profile_runner.py')
    validation = p / (name + '-iterations.jsonl')
    args = ['PR185_ITERATION_VALIDATION=' + str(validation) if s.startswith('PR185_ITERATION_VALIDATION=') else s for s in args]
    args[idx:idx] = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output=' + str(p / (name + '-q18'))]
    if (p / name).exists() or validation.exists():
        raise RuntimeError('refuse to overwrite profile')
    started = time.time()
    with (p / (name + '.log')).open('w') as output:
        result = subprocess.run(args, cwd=p, stdout=output, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f'{name}: exit {result.returncode}')
    rows = [json.loads(line) for line in validation.read_text().splitlines()]
    assert len(rows) == 88 and all(r['correct'] for r in rows)
    assert sum(r['captured'] for r in rows) == 1
    rep = p / (name + '-q18.nsys-rep')
    assert rep.exists()
    record = {'tag': tag, 'command': args, 'start_epoch': started, 'elapsed_seconds': time.time() - started, 'exit_status': result.returncode, 'validated_iterations': len(rows), 'capture_query': 18, 'capture_iteration': 1, 'trace': str(rep), 'trace_sha256': hashlib.sha256(rep.read_bytes()).hexdigest()}
    records.append(record)
    (p / 'PROFILE-MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(record), flush=True)
