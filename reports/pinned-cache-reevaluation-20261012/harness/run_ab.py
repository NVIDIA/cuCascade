import hashlib,json,subprocess,time
from pathlib import Path
root=Path('/tmp/cucascade-pinned-pr-20261012')
dataset=Path('/tmp/sirius-pr1834-streams-20261007-1940/worktree/test/cpp/integration/data/parquet/lineitem.parquet')
source=Path('/home/kkristensen/Code/cuCascade')
schedule=['B','C','C','B','C','B','B','C']
manifest={"baseline_commit":subprocess.check_output(['git','rev-parse','origin/main'],cwd=source,text=True).strip(),"candidate_commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),"dirty_status":subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True),"dataset":str(dataset),"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),"schedule":schedule,"binaries":{},"runs":[]}
for tag,kind in [('B','baseline'),('C','candidate')]:
 binary=root/('harness-'+kind)/'pinned_bench'
 manifest['binaries'][tag]={"path":str(binary),"sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}
for mode in ['parquet','churn']:
 for index,tag in enumerate(schedule):
  binary=manifest['binaries'][tag]['path']
  args=[binary,mode,str(dataset),'11'] if mode=='parquet' else [binary,mode,'500']
  out=root/f'ab-{mode}-{index}-{tag}.jsonl'
  log=root/f'ab-{mode}-{index}-{tag}.log'
  started=time.time()
  with out.open('w') as stdout,log.open('w') as stderr:
   completed=subprocess.run([str(root/'gpu-run.sh'),*args],stdout=stdout,stderr=stderr)
  manifest['runs'].append({"mode":mode,"index":index,"candidate":tag,"command":args,"start_utc_epoch":started,"elapsed_seconds":time.time()-started,"output":str(out),"log":str(log),"exit_status":completed.returncode})
  (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
  if completed.returncode:raise RuntimeError(f'{mode} {index} {tag} failed: {completed.returncode}')
  print(f'finished {mode} {index} {tag}',flush=True)
