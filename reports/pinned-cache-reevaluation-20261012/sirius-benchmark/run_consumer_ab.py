import csv,hashlib,json,math,os,re,statistics,subprocess,time
from pathlib import Path
p=Path('/tmp/cucascade-pinned-pr-20261012/sirius-benchmark')
reference=(p/'reference-path.txt').read_text().strip()
manifest={'consumer':'Sirius45253bf source-equivalent frozen objects','dependency':'e1b67fe accounting changes retained, allocator baseline identical to origin/main','config_sha256':hashlib.sha256((p/'config.yaml').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256((p/'validated_runner.py').read_bytes()).hexdigest(),'schedule':[['noise-'+str(i),'B'] for i in range(3)]+[['ab-'+str(i),t] for i,t in enumerate(['B','C','C','B','C','B','B','C'])],'runs':[]}
manifest['extensions']={t:{'path':str(Path('/tmp/cucascade-pinned-pr-20261012/sirius-build')/k/'sirius.duckdb_extension'),'sha256':hashlib.sha256((Path('/tmp/cucascade-pinned-pr-20261012/sirius-build')/k/'sirius.duckdb_extension').read_bytes()).hexdigest()} for t,k in [('B','baseline'),('C','candidate')]}
for name,tag in manifest['schedule']:
 out=p/name
 if out.exists():raise RuntimeError(f'refuse to overwrite {out}')
 iteration_validation=p/(name+'-iterations.jsonl')
 args=['/tmp/cucascade-pinned-pr-20261012/gpu-run.sh','pixi','run','--as-is','--manifest-path','/home/kkristensen/Code/sirius/pixi.toml','env','SIRIUS_BUILD_PATH=/tmp/cucascade-pinned-pr-20261012/sirius-runtime/'+tag,'SIRIUS_PRE_SQL=SET enable_duckdb_fallback=false;','PR185_CPU_REFERENCE='+reference,'PR185_ITERATION_VALIDATION='+str(iteration_validation),'python',str(p/'validated_runner.py'),'--input','/tmp/sirius-pr1834-streams-20261007-1940/data/tpch_sf50','--scale-factor','50','--data-source','parquet','--queries','1-22','--engine','gpu','--iterations','4','--pin','none','--config',str(p/'config.yaml'),'--output',str(out),'--name','pr185-'+name+'-'+tag,'--duckdb-results',reference]
 started=time.time()
 with (p/(name+'.log')).open('w') as log:
  completed=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,cwd=p,timeout=600)
 if completed.returncode:raise RuntimeError(f'{name} failed: exit{completed.returncode}')
 bench=list(out.glob('tpch_*'))
 if len(bench)!=1:raise RuntimeError(f'{name}: output dirs{bench}')
 bench=bench[0]
 validation=list(csv.DictReader((bench/'validation.csv').open()))
 if len(validation)!=22 or any(r['status']!='success' for r in validation):raise RuntimeError(f'{name}: incomplete CPU validation')
 rows=list(csv.DictReader((bench/'csv/runtimes.csv').open()))
 rows=[r for r in rows if r['query']!='TOTAL']
 if len(rows)!=88 or any(not math.isfinite(float(r['runtime_s'])) for r in rows):raise RuntimeError(f'{name}: missing/nonfinite timing')
 correctness=[json.loads(r) for r in iteration_validation.read_text().splitlines()]
 if len(correctness)!=88 or not all(r['correct'] for r in correctness):raise RuntimeError(f'{name}: iteration validation missing')
 logfiles=[f for f in (bench/'log_dir').rglob('*') if f.is_file()]
 text='\n'.join(f.read_text(errors='replace') for f in logfiles)
 path={'starts':text.count('Transparent GPU execution: executing query'),'completed':text.count('Transparent GPU execution: query completed'),'engine_queries':len(re.findall(r'query \d+ telemetry_query=',text)),'errors':re.findall(r'fallback to|Error executing query|Unknown error executing|sweep error',text,re.I)}
 if any(path[x]!=88 for x in ['starts','completed','engine_queries']) or path['errors']:raise RuntimeError(f'{name}: GPU path evidence failed{path}')
 per_query={str(q):statistics.median(float(r['runtime_s'])*1000 for r in rows if r['query']==f'q{q}' and int(r['iteration'])>0) for q in range(1,23)}
 rec={'name':name,'candidate':tag,'command':args,'start_epoch':started,'elapsed_seconds':time.time()-started,'exit_status':completed.returncode,'benchmark_directory':str(bench),'iteration_validation':str(iteration_validation),'GPU_path':path,'per_query_hot_median_ms':per_query,'FOM_ms':sum(per_query.values())}
 manifest['runs'].append(rec)
 (p/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'finished':name,'candidate':tag,'FOM_ms':rec['FOM_ms'],'elapsed_seconds':rec['elapsed_seconds']}),flush=True)
