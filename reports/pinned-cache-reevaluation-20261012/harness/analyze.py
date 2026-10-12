import json,statistics
from pathlib import Path
p=Path('/tmp/cucascade-pinned-pr-20261012')
load=lambda f:[json.loads(s) for s in f.read_text().splitlines()]
noise=[]
for f in sorted(p.glob('noise-baseline-*.jsonl')):
 rows=load(f);noise.append(statistics.median(r['ms'] for r in rows if r.get('type')=='parquet' and r['warm']))
summary={"noise_process_medians_ms":noise,"noise_median_ms":statistics.median(noise),"noise_range_ms":max(noise)-min(noise),"noise_mad_ms":statistics.median(abs(x-statistics.median(noise)) for x in noise),"parquet":{},"churn":{}}
digests=set(); path_counts=set()
for tag in ['B','C']:
 med=[];first=[]
 for f in sorted(p.glob(f'ab-parquet-*-{tag}.jsonl')):
  rows=[r for r in load(f) if r.get('type')=='parquet']
  if len(rows)!=11:raise RuntimeError(f'wrong sample count in {f}')
  med.append(statistics.median(r['ms'] for r in rows if r['warm']));first.append(rows[0]['ms'])
  digests.update((r['digest'],r['rows'],r['columns']) for r in rows)
  path_counts.update((r['allocations'],r['large_allocations'],r['requested_bytes']) for r in rows)
 summary['parquet'][tag]={"process_medians_ms":med,"median_ms":statistics.median(med),"first_read_ms":first}
if len(digests)!=1 or len(path_counts)!=1:raise RuntimeError('inconsistent cross-process result or execution path')
summary['parquet']['validated_content']=list(digests)[0]
summary['parquet']['allocations_large_bytes']=list(path_counts)[0]
summary['parquet']['candidate_percent_change']=(summary['parquet']['C']['median_ms']/summary['parquet']['B']['median_ms']-1)*100
for size in [16384,131072,1048576,16777216]:
 rec={}
 for tag in ['B','C']:
  med=[]
  for f in sorted(p.glob(f'ab-churn-*-{tag}.jsonl')):
   rows=[r for r in load(f) if r.get('type')=='churn' and r['bytes']==size and r['warm']]
   if len(rows)!=3:raise RuntimeError(f'wrong churn count in {f}')
   med.append(statistics.median(r['us_per_op'] for r in rows))
  rec[tag]={"process_medians_us":med,"median_us":statistics.median(med)}
 rec['speedup']=rec['B']['median_us']/rec['C']['median_us']
 rec['reduction_percent']=(1-rec['C']['median_us']/rec['B']['median_us'])*100
 summary['churn'][str(size)]=rec
(p/'RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
