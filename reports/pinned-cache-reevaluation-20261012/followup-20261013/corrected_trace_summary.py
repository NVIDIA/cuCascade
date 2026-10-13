"""Summarize the single captured Q18 NVTX query; profiler stop ends collection."""
import collections
import json
import sqlite3
from pathlib import Path

base = Path('/tmp/cucascade-pinned-pr-20261012')
profiles = base / 'corrected-input-sirius'
output = {}
for tag in ['B', 'C']:
    con = sqlite3.connect(profiles / f'profile-{tag}-q18.sqlite')
    names = dict(con.execute('SELECT id,value FROM StringIds'))
    queries = [(a,b) for a,b,t,n in con.execute('SELECT start,end,text,textId FROM NVTX_EVENTS') if (t or names.get(n)) == 'sirius::query']
    assert len(queries) == 1 and queries[0][1] is not None
    lo, hi = queries[0]
    runtime = list(con.execute('SELECT start,end,globalTid,nameId,returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME'))
    markers = [(a,b,names[n]) for a,b,_,n,_ in runtime if 'Profiler' in names[n]]
    assert len(markers)==1 and markers[0][2]=='cuProfilerStart' and markers[0][1]<lo
    selected = [r for r in runtime if lo <= r[0] and r[1] <= hi]
    crossing = [r for r in runtime if r[0]<hi and lo<r[1] and not(lo<=r[0] and r[1]<=hi)]
    assert not crossing
    api = collections.defaultdict(lambda: {'count':0,'inclusive_ns':0,'returns':collections.Counter()})
    for a,b,tid,n,result in selected:
        v=api[names[n]];v['count']+=1;v['inclusive_ns']+=b-a;v['returns'][result]+=1
    api={name:{'count':v['count'],'inclusive_ms':v['inclusive_ns']/1e6,'return_codes':dict(v['returns'])} for name,v in api.items()}
    for name in ['cudaHostAlloc_v3020','cudaFreeHost_v3020','cudaMalloc_v3020','cudaFree_v3020']:
        api.setdefault(name,{'count':0,'inclusive_ms':0,'return_codes':{}})
    kind={n:label for n,_,label in con.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER')}
    mem={n:label for n,_,label in con.execute('SELECT * FROM ENUM_CUDA_MEM_KIND')}
    copies=collections.defaultdict(lambda:{'records':0,'bytes':0})
    for a,b,nb,k,src,dst in con.execute('SELECT start,end,bytes,copyKind,srcKind,dstKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
        if lo<=a and b<=hi:
            v=copies[(kind[k],mem[src],mem[dst])];v['records']+=1;v['bytes']+=nb
    copies=[{'direction':k[0],'source':k[1],'destination':k[2],**v} for k,v in copies.items()]
    kernels=con.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?',(lo,hi)).fetchone()[0]
    output[tag]={'provenance':'corrected direct-core-object link; sharedRMM registry independentlyverified','scope':'single hotQ18 sirius::query NVTX range; profiler stop itself ends collection and is notrecorded','query_start_ns':lo,'query_end_ns':hi,'profiled_query_ms':(hi-lo)/1e6,'markers':markers,'runtime':api,'kernels':kernels,'copies':copies,'copy_bytes':sum(v['bytes'] for v in copies),'copy_records':sum(v['records'] for v in copies)}
(base/'CORRECTED-NSYS-SUMMARY.json').write_text(json.dumps(output,indent=2)+'\n')
for tag,r in output.items():
 print(tag)
 for prefix in ['cudaHostAlloc','cudaFreeHost','cudaMalloc','cudaFree','cudaStreamSynchronize','cuStreamSynchronize','cudaStreamWaitEvent','cudaEventSynchronize']:
  print(prefix,{k:v['count'] for k,v in r['runtime'].items() if k.startswith(prefix)})
 print('kernels/copies/bytes',r['kernels'],r['copy_records'],r['copy_bytes'])
 print('copy kinds',r['copies'])
