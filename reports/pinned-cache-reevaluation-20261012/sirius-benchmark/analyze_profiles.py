"""Attribute Q18 using CUDA profiler markers; ignore export records outside them."""
import json
import sqlite3
from pathlib import Path

p = Path(__file__).resolve().parent
summary = {}
for tag in ['B', 'C']:
    con = sqlite3.connect(p / f'profile-{tag}-q18.sqlite')
    names = dict(con.execute('SELECT id, value FROM StringIds'))
    markers = list(con.execute("SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE '%Profiler%' ORDER BY r.start"))
    assert len(markers) == 2
    assert markers[0][2] == 'cuProfilerStart' and markers[1][2].startswith('cudaProfilerStop')
    start, end = markers[0][1], markers[1][0]
    api = {}
    for name_id, count, ns, result in con.execute('SELECT nameId, COUNT(*), SUM(end-start), MAX(returnValue) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? GROUP BY nameId', (start, end)):
        name = names[name_id]
        if any(text in name for text in ['cudaHostAlloc', 'cudaFreeHost', 'cudaStreamWaitEvent', 'cudaEventSynchronize', 'cudaStreamSynchronize']):
            api[name] = {'count': count, 'profiled_total_ms': ns / 1e6, 'max_return_value': result}
    for name in ['cudaHostAlloc_v3020', 'cudaFreeHost_v3020']:
        api.setdefault(name, {'count': 0, 'profiled_total_ms': 0, 'max_return_value': None})
    kernels = con.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?', (start, end)).fetchone()[0]
    copies, copy_bytes = con.execute('SELECT COUNT(*), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=?', (start, end)).fetchone()
    queries = list(con.execute("SELECT n.start, n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value)='sirius::query' AND n.start>=? AND n.end<=?", (start, end)))
    assert len(queries) == 1
    nonzero = [{'name': names[name_id], 'code': code, 'count': count} for name_id, code, count in con.execute('SELECT nameId, returnValue, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND returnValue!=0 GROUP BY nameId, returnValue', (start, end))]
    summary[tag] = {'scope': 'Q18 first hot iteration; explicitly clipped to profiler API markers', 'clip_start_ns': start, 'clip_end_ns': end, 'range_seconds': (end-start)/1e9, 'sirius_query_ranges': queries, 'runtime': api, 'kernels': kernels, 'copies': copies, 'copy_bytes': copy_bytes, 'CUDA_API_nonzero_return_codes': nonzero}
(p / 'PROFILE-SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
