"""Profile one hot Q18 inside the unchanged canonical suite; validate every result."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/kkristensen/Code/sirius/test/tpch_performance')
import performance_test as canonical

canonical._RESULT_EVAL_GLOBALS['datetime'] = datetime
reference = Path(os.environ['PR185_CPU_REFERENCE'])
record_path = Path(os.environ['PR185_ITERATION_VALIDATION'])
expected = {}
iterations = {}
original_time_query = canonical.time_query


def checked_time_query(con, qnum, query_sql, use_gpu, profile_path=None):
    iteration = iterations.get(qnum, 0)
    iterations[qnum] = iteration + 1
    capture = use_gpu and qnum == 18 and iteration == 1
    if capture:
        con.execute('CALL profiler_start();').fetchall()
    try:
        elapsed, rows = original_time_query(con, qnum, query_sql, use_gpu, profile_path)
    finally:
        if capture:
            con.execute('CALL profiler_stop();').fetchall()
    if use_gpu:
        if qnum not in expected:
            expected[qnum] = sorted(canonical._load_result_file(str(reference / f'q{qnum}/result.txt')), key=str)
        ordered = sorted(rows, key=str)
        if len(ordered) != len(expected[qnum]) or not all(
            canonical._rows_match(a, b, canonical.VALIDATION_ABS_TOL)
            for a, b in zip(expected[qnum], ordered)
        ):
            raise RuntimeError(f'q{qnum}: result differs from CPU reference')
        digest = hashlib.sha256(''.join(repr(x) + '\n' for x in ordered).encode()).hexdigest()
        with record_path.open('a') as output:
            output.write(json.dumps({'query': qnum, 'iteration': iteration, 'rows': len(rows), 'digest': digest, 'correct': True, 'captured': capture}) + '\n')
    return elapsed, rows


canonical.time_query = checked_time_query
canonical.main()
