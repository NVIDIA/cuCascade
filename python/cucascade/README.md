# cucascade (Python)

Read remote parquet files through cuCascade's REST backend and host-memory cache, from
pylibcudf-based readers such as cudf-polars.

```python
import cucascade
from pylibcudf.io import SourceInfo

cucascade.start(
    cache_bytes=100 << 30,  # pinned host memory, allocated here (about 0.3 s per GiB)
    endpoint="https://s3.us-east-1.amazonaws.com",
    region="us-east-1",
    access_key=...,
    secret_key=...,
)
source = cucascade.CucascadeDatasource("s3://bucket/table/part-0.parquet")
info = SourceInfo([source])  # read it with any pylibcudf parquet reader
```

A column read by one query is then served from host memory by the next. The cache holds only
what its admission policy keeps: after `cucascade.register(...)` has told it where each column
chunk of a file lives, it ranks columns by how often they are read in full, caches the hot ones
up to `admission_budget` of the cache, and sets the eviction priority of their chunks from that
rank. `cucascade.column_stats()` reports the accounting per column.

## Build

The package builds against an installed cuCascade that has the `io` and `cudf` libraries (see the
top-level README), with `pylibcudf`, Cython and scikit-build-core available:

```
cmake --install build/release --prefix $PREFIX            # cuCascade
CMAKE_PREFIX_PATH=$PREFIX pip install ./python/cucascade   # this package
```

The cuCascade io headers include moodycamel's `concurrentqueue.h`, which cuCascade does not ship;
make it findable (for example `-DCUCASCADE_MOODYCAMEL_HEADER_DIR=...`).

## Test

```
pytest python/cucascade/tests
```

The tests start a session against an unreachable endpoint, so they need a GPU and pinned host
memory but no S3 access.
