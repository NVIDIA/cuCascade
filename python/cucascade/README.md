# cucascade (Python)

Python bindings for reading remote parquet files through cuCascade's REST backend and host-memory
cache, from pylibcudf-based readers such as cudf-polars. They are a thin Cython wrapper over the
C++ library; see the [Python development guide](../../docs/python-development-guide.md) for how
they are designed and how to extend them.

```python
import pylibcudf as plc
from cucascade.cudf import ScanCache, ScanCacheOptions

cache = ScanCache(
    ScanCacheOptions(
        cache_bytes=100 << 30,  # pinned host memory, allocated here (about 0.3 s per GiB)
        endpoint="https://s3.us-east-1.amazonaws.com",
        region="us-east-1",
        access_key=...,
        secret_key=...,
    )
)
source = cache.open("s3://bucket/table/part-0.parquet")  # a pylibcudf Datasource
table = plc.io.parquet.read_parquet(
    plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([source])).build()
)
```

A column read by one query is then served from host memory by the next. The cache keeps only what
its admission policy admits: after `cache.register_layout(...)` has told it where each column chunk
of a file lives, it ranks columns by how often they are read in full, caches the hot ones up to
`admission_budget` of the cache, and sets the eviction priority of their chunks from that rank.
`cache.column_stats()` reports the accounting per column.

## Build

The package builds against an installed cuCascade that has the `io` and `cudf` libraries (see the
top-level README), with `pylibcudf`, Cython, `cython-cmake` and `scikit-build-core` available:

```
cmake --install build/release --prefix $PREFIX            # cuCascade
CMAKE_PREFIX_PATH=$PREFIX pip install --no-build-isolation ./python/cucascade
```

The cuCascade io headers include moodycamel's `concurrentqueue.h`, which cuCascade does not ship;
make it findable (for example `-DCUCASCADE_MOODYCAMEL_HEADER_DIR=...` through `CMAKE_ARGS`).

## Test

```
pytest python/cucascade/tests
```

Run it from outside the source directory so Python imports the installed package. The tests create
a cache against an unreachable endpoint, so they need a GPU and pinned host memory but no S3 access.
