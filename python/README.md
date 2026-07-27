# cucascade Python bindings

Python bindings for cuCascade's cudf datasource layer, exposing `UringEngine`
(local NVMe via io_uring) and `RestEngine` (S3/HTTP) with advisory prefetch
support via `CuCascadeDatasource.fadvise()`.

## Prerequisites

- CUDA toolkit
- cudf built or installed (cmake config must be findable)
- cuCascade C++ libraries built and installed
- pylibcudf installed
- Python >= 3.11

## 1. Build and install cuCascade C++

```bash
CUCASCADE_SRC=/path/to/cuCascade
CUCASCADE_BUILD=${CUCASCADE_SRC}/build
CUDF_CMAKE_DIR=/path/to/cudf/build   # directory containing cudf-config.cmake

cmake -S "${CUCASCADE_SRC}" -B "${CUCASCADE_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
  -DCUCASCADE_BUILD_CUDF=ON \
  -DCUCASCADE_BUILD_IO=ON \
  -DCUCASCADE_BUILD_TESTS=OFF \
  -DCUCASCADE_BUILD_BENCHMARKS=OFF \
  -DCUCASCADE_BUILD_SHARED_LIBS=ON \
  -DCUCASCADE_BUILD_STATIC_LIBS=OFF \
  -Dcudf_DIR="${CUDF_CMAKE_DIR}" \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"

cmake --build "${CUCASCADE_BUILD}" -j$(nproc)
cmake --install "${CUCASCADE_BUILD}"
```

Install to `${CONDA_PREFIX}` so the shared libraries are on the default
dynamic linker search path at runtime.

## 2. Build and install the Python package

```bash
CUCASCADE_SRC=/path/to/cuCascade
CUCASCADE_BUILD=${CUCASCADE_SRC}/build
CUDF_CMAKE_DIR=/path/to/cudf/build

pip install --no-build-isolation --no-deps \
  --config-settings "cmake.args=-DCMAKE_PREFIX_PATH=${CONDA_PREFIX};${CUDF_CMAKE_DIR}" \
  --config-settings "cmake.args=-Dconcurrentqueue_dir=${CUCASCADE_BUILD}/_deps/concurrentqueue-src" \
  "${CUCASCADE_SRC}/python"
```

**`CMAKE_PREFIX_PATH`** must include:
- `${CONDA_PREFIX}` — finds cuCascade, liburing, libcurl, OpenSSL
- the cudf cmake build directory — lets `find_package(cudf)` resolve

**`concurrentqueue_dir`** — the moodycamel concurrentqueue headers are
fetched by the cuCascade C++ build and live at
`${CUCASCADE_BUILD}/_deps/concurrentqueue-src`.

**`rapids-cmake-dir`** (optional) — if rapids-cmake is already available
(e.g. inside a pylibcudf build directory), pass it to avoid a network fetch:

```bash
  --config-settings "cmake.args=-Drapids-cmake-dir=/path/to/rapids-cmake-src" \
```

Otherwise the build fetches rapids-cmake automatically from GitHub.

## Verify

```python
import cucascade
engine = cucascade.UringEngine()
ds = engine.open("/path/to/file.parquet")
```
