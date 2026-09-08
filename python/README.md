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

**If `rmm`, `kvikio`, or `nvtx3` aren't found from `${CONDA_PREFIX}` alone**
(e.g. a devcontainer with those built from source into their own repo build
trees, rather than installed as conda packages), point CMake at each one's
build directory directly rather than relying on `CMAKE_PREFIX_PATH` search
order — `find_package(rmm)` in particular can resolve to a stale/broken
transitive export (missing `nvtx3-targets.cmake`) if the wrong `rmm` build
tree is picked up first:

```bash
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
  -Dnvtx3_DIR=/path/to/cudf/build/_deps/nvtx3-build \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX};/path/to/rmm/cpp/build/<preset>;/path/to/kvikio/cpp/build" \
  -Drapids-cmake-dir=/path/to/cudf/build/_deps/rapids-cmake-src

cmake --build "${CUCASCADE_BUILD}" -j$(nproc)
cmake --install "${CUCASCADE_BUILD}"
```

`nvtx3_DIR` should point at a build tree with a complete
`nvtx3-targets.cmake` (cudf's own `_deps/nvtx3-build` is a reliable choice);
`rmm`'s build tree in `CMAKE_PREFIX_PATH` should likewise be one built
directly (e.g. via `build-rmm-cpp` in the rapids devcontainer tooling), not
picked up incidentally from another project's `_deps/rmm-build`, which may
be missing pieces needed for a standalone `find_package(rmm)`.

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

**If this step can't find `rmm`/`kvikio`/`nvtx3` either**, `scikit-build-core`
(the Python package's build backend) auto-populates its own
`CMAKE_PREFIX_PATH` from sibling rapids project directories, and that
auto-populated list can win over a `CMAKE_PREFIX_PATH` value passed via
`cmake.args` — in particular its guess at `kvikio`'s location may point at
the Python-wheel packaging tree rather than a real `kvikio-config.cmake`.
Pass each dependency's build directory as its own `<pkg>_DIR` cache
variable instead: `<pkg>_DIR` cache variables are consulted before
`CMAKE_PREFIX_PATH` search, so they aren't affected by that override:

```bash
pip install --no-build-isolation --no-deps \
  --config-settings "cmake.args=-Dcudf_DIR=${CUDF_CMAKE_DIR}" \
  --config-settings "cmake.args=-Drmm_DIR=/path/to/rmm/cpp/build/<preset>" \
  --config-settings "cmake.args=-Dkvikio_DIR=/path/to/kvikio/cpp/build" \
  --config-settings "cmake.args=-Dnvtx3_DIR=/path/to/cudf/build/_deps/nvtx3-build" \
  --config-settings "cmake.args=-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}" \
  --config-settings "cmake.args=-Dconcurrentqueue_dir=${CUCASCADE_BUILD}/_deps/concurrentqueue-src" \
  "${CUCASCADE_SRC}/python"
```

## Verify

```python
import cucascade
engine = cucascade.UringEngine()
ds = engine.open("/path/to/file.parquet")
```
