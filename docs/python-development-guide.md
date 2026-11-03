# Python Development Guide

`python/cucascade` is a lightweight Cython wrapper around cuCascade. It aims to give Python code a
near-zero-overhead interface to the C++ library, and to interoperate with
[pylibcudf](https://github.com/NVIDIA/cudf/tree/main/python/pylibcudf), whose conventions it follows
closely. When this guide is silent, do what pylibcudf does.

The first consumer is the cudf-polars engine, which reads remote parquet files through the cache
with a `ScanCache` and pylibcudf's parquet reader.

## Design principles

- **The wrapper contains no logic.** Behaviour lives in the C++ library, where it has C++ tests and
  is available to every consumer. The Python layer converts arguments, releases the GIL, calls C++
  and converts the result. `ScanCache` is the model: it is a C++ class
  (`include/cucascade/cudf/scan_cache.hpp`) that assembles the cache, and the Python class mirrors it
  one to one. If you find yourself writing a loop or a policy in a `.pyx`, move it to C++.
- **Mirror the C++ API.** Each C++ header gets a Python module with the same name, each C++ type a
  `cdef class` with the same name in `CapsCase`, and each function a method with the same signature.
  Python types stand in for C++ types (`str` for `std::string`, `list` for `std::vector`).
- **Every public function and method is `cpdef`.** It is then callable from Cython and Python. The
  call overhead is nanoseconds, against microseconds for a CUDA launch or a network request.
- **Type everything with Cython syntax**, not PEP 484: variables are C types or `cdef class`es.
- **Release the GIL around C++ calls** (`with nogil:`), after the Python preprocessing and before
  the C++ call, and convert what comes back immediately.
- **Keep dependencies minimal.** The runtime dependency is pylibcudf (the bindings return its
  `Datasource`), plus the libraries cuCascade itself needs.
- **Stubs are written by hand.** Every `.pyx` has a matching `.pyi`; update it with the code.

## Layout

```
python/cucascade/
  CMakeLists.txt, cmake/cucascade_cython.cmake   one extension module per .pyx
  pyproject.toml, VERSION
  cucascade/
    exception_handler.pxd       C++ exception -> Python exception mapping
    libcucascade/               declarations of the C++ API, mirroring include/cucascade/
      cudf/scan_cache.pxd
    cudf/                       the wrappers, one module per C++ header
      scan_cache.{pxd,pyx,pyi}
    py.typed
  tests/                        one test module per wrapper module
```

- `libcucascade/` only declares C++: `cdef extern from "<header>" namespace "..." nogil:`, one `.pxd`
  per header, with every call declared `except +cucascade_exception_handler`. It declares what is
  bound so far, not the whole header. Add to it when a consumer needs more.
- A wrapper module is a triplet. The `.pxd` declares the classes so other Cython code can `cimport`
  them, the `.pyx` implements them, and the `.pyi` types them for Python tools.
- C++ names are aliased `as cpp_name` when imported, and locals that hold a C++ value are prefixed
  `c_`, so the two worlds stay apart.

## Wrapping a type

A type that owns a C++ object has an attribute `c_obj` holding it (a value, or a `unique_ptr`):

```cython
cdef class ColumnStats:
    cdef cpp_column_stats c_obj

    @staticmethod
    cdef ColumnStats from_libcucascade(cpp_column_stats stats)
```

- A type that Python creates (`ScanCacheOptions`, `ScanCache`) has a documented `__init__`.
- A type that only C++ produces (`ColumnStats`, `AdvisingDatasource`) refuses direct construction,
  with a message that says where to get one, and is built by a `from_libcucascade` factory:

  ```cython
  def __init__(self):
      raise ValueError("Get ColumnStats from ScanCache.column_stats.")
  ```
- An object that depends on another keeps a Python reference to it, so lifetimes follow Python's
  reference counting: an `AdvisingDatasource` holds the `ScanCache` that opened it.
- Plain attributes are `@property`; operations are `cpdef` methods with a numpydoc docstring
  (`Parameters`, `Returns`) and `__all__` lists the module's public names, sorted.
- C++ enums are declared `cpdef enum class` in `libcucascade/`, and a module that exposes one needs a
  matching `.pyx` so the Python enum is generated (see pylibcudf's developer guide for the details).

## Calling C++

```cython
cpdef AdvisingDatasource open(self, str path):
    cdef string c_path = path.encode()
    cdef unique_ptr[datasource] c_source
    with nogil:
        c_source = move(self.c_obj.get().open(c_path))
    return AdvisingDatasource.from_libcucascade(move(c_source), self)
```

Python objects are converted before the `nogil` block and the result right after it.

## Exceptions

Declare every C++ call that can throw with `except +cucascade_exception_handler`. The handler in
`exception_handler.pxd` maps C++ exceptions to Python ones:

| C++ | Python |
|---|---|
| `std::invalid_argument`, `std::domain_error` | `ValueError` |
| `std::out_of_range` | `IndexError` |
| `std::bad_alloc` | `MemoryError` |
| `std::ios_base::failure` | `OSError` |
| `std::overflow_error` | `OverflowError` |
| `std::range_error`, `std::underflow_error` | `ArithmeticError` |
| `cucascade::io::credential_error` | `PermissionError` |
| `cucascade::io::object_changed_error` | `OSError` |
| anything else derived from `std::exception` | `RuntimeError` |

If the exception a caller should see is not the one the C++ throws, change the C++ so it throws the
right type (a bad argument should be `std::invalid_argument`), rather than translating it in Python.
When a new C++ exception type needs a mapping, add it to the handler, before its base class, and add
a test.

## ABI

The extension modules are built against Python's limited API (`abi3`, `CYTHON_LIMITED_API`), the
same way pylibcudf is, because they subclass its types. A module built any other way crashes inside
CPython when it touches the parent type. `cmake/cucascade_cython.cmake` sets this up for every
module; use it for new ones.

## Testing

The tests check the **bindings**: that arguments convert, results convert back, exceptions have the
right type, and lifetimes are right. The behaviour behind them is tested in C++
(`test/cudf/test_scan_cache.cpp` and friends). If a Python test finds a C++ bug, add a C++ test.

- One test module per wrapper module (`tests/cudf/test_scan_cache.py` for `cudf/scan_cache.pyx`).
- Test the exceptions a caller can hit, from the caller's side.
- The tests must not need S3. They create a cache against an endpoint nothing listens on, which
  needs a GPU and pinned host memory but makes no request.
- Creating a `ScanCache` pins memory, so share one through a module-scoped fixture, and treat it as
  shared: use names that other tests do not.
- `pytest` runs with `--strict-markers`, `--strict-config`, `filterwarnings = error` and
  `xfail_strict`; keep it that way.

## Adding a binding

1. Add or extend the C++ in `include/cucascade/` and `src/`, with C++ tests. The Python layer comes
   last.
2. Declare what you bind in `libcucascade/<dir>/<header>.pxd`, with
   `except +cucascade_exception_handler` on every call that can throw.
3. Write `<module>.pxd`, `<module>.pyx` and `<module>.pyi` in the matching directory.
4. List the `.pyx` in that directory's `CMakeLists.txt`, and export the public names from the
   package `__init__.py` and `__init__.pxd`.
5. Add `tests/<dir>/test_<module>.py`.
6. Build and run the tests (below), and run `pre-commit run -a`.

## Building and testing

The package builds against an installed cuCascade that has the `io` and `cudf` libraries:

```
cmake --install build/release --prefix $PREFIX
CMAKE_PREFIX_PATH=$PREFIX pip install --no-build-isolation ./python/cucascade
pytest python/cucascade/tests
```

Run the tests from outside the source directory, or Python imports the source tree instead of the
build. See `python/cucascade/README.md` for the dependencies the build needs.

## Style

- Follow the conventions of the code around you; both cuCascade (leading underscore on private C++
  members) and pylibcudf (leading underscore on private module names) have them.
- Python is formatted with `black`; Cython and `.pxd` files are checked with `cython-lint`; C++ with
  `clang-format`; CMake with `cmake-format`. `pre-commit run -a` runs all of them.
- Every file starts with the SPDX header the other files of its kind have.
