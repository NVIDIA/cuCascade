# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# distutils: language = c++

"""Read remote parquet files through cuCascade's REST backend and host-memory cache."""

from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool as cpp_bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource


cdef extern from "session.hpp" namespace "cucascade::python" nogil:
    cdef cppclass session_options:
        size_t cache_bytes
        size_t block_bytes
        double eviction_threshold
        double eviction_batch
        double admission_budget
        size_t min_advise_bytes
        size_t reactors
        size_t connections
        long stall_seconds
        size_t pinning_threads
        string endpoint
        string region
        string access_key
        string secret_key
        string session_token

    void c_start_session "cucascade::python::start_session"(const session_options&) except +
    unique_ptr[datasource] c_open "cucascade::python::open"(const string&) except +
    void c_register_layout "cucascade::python::register_layout"(
        const string&,
        const string&,
        const vector[string]&,
        const vector[uint32_t]&,
        const vector[uint64_t]&,
        const vector[uint64_t]&,
    ) except +
    string c_cache_summary "cucascade::python::cache_summary"() except +
    void c_stop_session "cucascade::python::stop_session"() except +


cdef extern from "cucascade/cudf/column_admission.hpp" namespace "cucascade::io" nogil:
    cdef cppclass c_column_stats "cucascade::io::column_stats":
        string name
        uint64_t registered
        uint64_t requested
        uint64_t admitted
        uint64_t refused
        uint64_t admitted_footprint
        double rank


cdef extern from "session.hpp" namespace "cucascade::python" nogil:
    vector[c_column_stats] c_column_stats_all "cucascade::python::column_stats"() except +


cdef extern from "cudf/io/datasource.hpp" namespace "cudf::io" nogil:
    cdef cppclass sized_datasource "cudf::io::datasource":
        size_t size()


def start(
    *,
    size_t cache_bytes,
    str endpoint,
    str region,
    str access_key,
    str secret_key,
    str session_token="",
    size_t block_bytes=262144,
    double eviction_threshold=0.97,
    double eviction_batch=0.02,
    double admission_budget=0.9,
    size_t min_advise_bytes=262144,
    size_t reactors=16,
    size_t connections=128,
    long stall_seconds=5,
    size_t pinning_threads=8,
):
    """
    Create the process-wide session: pin ``cache_bytes`` of host memory as the cache and start
    the REST reactors. Does nothing if a session exists already.

    The cache is pinned here, so this takes about 0.3 s per GiB.
    """
    cdef session_options options
    options.cache_bytes = cache_bytes
    options.block_bytes = block_bytes
    options.eviction_threshold = eviction_threshold
    options.eviction_batch = eviction_batch
    options.admission_budget = admission_budget
    options.min_advise_bytes = min_advise_bytes
    options.reactors = reactors
    options.connections = connections
    options.stall_seconds = stall_seconds
    options.pinning_threads = pinning_threads
    options.endpoint = endpoint.encode()
    options.region = region.encode()
    options.access_key = access_key.encode()
    options.secret_key = secret_key.encode()
    options.session_token = session_token.encode()
    with nogil:
        c_start_session(options)


def stop():
    """Drain the cache and the reactors. Every datasource must be destroyed first."""
    with nogil:
        c_stop_session()


def register(str path, str table, list columns, list column_of_chunk, list offsets, list sizes):
    """
    Register the column chunks of one file with the column admission policy.

    ``column_of_chunk[k]``, ``offsets[k]`` and ``sizes[k]`` give the column index, byte offset
    and size of column chunk ``k``. Columns are ranked across the files that share ``table``.
    """
    cdef vector[string] names = [name.encode() for name in columns]
    c_register_layout(path.encode(), table.encode(), names, column_of_chunk, offsets, sizes)


def column_stats():
    """Per-column admission accounting, as a list of dicts."""
    cdef vector[c_column_stats] stats = c_column_stats_all()
    return [
        {
            "name": s.name.decode(),
            "registered": s.registered,
            "requested": s.requested,
            "admitted": s.admitted,
            "refused": s.refused,
            "admitted_footprint": s.admitted_footprint,
            "rank": s.rank,
        }
        for s in stats
    ]


def cache_summary():
    """Counters of the cache as text, or an empty string without a session."""
    return c_cache_summary().decode()


cdef class CucascadeDatasource(Datasource):
    """An ``s3://`` object read through cuCascade and its host-memory cache."""

    cdef unique_ptr[datasource] c_source

    def __cinit__(self, str path):
        cdef string c_path = path.encode()
        with nogil:
            self.c_source = c_open(c_path)

    cdef datasource* get_datasource(self) except * nogil:
        return self.c_source.get()

    def nbytes(self):
        """Size of the object in bytes."""
        return (<sized_datasource*> self.c_source.get()).size()
