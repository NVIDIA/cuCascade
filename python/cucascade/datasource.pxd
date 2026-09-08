# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t, uint8_t
from libcpp cimport bool as cpp_bool
from libcpp.future cimport future
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


from pylibcudf.libcudf.io.datasource cimport datasource as cudf_datasource


cdef extern from "cudf/io/text/byte_range_info.hpp" namespace "cudf::io::text" nogil:
    cdef cppclass byte_range_info:
        byte_range_info(int64_t offset, int64_t size) except +
        int64_t offset() const
        int64_t size() const


cdef extern from "cucascade/io/types.hpp" namespace "cucascade::io" nogil:
    cdef cppclass io_object_segment:
        io_object_segment(size_t offset, size_t size, uint8_t* buffer) except +


cdef extern from "cucascade/cudf/datasource.hpp" namespace "cucascade::io" nogil:
    cdef cppclass cc_datasource "cucascade::io::datasource"(cudf_datasource):
        unique_ptr[cc_datasource] duplicate() except +
        void fadvise(const vector[byte_range_info]& ranges, int dev_id) except +
        future[size_t] host_read_async(size_t offset, size_t size, uint8_t* dst) except +
        future[size_t] host_read_ranges_async(vector[io_object_segment]& segments) except +


cdef extern from "cucascade/cudf/uring_datasource_engine.hpp" namespace "cucascade::io" nogil:
    cdef cppclass uring_datasource_engine:
        uring_datasource_engine(size_t n_reactors,
                                size_t pool_capacity,
                                size_t block_size,
                                cpp_bool use_odirect,
                                int numa_node) except +
        unique_ptr[cc_datasource] open(string path) except +


cdef extern from "cucascade/cudf/rest_datasource_engine.hpp" namespace "cucascade::io" nogil:
    cdef cppclass rest_datasource_engine:
        rest_datasource_engine(string access_key_id,
                               string secret_access_key,
                               string session_token,
                               string region,
                               string endpoint,
                               size_t n_reactors,
                               cpp_bool tls_verify,
                               size_t pool_capacity,
                               size_t block_size,
                               size_t max_connections,
                               size_t chunk_size,
                               size_t max_n_chunks,
                               cpp_bool enable_cache) except +
        unique_ptr[cc_datasource] open(string path) except +
        string cache_summary() except +
