# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from cucascade.cudf.advising_datasource cimport AdvisingDatasource
from cucascade.io.rest.request_stats cimport RequestStats
from cucascade.libcucascade.cudf.scan_cache cimport (
    scan_cache as cpp_scan_cache,
    scan_cache_options as cpp_scan_cache_options,
)


cdef class ScanCacheOptions:
    cdef cpp_scan_cache_options c_obj


cdef class ScanCache:
    cdef unique_ptr[cpp_scan_cache] c_obj

    cpdef AdvisingDatasource open(self, str path)

    cpdef void register_layout(
        self,
        str path,
        str table,
        list columns,
        list column_of_chunk,
        list offsets,
        list sizes,
    )

    cpdef list column_stats(self)

    cpdef str summary(self)

    cpdef RequestStats request_stats(self)
