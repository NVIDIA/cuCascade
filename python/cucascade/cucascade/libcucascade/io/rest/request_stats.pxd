# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector


cdef extern from "cucascade/io/rest/request_stats.hpp" namespace "cucascade::io::rest" nogil:

    cdef cppclass request_stats_snapshot:
        uint64_t attempts
        uint64_t succeeded
        uint64_t throttled
        uint64_t http_client_errors
        uint64_t http_server_errors
        uint64_t timed_out
        uint64_t transport_errors
        uint64_t bytes
        uint64_t max_total_ms
        vector[uint64_t] total_ms_histogram
        vector[uint64_t] first_byte_ms_histogram

        @staticmethod
        vector[uint32_t] latency_edges_ms()
