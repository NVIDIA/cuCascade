# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

from cucascade.libcucascade.io.rest.request_stats cimport (
    request_stats_snapshot as cpp_request_stats_snapshot,
)

__all__ = ["RequestStats"]


cdef class RequestStats:
    """
    Counters of the ranged GETs made to the object store.

    Every attempt is counted, including retries, so a run can be compared with another by what the
    store did: how many requests were throttled or failed, and how long they took. The counters only
    grow; subtract two snapshots to see an interval.

    Not constructible directly: get them from :meth:`cucascade.cudf.ScanCache.request_stats`.
    """
    __hash__ = None

    def __init__(self):
        raise ValueError("Get RequestStats from ScanCache.request_stats.")

    @staticmethod
    cdef RequestStats from_libcucascade(cpp_request_stats_snapshot stats):
        cdef RequestStats result = RequestStats.__new__(RequestStats)
        result.c_obj = stats
        return result

    @property
    def attempts(self):
        """Every GET attempt, including retries."""
        return self.c_obj.attempts

    @property
    def succeeded(self):
        """Attempts that returned 200 or 206."""
        return self.c_obj.succeeded

    @property
    def throttled(self):
        """Attempts the store answered with 429 or 503, asking the client to slow down."""
        return self.c_obj.throttled

    @property
    def http_client_errors(self):
        """Attempts that got any other non-success status below 500."""
        return self.c_obj.http_client_errors

    @property
    def http_server_errors(self):
        """Attempts that got 500 or above, except 503."""
        return self.c_obj.http_server_errors

    @property
    def timed_out(self):
        """Attempts that timed out or stalled below the stall detector's rate."""
        return self.c_obj.timed_out

    @property
    def transport_errors(self):
        """Attempts that failed in the transport: reset connection, TLS, DNS."""
        return self.c_obj.transport_errors

    @property
    def bytes(self):
        """Bytes received."""
        return self.c_obj.bytes

    @property
    def max_total_ms(self):
        """Duration of the slowest attempt, in milliseconds."""
        return self.c_obj.max_total_ms

    @property
    def latency_edges_ms(self):
        """Upper edges, in milliseconds, of all but the last histogram bucket."""
        cdef vector[uint32_t] edges = cpp_request_stats_snapshot.latency_edges_ms()
        return list(edges)

    @property
    def total_ms_histogram(self):
        """Attempts per bucket of total duration. One more entry than ``latency_edges_ms``."""
        return list(self.c_obj.total_ms_histogram)

    @property
    def first_byte_ms_histogram(self):
        """Attempts per bucket of time to the first byte."""
        return list(self.c_obj.first_byte_ms_histogram)

    def __repr__(self):
        return (
            f"RequestStats(attempts={self.attempts}, succeeded={self.succeeded}, "
            f"throttled={self.throttled}, timed_out={self.timed_out}, bytes={self.bytes})"
        )
