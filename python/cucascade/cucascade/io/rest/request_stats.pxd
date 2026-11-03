# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cucascade.libcucascade.io.rest.request_stats cimport (
    request_stats_snapshot as cpp_request_stats_snapshot,
)


cdef class RequestStats:
    cdef cpp_request_stats_snapshot c_obj

    @staticmethod
    cdef RequestStats from_libcucascade(cpp_request_stats_snapshot stats)
