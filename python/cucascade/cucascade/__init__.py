# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read remote parquet files through cuCascade's REST backend and host-memory cache."""

from cucascade._io import (
    CucascadeDatasource,
    cache_summary,
    column_stats,
    register,
    start,
    stop,
)

__all__ = [
    "CucascadeDatasource",
    "cache_summary",
    "column_stats",
    "register",
    "start",
    "stop",
]
