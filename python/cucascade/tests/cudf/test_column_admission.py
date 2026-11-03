# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cucascade.cudf import ColumnStats


def test_column_stats_are_not_constructible():
    with pytest.raises(ValueError, match="ScanCache.column_stats"):
        ColumnStats()
