# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cucascade.io.rest import RequestStats


def test_request_stats_are_not_constructible():
    with pytest.raises(ValueError, match="ScanCache.request_stats"):
        RequestStats()
