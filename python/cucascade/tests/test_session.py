# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cucascade

START_ARGS = {
    "endpoint": "http://127.0.0.1:9",
    "region": "us-east-1",
    "access_key": "test",
    "secret_key": "test",
}


def test_without_a_session():
    assert cucascade.column_stats() == []
    assert cucascade.cache_summary() == ""
    with pytest.raises(RuntimeError, match="no session"):
        cucascade.CucascadeDatasource("s3://bucket/a.parquet")
    with pytest.raises(RuntimeError, match="no session"):
        cucascade.register("s3://bucket/a.parquet", "t", ["a"], [0], [0], [10])
    cucascade.stop()


def test_a_session_needs_a_cache():
    with pytest.raises(ValueError, match="cache_bytes"):
        cucascade.start(cache_bytes=0, **START_ARGS)


def test_session_registers_layouts_and_reports_them():
    cucascade.start(cache_bytes=1 << 30, **START_ARGS)
    try:
        cucascade.register(
            "s3://bucket/t/f1.parquet",
            "s3://bucket/t",
            ["a", "b"],
            [0, 1],
            [0, 4000],
            [4000, 4000],
        )
        stats = {s["name"]: s for s in cucascade.column_stats()}
        assert stats["s3://bucket/t/a"]["registered"] == 4000
        assert stats["s3://bucket/t/b"]["registered"] == 4000
        assert stats["s3://bucket/t/a"]["requested"] == 0
        assert cucascade.cache_summary().startswith("fs_cache:")
    finally:
        cucascade.stop()


def test_a_malformed_layout_is_rejected():
    cucascade.start(cache_bytes=1 << 30, **START_ARGS)
    try:
        with pytest.raises(ValueError, match="same length"):
            cucascade.register(
                "s3://bucket/t/f2.parquet", "t", ["a"], [0], [0], [10, 10]
            )
        with pytest.raises(ValueError, match="out of range"):
            cucascade.register("s3://bucket/t/f2.parquet", "t", ["a"], [3], [0], [10])
    finally:
        cucascade.stop()
