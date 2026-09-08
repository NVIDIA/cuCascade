# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pylibcudf.io.datasource import Datasource

class ReadFuture:
    def get(self) -> int: ...

class CuCascadeDatasource(Datasource):
    def fadvise(self, ranges: list[tuple[int, int]], dev_id: int = -1) -> None: ...
    def duplicate(self) -> CuCascadeDatasource: ...
    def read_ranges_async(
        self, ranges: list[tuple[int, int]], buffer: memoryview
    ) -> list[ReadFuture]: ...
    def read_all_ranges_async(
        self, ranges: list[tuple[int, int]], buffer: memoryview
    ) -> ReadFuture: ...

class UringEngine:
    def __init__(
        self,
        n_reactors: int = 2,
        pool_capacity: int = 2684354560,
        block_size: int = 1048576,
        use_odirect: bool = True,
        numa_node: int = 0,
    ) -> None: ...
    def open(self, path: str) -> CuCascadeDatasource: ...

class RestEngine:
    def __init__(
        self,
        access_key_id: str = "",
        secret_access_key: str = "",
        session_token: str = "",
        region: str = "us-east-1",
        endpoint: str = "",
        n_reactors: int = 4,
        tls_verify: bool = True,
        pool_capacity: int = 2684354560,
        block_size: int = 1048576,
        max_connections: int = 16,
        chunk_size: int = 8388608,
        max_n_chunks: int = 16,
        enable_cache: bool = False,
    ) -> None: ...
    def open(self, path: str) -> CuCascadeDatasource: ...
    def cache_summary(self) -> str: ...
