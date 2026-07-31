# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuCascade datasource bindings for use with pylibcudf readers.

Provides :class:`UringEngine` (local NVMe via io_uring) and
:class:`RestEngine` (S3/HTTP via libcurl), both producing
:class:`CuCascadeDatasource` instances that implement
``cudf::io::datasource`` and support advisory prefetch via
:meth:`CuCascadeDatasource.fadvise`.
"""

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint8_t
from libcpp.future cimport future
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource as cudf_datasource

from cucascade.datasource cimport (
    byte_range_info,
    cc_datasource,
    io_object_segment,
    rest_datasource_engine,
    uring_datasource_engine,
)

__all__ = ["CuCascadeDatasource", "ReadFuture", "RestEngine", "UringEngine"]


cdef class ReadFuture:
    """A pending async host read issued by :meth:`CuCascadeDatasource.read_ranges_async`.

    Call :meth:`get` to block until the read completes.
    """

    cdef future[size_t] _fut

    def get(self):
        """Block until the read completes.

        Returns
        -------
        int
            Number of bytes transferred.
        """
        cdef size_t result
        with nogil:
            result = self._fut.get()
        return result


cdef class CuCascadeDatasource(Datasource):
    """A pylibcudf :class:`~pylibcudf.io.datasource.Datasource` backed by
    a ``cucascade::io::datasource``.

    Instances are produced by :meth:`UringEngine.open` or
    :meth:`RestEngine.open` and are not constructed directly.

    The owning engine must outlive every datasource it produces.
    When a single file is read by multiple concurrent splits, call
    :meth:`duplicate` to obtain an independent datasource per split so
    that their :meth:`fadvise` calls do not interfere with each other.
    """

    cdef unique_ptr[cc_datasource] _ds

    cdef cudf_datasource* get_datasource(self) except * nogil:
        return <cudf_datasource*>self._ds.get()

    def fadvise(self, list ranges, int dev_id=-1):
        """Hint the IO layer about byte ranges this scan will read soon.

        Queues the ranges into the engine's prefetch cache so that the
        data is staged into pinned host memory before the caller blocks
        on :meth:`~pylibcudf.io.types.SourceInfo`-based reads.

        Parameters
        ----------
        ranges : list[tuple[int, int]]
            Byte ranges as ``(offset, size)`` pairs, for example from
            :meth:`~pylibcudf.io.experimental.HybridScanReader.filter_column_chunks_byte_ranges`
            or
            :meth:`~pylibcudf.io.experimental.HybridScanReader.payload_column_chunks_byte_ranges`.
        dev_id : int, optional
            Preferred CUDA device id for pinned-host staging placement.
            Pass ``-1`` (the default) to express no preference.
        """
        cdef vector[byte_range_info] c_ranges
        cdef int64_t off, sz
        for off, sz in ranges:
            c_ranges.emplace_back(off, sz)
        with nogil:
            deref(self._ds).fadvise(c_ranges, dev_id)

    def duplicate(self):
        """Return a datasource sharing the same file handle with an independent
        prefetch handle.

        Use one duplicate per split when a single file is read by several
        concurrent :class:`~pylibcudf.io.experimental.HybridScanReader`
        instances so that per-split :meth:`fadvise` calls do not overwrite
        each other's prefetch state.

        Returns
        -------
        CuCascadeDatasource
            A new datasource over the same underlying file.
        """
        cdef CuCascadeDatasource out = CuCascadeDatasource.__new__(CuCascadeDatasource)
        with nogil:
            out._ds = move(deref(self._ds).duplicate())
        return out

    def read_ranges_async(self, list ranges, object buffer):
        """Submit async host reads for each range into a contiguous buffer.

        The reads are submitted immediately and run on the engine's reactor
        threads. Call :meth:`ReadFuture.get` on each returned future to wait
        for the corresponding range to complete.

        Parameters
        ----------
        ranges : list[tuple[int, int]]
            Byte ranges as ``(offset, size)`` pairs in the same order that data
            should appear in ``buffer``.
        buffer : memoryview
            Contiguous writable host buffer sized to hold the sum of all range
            sizes. Typically a slice of a :class:`PinnedBuffer` array.

        Returns
        -------
        list[ReadFuture]
            One future per range, in the same order as ``ranges``.
        """
        cdef uint8_t[::1] c_buf = buffer
        cdef uint8_t* base = &c_buf[0]
        cdef list futures = []
        cdef ReadFuture rf
        cdef int64_t off, sz
        cdef size_t dst_offset = 0
        for off, sz in ranges:
            rf = ReadFuture.__new__(ReadFuture)
            rf._fut = deref(self._ds).host_read_async(off, sz, base + dst_offset)
            futures.append(rf)
            dst_offset += sz
        return futures

    def read_all_ranges_async(self, list ranges, object buffer):
        """Submit all byte ranges as a single vectorized host read.

        Uses the engine's scatter-read backend to fetch all ranges in as few
        HTTP requests as possible, writing each range into ``buffer`` at the
        corresponding offset. Returns a single future that resolves when every
        range has been written.

        Parameters
        ----------
        ranges : list[tuple[int, int]]
            Byte ranges as ``(offset, size)`` pairs in file order.
        buffer : memoryview
            Contiguous writable host buffer sized to hold the sum of all range
            sizes.

        Returns
        -------
        ReadFuture
            A single future that resolves when all ranges have been written.
        """
        cdef uint8_t[::1] c_buf = buffer
        cdef uint8_t* base = &c_buf[0]
        cdef vector[io_object_segment] segments
        cdef int64_t off, sz
        cdef size_t dst_offset = 0
        for off, sz in ranges:
            segments.emplace_back(off, sz, base + dst_offset)
            dst_offset += sz
        cdef ReadFuture rf = ReadFuture.__new__(ReadFuture)
        rf._fut = deref(self._ds).host_read_ranges_async(segments)
        return rf


cdef class UringEngine:
    """io_uring-backed datasource engine for local NVMe reads.

    Owns a NUMA-local pinned host staging pool and a pool of io_uring
    reactor threads. All datasources produced by :meth:`open` share the
    engine's resources; the engine must therefore outlive every datasource
    it produces.

    Parameters
    ----------
    n_reactors : int, optional
        Number of io_uring reactor threads. Default is 2.
    pool_capacity : int, optional
        Total capacity of the pinned host staging pool in bytes.
        Default is approximately 2.5 GiB (20 × 128 MiB).
    block_size : int, optional
        Fixed block size in bytes for the staging pool. Must be a power
        of two and at least the O_DIRECT alignment requirement of the
        target filesystem. Default is 1 MiB.
    use_odirect : bool, optional
        Whether to open files with ``O_DIRECT`` to bypass the page cache.
        Default is ``True``.
    numa_node : int, optional
        NUMA node from which to allocate the pinned staging pool.
        Default is 0.
    """

    cdef unique_ptr[uring_datasource_engine] _engine

    def __cinit__(
        self,
        size_t n_reactors=2,
        size_t pool_capacity=2684354560,
        size_t block_size=1048576,
        bint use_odirect=True,
        int numa_node=0,
    ):
        with nogil:
            self._engine.reset(
                new uring_datasource_engine(
                    n_reactors, pool_capacity, block_size, use_odirect, numa_node
                )
            )

    def open(self, str path):
        """Open a datasource for a local file.

        Parameters
        ----------
        path : str
            Path to the local file.

        Returns
        -------
        CuCascadeDatasource
            A datasource bound to this engine. Must not outlive the engine.

        Raises
        ------
        RuntimeError
            If the file cannot be opened.
        """
        cdef CuCascadeDatasource ds = CuCascadeDatasource.__new__(CuCascadeDatasource)
        cdef string c_path = path.encode()
        with nogil:
            ds._ds = move(deref(self._engine).open(c_path))
        return ds


cdef class RestEngine:
    """libcurl-backed datasource engine for S3/HTTP object-store reads.

    Owns a NUMA-local pinned host staging pool and a pool of libcurl
    reactor threads with SigV4 presigned-URL signing. All datasources
    produced by :meth:`open` share the engine's resources; the engine
    must therefore outlive every datasource it produces.

    Parameters
    ----------
    access_key_id : str, optional
        AWS access key ID. Default is ``""`` (reads from environment).
    secret_access_key : str, optional
        AWS secret access key. Default is ``""`` (reads from environment).
    session_token : str, optional
        STS session token; leave empty for long-lived credentials.
    region : str, optional
        AWS region. Default is ``"us-east-1"``.
    endpoint : str, optional
        S3-compatible endpoint host (e.g. ``"s3.amazonaws.com"`` or a
        MinIO ``host:port``). Leave empty to derive from region.
    n_reactors : int, optional
        Number of libcurl reactor threads. Default is 4.
    tls_verify : bool, optional
        Whether to verify TLS peer certificates. Default is ``True``.
    pool_capacity : int, optional
        Total capacity of the pinned host staging pool in bytes.
        Default is approximately 2.5 GiB.
    block_size : int, optional
        Fixed block size in bytes for the staging pool. Default is 1 MiB.
    max_connections : int, optional
        Maximum concurrent in-flight HTTP connections per reactor.
        Default is 16.
    chunk_size : int, optional
        Maximum bytes per ranged GET request. Adjacent segments are fused
        up to this size; oversized segments are split. Default is 8 MiB.
    max_n_chunks : int, optional
        Maximum number of destination buffers fused into a single scatter
        GET. Default is 16.
    enable_cache : bool, optional
        Whether to enable cuCascade's internal prefetch cache. When ``True``,
        :meth:`CuCascadeDatasource.fadvise` queues S3 downloads into cuCascade's
        bounce buffer pool so that subsequent reads may be served from cache
        rather than S3. Default is ``False``.
    """

    cdef unique_ptr[rest_datasource_engine] _engine

    def __cinit__(
        self,
        str access_key_id="",
        str secret_access_key="",
        str session_token="",
        str region="us-east-1",
        str endpoint="",
        size_t n_reactors=4,
        bint tls_verify=True,
        size_t pool_capacity=2684354560,
        size_t block_size=1048576,
        size_t max_connections=16,
        size_t chunk_size=8388608,
        size_t max_n_chunks=16,
        bint enable_cache=False,
    ):
        cdef string c_access_key_id     = access_key_id.encode()
        cdef string c_secret_access_key = secret_access_key.encode()
        cdef string c_session_token     = session_token.encode()
        cdef string c_region            = region.encode()
        cdef string c_endpoint          = endpoint.encode()
        with nogil:
            self._engine.reset(
                new rest_datasource_engine(
                    c_access_key_id,
                    c_secret_access_key,
                    c_session_token,
                    c_region,
                    c_endpoint,
                    n_reactors,
                    tls_verify,
                    pool_capacity,
                    block_size,
                    max_connections,
                    chunk_size,
                    max_n_chunks,
                    enable_cache,
                )
            )

    def open(self, str path):
        """Open a datasource for an S3 URI.

        Issues an HTTP HEAD request to resolve the object size.

        Parameters
        ----------
        path : str
            S3 URI of the form ``s3://bucket/key``.

        Returns
        -------
        CuCascadeDatasource
            A datasource bound to this engine. Must not outlive the engine.

        Raises
        ------
        RuntimeError
            If the HEAD request fails or the URI is malformed.
        """
        cdef CuCascadeDatasource ds = CuCascadeDatasource.__new__(CuCascadeDatasource)
        cdef string c_path = path.encode()
        with nogil:
            ds._ds = move(deref(self._engine).open(c_path))
        return ds
