/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Standalone parquet range-read I/O benchmark.  Unlike parquet_benchmark
// (which times a full cudf::io::read_parquet — I/O + decode + materialize),
// this benchmark isolates the *raw range-read cost*: it uses hybrid_scan to
// find the column-chunk byte ranges a column projection touches, then times
// only the reads of those ranges into a destination tier (host or device).
//
// Two read paths over the SAME uring backend are compared:
//   io_context – the native cucascade::io ioctx.  host reads go through the
//                vector-I/O primitive (host_read_ranges_async_io); device reads
//                through device_read_async.
//   cudf       – the cucascade::io::datasource (a cudf::io::datasource) whose
//                host_read_async / device_read_async are issued per range.
// All async operations are collected first and synchronized at the end.

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <fcntl.h>
#include <glob.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// 4 columns used by the classic TPC-H lineitem aggregations (Q1, Q6, …).
static const std::vector<std::string> COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

enum class Backend { io_context, cudf };
enum class Dest { host, device };

static Backend parse_backend(std::string_view s)
{
  if (s == "io_context") return Backend::io_context;
  if (s == "cudf") return Backend::cudf;
  throw std::invalid_argument(std::string("unknown backend: ") + std::string(s) +
                              "  (expected: io_context | cudf)");
}

static Dest parse_dest(std::string_view s)
{
  if (s == "host") return Dest::host;
  if (s == "device") return Dest::device;
  throw std::invalid_argument(std::string("unknown dest: ") + std::string(s) +
                              "  (expected: host | device)");
}

static bool drop_caches()
{
  ::sync();
  int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
  if (fd < 0) return false;
  bool ok = ::write(fd, "3", 1) == 1;
  ::close(fd);
  return ok;
}

// Expand @p spec to a sorted list of file paths.  If it contains '*', POSIX
// glob expansion is used; otherwise the spec is returned as a single path.
static std::vector<std::string> expand_paths(std::string const& spec)
{
  if (spec.find('*') == std::string::npos) return {spec};

  std::vector<std::string> paths;
  glob_t g{};
  int rc = ::glob(spec.c_str(), GLOB_TILDE, nullptr, &g);
  if (rc == 0) {
    paths.reserve(g.gl_pathc);
    for (size_t i = 0; i < g.gl_pathc; ++i)
      paths.emplace_back(g.gl_pathv[i]);
  }
  ::globfree(&g);
  std::sort(paths.begin(), paths.end());
  return paths;
}

static void usage(char const* prog)
{
  std::cerr << "usage: " << prog
            << " <path|glob> <io_context|cudf> <host|device> <num_rows> [n_threads]\n"
            << "  path|glob  – parquet file path, or shell glob (e.g. 'dir/*.parquet')\n"
            << "  io_context – native cucascade::io ioctx (host=vector-io, device=device_read)\n"
            << "  cudf       – cucascade::io::datasource per-range host_read/device_read_async\n"
            << "  host|device– destination tier the ranges are read into\n"
            << "  num_rows   – rows to read (0 = all)\n"
            << "  n_threads  – reader threads / uring reactors (default 2)\n";
}

// One coalesced column-chunk byte range in a specific file.
struct coalesced_range {
  size_t file_idx;
  size_t offset;
  size_t size;
};

int main(int argc, char** argv)
{
  if (argc < 5 || argc > 6) {
    usage(argv[0]);
    return 1;
  }

  std::string path_spec = argv[1];

  Backend backend;
  Dest dest;
  try {
    backend = parse_backend(argv[2]);
    dest    = parse_dest(argv[3]);
  } catch (std::invalid_argument const& e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  long long num_rows_arg = std::stoll(argv[4]);
  if (num_rows_arg < 0) {
    std::cerr << "num_rows must be >= 0\n";
    return 1;
  }
  size_t num_rows = static_cast<size_t>(num_rows_arg);  // 0 means all

  size_t n_threads = 2;
  if (argc == 6) {
    long long n_threads_arg = std::stoll(argv[5]);
    if (n_threads_arg <= 0) {
      std::cerr << "n_threads must be > 0\n";
      return 1;
    }
    n_threads = static_cast<size_t>(n_threads_arg);
  }

  auto paths = expand_paths(path_spec);
  if (paths.empty()) {
    std::cerr << "no files matched: " << path_spec << "\n";
    return 1;
  }

  std::cout << "Backend: " << argv[2] << "\n"
            << "Dest   : " << argv[3] << "\n"
            << "Files  : " << paths.size() << "\n";
  for (auto const& p : paths)
    std::cout << "  " << p << "\n";
  std::cout << "Rows   : " << (num_rows == 0 ? "all" : std::to_string(num_rows)) << "\n"
            << "Threads: " << n_threads << "\n"
            << "Columns: ";
  for (auto const& c : COLUMNS)
    std::cout << c << "  ";
  std::cout << "\n\n";

  bool can_drop = drop_caches();
  if (!can_drop) std::cout << "WARNING: cannot drop caches (run as root for cold results)\n\n";

  cudaFree(nullptr);

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(::cuda::mr::any_resource<::cuda::mr::device_accessible>{
    rmm::device_async_resource_ref{async_mr}});

  auto time_ms = [](auto fn) -> double {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
      .count();
  };

  auto scan_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();

  // Per-file (untimed): probe the footer with cudf's default datasource, then
  // run hybrid_scan to collect the byte ranges the selected columns touch.  The
  // num_rows budget is consumed in file order.  This block mirrors the metadata
  // scan in parquet_benchmark.cpp.
  std::vector<coalesced_range> ranges;
  size_t total_range_bytes = 0;
  int64_t accumulated_rows  = 0;

  for (size_t file_idx = 0; file_idx < paths.size(); ++file_idx) {
    auto const& path = paths[file_idx];
    std::vector<uint8_t> footer_buf;
    {
      auto probe_sources = cudf::io::make_datasources(cudf::io::source_info{{path}});
      auto& probe_ds     = *probe_sources.front();
      auto file_size     = probe_ds.size();
      cudf::io::parquet::file_ender_s ender{};
      probe_ds.host_read(
        file_size - sizeof(ender), sizeof(ender), reinterpret_cast<uint8_t*>(&ender));
      footer_buf.resize(ender.footer_len);
      probe_ds.host_read(
        file_size - sizeof(ender) - ender.footer_len, ender.footer_len, footer_buf.data());
    }

    cudf::io::parquet::experimental::hybrid_scan_reader scanner{
      cudf::host_span<uint8_t const>{footer_buf.data(), footer_buf.size()}, scan_opts};
    auto file_metadata = scanner.parquet_metadata();

    std::vector<cudf::size_type> selected_row_groups;
    if (num_rows == 0) {
      selected_row_groups = scanner.all_row_groups(scan_opts);
    } else if (accumulated_rows < static_cast<int64_t>(num_rows)) {
      for (cudf::size_type i = 0;
           i < static_cast<cudf::size_type>(file_metadata.row_groups.size()) &&
           accumulated_rows < static_cast<int64_t>(num_rows);
           ++i) {
        selected_row_groups.push_back(i);
        accumulated_rows += file_metadata.row_groups[static_cast<size_t>(i)].num_rows;
      }
    }

    auto byte_ranges = scanner.all_column_chunks_byte_ranges(selected_row_groups, scan_opts);
    std::sort(byte_ranges.begin(), byte_ranges.end(), [](auto const& a, auto const& b) {
      return a.offset() < b.offset();
    });
    std::vector<cudf::io::text::byte_range_info> coalesced;
    coalesced.reserve(byte_ranges.size());
    for (auto const& br : byte_ranges) {
      if (br.size() <= 0) continue;
      if (!coalesced.empty()) {
        auto& back       = coalesced.back();
        int64_t back_end = back.offset() + back.size();
        if (br.offset() <= back_end) {
          int64_t br_end = br.offset() + br.size();
          if (br_end > back_end)
            back = cudf::io::text::byte_range_info{back.offset(), br_end - back.offset()};
          continue;
        }
      }
      coalesced.push_back(br);
    }

    for (auto const& br : coalesced) {
      ranges.push_back(coalesced_range{
        file_idx, static_cast<size_t>(br.offset()), static_cast<size_t>(br.size())});
      total_range_bytes += static_cast<size_t>(br.size());
    }
  }

  if (ranges.empty()) {
    std::cerr << "hybrid scan produced no byte ranges\n";
    return 1;
  }

  std::cout << "Hybrid scan: " << ranges.size() << " byte range(s), " << std::fixed
            << std::setprecision(2) << static_cast<double>(total_range_bytes) / (1024.0 * 1024.0)
            << " MiB total\n\n";

  // -- Backend / staging setup (untimed) --------------------------------------

  // Host staging pool for the uring reactor's bounce slots.
  constexpr uint32_t POOL_MAX_SLABS = 20;
  constexpr size_t CHUNKS_PER_SLAB =
    cucascade::memory::fixed_size_host_memory_resource::default_pool_size;
  constexpr size_t POOL_CAPACITY =
    static_cast<size_t>(POOL_MAX_SLABS) * CHUNKS_PER_SLAB * (1 << 20);

  cucascade::memory::numa_region_pinned_host_memory_resource upstream(0, /*make_portable=*/true);
  cucascade::memory::fixed_size_host_memory_resource host_mr(0,                // device_id
                                                             upstream,         // upstream allocator
                                                             POOL_CAPACITY,    // mem_limit
                                                             POOL_CAPACITY,    // capacity
                                                             1 << 20,          // block_size = 1 MiB
                                                             CHUNKS_PER_SLAB,  // pool_size
                                                             1);               // initial_pools

  auto uring_ctx = std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
    cucascade::io::uring::uring_reactor::reactor_config_type{.bounce_size = host_mr.get_block_size(),
                                                             .use_odirect = true},
    &host_mr);
  std::shared_ptr<cucascade::io::ioctx> io_ctx =
    std::make_shared<cucascade::io::uring::uring_ioctx>(n_threads, std::move(uring_ctx));
  io_ctx->start();

  // One handle per file for the native path, one datasource per file for the
  // cudf path.  Opening does one untimed open()/stat() per file; both are
  // shared read-only across worker threads (no cache is wired, so per-scan
  // datasource state is inert).
  std::vector<std::shared_ptr<cucascade::io::io_object>> io_objects;
  std::vector<std::unique_ptr<cucascade::io::datasource>> datasources;
  if (backend == Backend::io_context) {
    io_objects.reserve(paths.size());
    for (auto const& path : paths)
      io_objects.push_back(io_ctx->open_io_object(path));
  } else {
    datasources.reserve(paths.size());
    for (auto const& path : paths)
      datasources.push_back(cucascade::io::open_datasource(io_ctx, path));
  }

  // Per-range destination buffers (untimed).  host → pinned; device → rmm.
  rmm::cuda_stream alloc_stream;
  std::vector<uint8_t*> dsts(ranges.size(), nullptr);
  std::vector<void*> host_bufs;             // owned pinned allocations (host dest)
  std::vector<rmm::device_buffer> dev_bufs;  // owned device allocations (device dest)
  if (dest == Dest::host) {
    host_bufs.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
      void* p = upstream.allocate_sync(ranges[i].size);
      host_bufs.push_back(p);
      dsts[i] = static_cast<uint8_t*>(p);
    }
  } else {
    dev_bufs.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
      dev_bufs.emplace_back(ranges[i].size, alloc_stream.view());
      dsts[i] = static_cast<uint8_t*>(dev_bufs[i].data());
    }
    alloc_stream.synchronize();
  }

  // Warm-up: give the reactors a moment to finish spinning up (matches the
  // settle delay used by parquet_benchmark before its timed run).
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));

  size_t const n_threads_eff = std::min(n_threads, ranges.size());
  std::cout << "Read   : " << n_threads_eff << " worker thread(s)\n\n";

  // -- Timed run --------------------------------------------------------------
  //
  // Partition the coalesced ranges into contiguous [lo, hi) slices, one per
  // worker.  Each worker issues its reads (collecting futures) then blocks on
  // them at the end, so the region measures overlapped in-flight I/O.

  double ms = time_ms([&] {
    std::vector<std::thread> workers;
    workers.reserve(n_threads_eff);
    size_t const base = ranges.size() / n_threads_eff;
    size_t const rem  = ranges.size() % n_threads_eff;

    size_t lo = 0;
    for (size_t t = 0; t < n_threads_eff; ++t) {
      size_t const count = base + (t < rem ? 1 : 0);
      size_t const hi    = lo + count;
      workers.emplace_back([&, lo, hi] {
        rmm::cuda_stream stream;  // per-worker stream for device reads

        if (backend == Backend::io_context && dest == Dest::host) {
          // Vector I/O: one host_read_ranges_async_io per file, over that
          // file's segments.  Segment vectors must outlive the futures.
          std::vector<std::vector<cucascade::io::io_object_segment>> seg_sets;
          std::vector<cucascade::exec::semi_future<size_t>> futs;
          size_t cur_file = SIZE_MAX;
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            if (r.file_idx != cur_file) {
              seg_sets.emplace_back();
              cur_file = r.file_idx;
            }
            seg_sets.back().push_back(
              cucascade::io::io_object_segment{r.offset, r.size, dsts[i]});
          }
          // Re-walk to bind each segment set to its io_object and dispatch.
          size_t set = 0;
          cur_file   = SIZE_MAX;
          for (size_t i = lo; i < hi; ++i) {
            if (ranges[i].file_idx != cur_file) {
              cur_file  = ranges[i].file_idx;
              auto& seg = seg_sets[set++];
              futs.push_back(io_ctx->host_read_ranges_async_io(
                *io_objects[cur_file], std::span<cucascade::io::io_object_segment>(seg)));
            }
          }
          for (auto& f : futs)
            std::move(f).get();
        } else if (backend == Backend::io_context && dest == Dest::device) {
          std::vector<cucascade::exec::semi_future<size_t>> futs;
          futs.reserve(hi - lo);
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            futs.push_back(io_ctx->device_read_async(
              *io_objects[r.file_idx], r.offset, r.size, dsts[i], stream.view()));
          }
          for (auto& f : futs)
            std::move(f).get();
          stream.synchronize();
        } else if (backend == Backend::cudf && dest == Dest::host) {
          std::vector<std::future<size_t>> futs;
          futs.reserve(hi - lo);
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            futs.push_back(datasources[r.file_idx]->host_read_async(r.offset, r.size, dsts[i]));
          }
          for (auto& f : futs)
            f.get();
        } else {  // cudf + device
          std::vector<std::future<size_t>> futs;
          futs.reserve(hi - lo);
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            futs.push_back(
              datasources[r.file_idx]->device_read_async(r.offset, r.size, dsts[i], stream.view()));
          }
          for (auto& f : futs)
            f.get();
          stream.synchronize();
        }
      });
      lo = hi;
    }
    for (auto& w : workers)
      w.join();
  });

  double const gib   = static_cast<double>(total_range_bytes) / (1024.0 * 1024.0 * 1024.0);
  double const gib_s = gib / (ms / 1000.0);
  std::cout << std::fixed << std::setprecision(1) << ms << " ms  " << std::setprecision(2) << gib_s
            << " GiB/s\n";

  // Release pinned host allocations (device buffers free via rmm on scope exit).
  for (size_t i = 0; i < host_bufs.size(); ++i)
    upstream.deallocate_sync(host_bufs[i], ranges[i].size);

  io_ctx->shutdown();
  return 0;
}
