/*
 * Copyright 2025, cuCascade Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_cufile_compare.cpp
 * @brief Two read-path comparisons on real storage, hidden behind `[.cufile-compare]`.
 *
 * Every submitter repeats one group of 8 device reads of 1 MiB each (unaligned offsets, an
 * unregistered RMM destination) until the time is up.  The group is either *scattered* (eight
 * random offsets) or *adjacent* (one run of eight consecutive chunks, as the column chunks of
 * a Parquet row group are).
 *
 *   A. How the group is issued to a cuFile / GDS pool: eight single reads one after the other,
 *      eight single reads issued together, one vectored read, and one vectored read with
 *      fusing on (`config::fuse_gap_bytes`).
 *   B. Which path carries the bytes: the cuFile reactor's POSIX fallback (`compat::on`: a
 *      buffered pread into pinned host memory, then a host-to-device copy), the io_uring
 *      reactor reading to the device, buffered and with O_DIRECT, and GDS as the reference.
 *
 * Environment:
 *   CUCASCADE_IO_COMPARE_FILE        required: a large file on the storage under test
 *   CUCASCADE_IO_COMPARE_GPU         CUDA device (default 0)
 *   CUCASCADE_IO_COMPARE_SECONDS     seconds per row (default 4)
 *   CUCASCADE_IO_COMPARE_SUBMITTERS  submitting threads (default 8)
 *   CUCASCADE_IO_COMPARE_REACTORS    cuFile reactors for the GPU (default 32)
 *   CUCASCADE_IO_COMPARE_RANGE_MIB   "begin,end": the part of the file to read (default all)
 *   CUCASCADE_IO_COMPARE_SKIP        "A" or "B" to skip a comparison
 *
 * Results, 2026-11-15, one NVIDIA GB200 (aarch64, 144 cores), 8 submitters, 32 cuFile
 * reactors, 4 s per row, libcufile 1.17.  "groups/s" counts groups of eight reads; the
 * latencies are per group.
 *
 * A, on IBM Storage Scale over RoCE with GDS in DMA-BUF mode (two 400 Gb/s ports local to
 * the GPU, 100 GB/s), a 4.3 GB Parquet file:
 *
 *   layout    issued as                  GB/s  groups/s  mean ms  p99 ms
 *   scattered 8 single reads, in turn    31.4      3746     2.13    2.49
 *   scattered 8 single reads, together   74.7      8903     0.90    1.48
 *   scattered one vectored read          70.7      8423     0.95    1.55
 *   scattered one vectored read, fused   68.8      8199     0.98    1.61
 *   adjacent  8 single reads, in turn    32.4      3864     2.07    2.32
 *   adjacent  8 single reads, together   73.1      8718     0.92    1.58
 *   adjacent  one vectored read          69.2      8247     0.97    1.63
 *   adjacent  one vectored read, fused   43.5      5186     1.54    2.01
 *
 *   - Issuing the eight reads one after the other leaves the pool idle: 8 reads in flight
 *     for 32 reactors.  Issued together or as one vectored read they reach 70-75 GB/s; the
 *     vectored read is about 5% behind eight single reads issued together.
 *   - Fusing does nothing for scattered chunks (nothing to fuse) and costs 37% for adjacent
 *     1 MiB chunks: the group becomes one 8 MiB read on one reactor, so 8 reads are in
 *     flight where 64 were.  Fusing pays when the chunks are small and the reactors are the
 *     bottleneck (the tuning sweep measured 16 -> 54 GB/s for 16-200 KiB chunks), not when
 *     the parts are already large enough to fill a reactor.
 *
 * B, on a local NVMe drive (ext4, an 8 GiB file).  The drive, not the path, bounds every
 * row that reads it directly; the buffered rows are fed from the page cache after the first
 * pass and so measure the host path alone (copy into pinned memory, host-to-device copy):
 *
 *   path                              layout     GB/s  groups/s  mean ms  p99 ms
 *   cuFile reactor, buffered + H2D    scattered  52.7      6280     1.27   10.93
 *   cuFile reactor, buffered + H2D    adjacent   80.8      9627     0.83    1.49
 *   io_uring buffered, 4 reactors     scattered  36.6      4362     1.83    5.97
 *   io_uring buffered, 4 reactors     adjacent   39.5      4707     1.70    5.20
 *   io_uring buffered, 16 reactors    scattered  63.7      7593     1.05    1.45
 *   io_uring buffered, 16 reactors    adjacent   64.0      7626     1.05    1.46
 *   io_uring O_DIRECT, 4 reactors     either      5.15      614    13.0    18.7
 *   io_uring O_DIRECT, 16 reactors    either      5.15      614    13.0    13.9
 *   cuFile reactor, cuFileRead        either      5.15      614    13.0    18.0
 *
 *   - Per thread, io_uring's host path is four to five times the cuFile reactor's POSIX
 *     fallback: 9-10 GB/s per reactor against 1.6-2.5 GB/s.  The fallback keeps up only by
 *     running 32 threads.
 *   - Read directly, the drive gives 5.15 GB/s whichever path asks, with 4 or 16 reactors.
 *
 * B could not be measured on Storage Scale that day: ordinary direct reads from the client
 * were stalling for seconds (io_uring reached 0.01-0.06 GB/s, and the buffered fallback was
 * served from the client's page pool), while GDS reads of the same file ran at 70 GB/s.
 */

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace cufile = cucascade::io::cufile;
namespace uring  = cucascade::io::uring;

namespace {

using namespace std::chrono_literals;
using clock_type = std::chrono::steady_clock;

constexpr std::size_t chunk_bytes = 1UL << 20;
constexpr std::size_t group_size  = 8;

[[nodiscard]] std::vector<std::size_t> env_list(char const* name, std::vector<std::size_t> fallback)
{
  char const* text = std::getenv(name);
  if (text == nullptr || *text == '\0') return fallback;
  std::vector<std::size_t> out;
  std::stringstream stream{text};
  for (std::string item; std::getline(stream, item, ',');) {
    if (!item.empty()) out.push_back(std::stoull(item));
  }
  return out.empty() ? fallback : out;
}

[[nodiscard]] bool skipped(char which)
{
  char const* text = std::getenv("CUCASCADE_IO_COMPARE_SKIP");
  return text != nullptr && std::string{text}.find(which) != std::string::npos;
}

class rng {
 public:
  explicit rng(std::uint64_t seed) : _state(seed | 1) {}
  std::uint64_t operator()() noexcept
  {
    _state ^= _state << 13;
    _state ^= _state >> 7;
    _state ^= _state << 17;
    return _state;
  }

 private:
  std::uint64_t _state;
};

enum class layout : std::uint8_t { scattered, adjacent };
enum class issue : std::uint8_t { one_by_one, together, vectored };

struct setup {
  std::string path;
  int gpu{0};
  std::chrono::duration<double> seconds{4.0};
  std::size_t submitters{8};
  std::size_t reactors{32};
  std::size_t range_begin{0};
  std::size_t range_end{0};
};

struct outcome {
  double gbps{0};
  double groups_per_second{0};
  double mean_ms{0};
  double p99_ms{0};
};

/// Each submitter reads groups of eight 1 MiB chunks through @p ioctx until the time is up.
[[nodiscard]] outcome run_groups(cucascade::io::ioctx& ioctx,
                                 cucascade::io::io_object const& object,
                                 setup const& s,
                                 layout where,
                                 issue how)
{
  std::atomic<bool> failed{false};
  std::string failure;
  std::mutex mutex;
  std::vector<double> latencies;
  std::atomic<std::size_t> bytes{0};
  std::vector<std::thread> threads;
  auto const start    = clock_type::now();
  auto const deadline = start + std::chrono::duration_cast<clock_type::duration>(s.seconds);
  for (std::size_t t = 0; t < s.submitters; ++t) {
    threads.emplace_back([&, t] {
      std::vector<double> mine;
      try {
        rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{s.gpu}};
        rmm::cuda_stream stream;
        constexpr std::size_t stride = chunk_bytes + 512;
        rmm::device_buffer buffer(group_size * stride, stream);
        stream.synchronize();
        auto* const base = static_cast<std::uint8_t*>(buffer.data());
        ::cuda::stream_ref const on_stream{stream.value()};
        rng next{0x9E3779B97F4A7C15ULL * (t + 1)};
        auto const span = s.range_end - s.range_begin - group_size * chunk_bytes;
        while (clock_type::now() < deadline && !failed.load()) {
          std::vector<cucascade::io::slice> slices;
          auto const run = s.range_begin + next() % span;
          for (std::size_t i = 0; i < group_size; ++i) {
            auto const offset =
              where == layout::adjacent ? run + i * chunk_bytes : s.range_begin + next() % span;
            slices.emplace_back(offset, chunk_bytes, base + i * stride + (next() & 1));
          }
          auto const begun = clock_type::now();
          std::size_t got  = 0;
          switch (how) {
            case issue::one_by_one:
              for (auto const& slice : slices) {
                got += ioctx
                         .device_read_async_io(
                           object, slice.offset(), slice.size(), slice.dst, on_stream)
                         .get(600s);
              }
              break;
            case issue::together: {
              std::vector<cucascade::exec::semi_future<std::size_t>> pending;
              pending.reserve(slices.size());
              for (auto const& slice : slices) {
                pending.push_back(ioctx.device_read_async_io(
                  object, slice.offset(), slice.size(), slice.dst, on_stream));
              }
              for (auto& future : pending) {
                got += std::move(future).get(600s);
              }
              break;
            }
            case issue::vectored:
              got = ioctx.device_readv_async_io(object, slices, on_stream).get(600s);
              break;
          }
          if (got != group_size * chunk_bytes) throw std::runtime_error("short read");
          mine.push_back(
            std::chrono::duration<double, std::milli>(clock_type::now() - begun).count());
          bytes.fetch_add(got);
        }
      } catch (std::exception const& e) {
        failed = true;
        std::lock_guard lock(mutex);
        failure = e.what();
      }
      std::lock_guard lock(mutex);
      latencies.insert(latencies.end(), mine.begin(), mine.end());
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  if (failed) throw std::runtime_error(failure);
  std::chrono::duration<double> const elapsed = clock_type::now() - start;
  outcome out;
  if (latencies.empty()) return out;
  std::sort(latencies.begin(), latencies.end());
  double total = 0;
  for (auto const value : latencies) {
    total += value;
  }
  out.gbps              = static_cast<double>(bytes.load()) / 1e9 / elapsed.count();
  out.groups_per_second = static_cast<double>(latencies.size()) / elapsed.count();
  out.mean_ms           = total / static_cast<double>(latencies.size());
  out.p99_ms            = latencies[latencies.size() * 99 / 100];
  return out;
}

[[nodiscard]] char const* name_of(layout where)
{
  return where == layout::adjacent ? "adjacent" : "scattered";
}

void print_row(char const* path, layout where, char const* how, outcome const& o)
{
  std::printf("%-34s %-9s %-26s %7.2f %9.0f %8.2f %8.2f\n",
              path,
              name_of(where),
              how,
              o.gbps,
              o.groups_per_second,
              o.mean_ms,
              o.p99_ms);
  std::fflush(stdout);
}

void print_header(char const* title)
{
  std::printf("\n== %s\n%-34s %-9s %-26s %7s %9s %8s %8s\n",
              title,
              "path",
              "layout",
              "issued as",
              "GB/s",
              "groups/s",
              "mean ms",
              "p99 ms");
}

/// A started cuFile pool for one GPU; its staging is released on scope exit.
class cufile_pool {
 public:
  cufile_pool(cufile::config cfg, setup const& s) : _gpu(s.gpu)
  {
    cfg.devices             = {s.gpu};
    cfg.reactors_per_device = s.reactors;
    cfg.storage_hint_path   = s.path;
    _ioctx                  = cufile::make_cufile_ioctx(0, cfg);
    _ioctx->start();
  }
  ~cufile_pool()
  {
    _ioctx->shutdown();
    _ioctx.reset();
    std::ignore = cufile::release_staging(_gpu);
  }
  cufile_pool(cufile_pool const&)            = delete;
  cufile_pool& operator=(cufile_pool const&) = delete;

  [[nodiscard]] cufile::cufile_ioctx& ioctx() noexcept { return *_ioctx; }

 private:
  int _gpu;
  std::shared_ptr<cufile::cufile_ioctx> _ioctx;
};

/// A started io_uring pool with its own pinned staging (128 MiB per reactor).
class uring_pool {
 public:
  uring_pool(std::size_t n_reactors, bool odirect)
    : _mr{0, _upstream, n_reactors * (128UL << 20), n_reactors * (128UL << 20), 1UL << 20, 64, 0},
      _ioctx{n_reactors,
             std::make_shared<uring::uring_reactor::reactor_context>(config_for(odirect), &_mr)}
  {
    _ioctx.start();
  }
  ~uring_pool() { _ioctx.shutdown(); }
  uring_pool(uring_pool const&)            = delete;
  uring_pool& operator=(uring_pool const&) = delete;

  [[nodiscard]] uring::uring_ioctx& ioctx() noexcept { return _ioctx; }

 private:
  [[nodiscard]] static uring::config config_for(bool odirect)
  {
    uring::config cfg;
    cfg.use_odirect = odirect;
    return cfg;
  }

  cucascade::memory::numa_region_pinned_host_memory_resource _upstream{0};
  cucascade::memory::fixed_size_host_memory_resource _mr;
  uring::uring_ioctx _ioctx;
};

/// The run's parameters, or nothing when the comparison cannot run here.
[[nodiscard]] std::optional<setup> read_setup()
{
  char const* path = std::getenv("CUCASCADE_IO_COMPARE_FILE");
  int devices      = 0;
  if (path == nullptr || cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    return std::nullopt;
  }
  setup s;
  s.path    = path;
  s.gpu     = static_cast<int>(env_list("CUCASCADE_IO_COMPARE_GPU", {0}).front());
  s.seconds = std::chrono::duration<double>(
    static_cast<double>(env_list("CUCASCADE_IO_COMPARE_SECONDS", {4}).front()));
  s.submitters = env_list("CUCASCADE_IO_COMPARE_SUBMITTERS", {8}).front();
  s.reactors   = env_list("CUCASCADE_IO_COMPARE_REACTORS", {32}).front();
  return s;
}

/// Clamp the range to the file, which must hold a few groups.
void set_range(setup& s, std::size_t file_size)
{
  auto const range = env_list("CUCASCADE_IO_COMPARE_RANGE_MIB", {0, file_size >> 20});
  s.range_begin    = std::min(range.front() << 20, file_size);
  s.range_end      = std::min(range.back() << 20, file_size);
  if (s.range_end < s.range_begin + 4 * group_size * chunk_bytes) {
    throw std::runtime_error("the file (or CUCASCADE_IO_COMPARE_RANGE_MIB) is too small");
  }
}

}  // namespace

TEST_CASE("cufile: eight 1 MiB reads issued singly or as one vectored read", "[.cufile-compare]")
{
  auto s = read_setup();
  if (!s || skipped('A')) SKIP("set CUCASCADE_IO_COMPARE_FILE (and have a GPU) to run");
  rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{s->gpu}};

  print_header("A. cuFile / GDS: how a group of eight 1 MiB reads is issued");
  for (bool const fuse : {false, true}) {
    cufile::config cfg;
    cfg.gds_threshold  = 0;
    cfg.fuse_gap_bytes = fuse ? 1UL << 20 : 0;
    cufile_pool pool{cfg, *s};
    auto object = pool.ioctx().open_io_object(s->path);
    set_range(*s, object->size());
    for (auto const where : {layout::scattered, layout::adjacent}) {
      if (fuse) {
        print_row("GDS staged",
                  where,
                  "one vectored read, fused",
                  run_groups(pool.ioctx(), *object, *s, where, issue::vectored));
        continue;
      }
      print_row("GDS staged",
                where,
                "8 single reads, in turn",
                run_groups(pool.ioctx(), *object, *s, where, issue::one_by_one));
      print_row("GDS staged",
                where,
                "8 single reads, together",
                run_groups(pool.ioctx(), *object, *s, where, issue::together));
      print_row("GDS staged",
                where,
                "one vectored read",
                run_groups(pool.ioctx(), *object, *s, where, issue::vectored));
    }
    auto const stats = pool.ioctx().stats();
    std::printf("  (fuse %s: %llu staged reads, %llu fused slices, %llu cuFile failures)\n",
                fuse ? "on" : "off",
                static_cast<unsigned long long>(stats.staged_reads),
                static_cast<unsigned long long>(stats.fused_slices),
                static_cast<unsigned long long>(stats.cufile_failures));
  }
}

TEST_CASE("cufile: host bounce paths to the device against io_uring and GDS", "[.cufile-compare]")
{
  auto s = read_setup();
  if (!s || skipped('B')) SKIP("set CUCASCADE_IO_COMPARE_FILE (and have a GPU) to run");
  rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{s->gpu}};

  print_header("B. which path carries a vectored read of eight 1 MiB chunks to the GPU");
  auto const both = {layout::scattered, layout::adjacent};
  {
    cufile::config cfg;
    cfg.compat_mode = cufile::compat::on;
    cufile_pool pool{cfg, *s};
    auto object = pool.ioctx().open_io_object(s->path);
    set_range(*s, object->size());
    for (auto const where : both) {
      print_row("cuFile reactor, buffered + H2D",
                where,
                "one vectored read",
                run_groups(pool.ioctx(), *object, *s, where, issue::vectored));
    }
  }
  for (bool const odirect : {false, true}) {
    for (std::size_t const reactors : {std::size_t{4}, std::size_t{16}}) {
      uring_pool pool{reactors, odirect};
      auto object      = pool.ioctx().open_io_object(s->path);
      auto const label = std::string{"io_uring "} + (odirect ? "O_DIRECT" : "buffered") + ", " +
                         std::to_string(reactors) + " reactors";
      for (auto const where : both) {
        print_row(label.c_str(),
                  where,
                  "one vectored read",
                  run_groups(pool.ioctx(), *object, *s, where, issue::vectored));
      }
    }
  }
  {
    cufile::config cfg;
    cfg.gds_threshold = 0;
    cufile_pool pool{cfg, *s};
    auto object = pool.ioctx().open_io_object(s->path);
    for (auto const where : both) {
      print_row("cuFile reactor, cuFileRead",
                where,
                "one vectored read",
                run_groups(pool.ioctx(), *object, *s, where, issue::vectored));
    }
  }
}
