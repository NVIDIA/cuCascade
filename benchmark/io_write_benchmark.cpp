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

// Standalone local-file write benchmark for the cucascade::io write API.
//
// mode=write  : sequentially writes `size` bytes in `block`-sized requests
//               (host_write_async / device_write_async) through a uring or
//               kvikio ioctx, keeping up to `qd` requests in flight per
//               submitting thread, then flush_async()es the file.  Reports
//               GB/s with and without the final flush, next to a built-in
//               dd-equivalent pwrite baseline (1 thread) and a parallel pwrite
//               baseline (`baseline_threads`).
// mode=mixed  : measures small-read latency (host_read_async of `read_size`
//               at random aligned offsets of a separate, cache-evicted file)
//               first on an idle context, then while the write workload of
//               mode=write runs on the same context.  Reports p50/p99/max and
//               the concurrent write throughput.  A pread/pwrite baseline of the
//               same experiment is reported for reference.
// mode=readmix: demand-read latency while a large background read runs on the
//               same context -- the prefetch-isolation scenario.  One vectored
//               read of `bg_size` in `bg_slice` slices, submitted with class
//               `bg_class` (background = how prefetching_cache::prefetch tags
//               its reads; read = the pre-tagging classification of a
//               prefetch), is timed alone, then concurrently with a loop of
//               host_read_async (demand_dst=host) or device_read_async
//               (demand_dst=device: staged through pinned slots, so a 16 MiB
//               read is one 16-slot op) demand reads (automatic class: latency
//               for <= 256 KiB, read above -- never background) for each of
//               `demand_sizes`.  Reports idle and concurrent p50/p99/max, the
//               concurrent background GB/s and per-window ctx.stats() (first-I/O
//               delay per class, per-runner peak in-flight ops) as key=value
//               `RESULT` / `STATS` lines.
//
// The uring scheduling knobs (slices_per_pass, bg_groups, bg_share,
// bg_reserve) apply to every mode.  All arguments are key=value; run with
// `help` for the list.

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <kvikio/defaults.hpp>

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

namespace io = cucascade::io;

constexpr std::size_t KiB = 1024ULL;
constexpr std::size_t MiB = 1024ULL * KiB;
constexpr std::size_t GiB = 1024ULL * MiB;

constexpr std::size_t PAGE = 4096;

using clock_type = std::chrono::steady_clock;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

struct options {
  std::string mode{"write"};     // write | mixed | readmix
  std::string backend{"uring"};  // uring | kvikio
  std::string source{"host"};    // host | device
  std::size_t size{4 * GiB};     // bytes written per repetition
  std::size_t block{16 * MiB};   // bytes per write request
  std::size_t runners{2};        // uring runner threads / kvikio pool threads
  std::size_t threads{1};        // submitting threads
  std::size_t qd{32};            // in-flight requests per submitting thread
  bool odirect{true};
  io::write_durability durability{io::write_durability::none};
  std::string path;  // directory for the test files
  std::size_t reps{3};
  bool baseline{true};
  std::size_t baseline_threads{4};
  std::size_t pause_ms{0};  // idle time between measurements
  // mixed mode (read_file_size, idle_reads also readmix)
  std::size_t read_size{4 * KiB};
  std::size_t read_file_size{1 * GiB};
  std::size_t idle_reads{2000};
  // uring scheduling knobs (all modes); defaults are the library defaults
  std::size_t slices_per_pass{io::uring::config{}.slices_per_pass};
  io::detail::scheduling_config scheduling{};
  // readmix mode
  std::size_t bg_size{2 * GiB};
  std::size_t bg_slice{1 * MiB};
  io::request_class bg_class{io::request_class::background};
  std::vector<std::size_t> demand_sizes{4 * KiB, 1 * MiB, 16 * MiB};
  std::string demand_dst{"host"};  // host | device
};

std::size_t parse_size(std::string_view s)
{
  if (s.empty()) { throw std::invalid_argument("empty size"); }
  std::size_t mult = 1;
  switch (s.back()) {
    case 'k':
    case 'K': mult = KiB; break;
    case 'm':
    case 'M': mult = MiB; break;
    case 'g':
    case 'G': mult = GiB; break;
    default: break;
  }
  if (mult != 1) { s.remove_suffix(1); }
  return static_cast<std::size_t>(std::stoull(std::string(s))) * mult;
}

/// Comma-separated list of sizes, e.g. "4K,1M,16M".
std::vector<std::size_t> parse_size_list(std::string_view s)
{
  std::vector<std::size_t> out;
  while (!s.empty()) {
    auto const comma = s.find(',');
    out.push_back(parse_size(s.substr(0, comma)));
    s = comma == std::string_view::npos ? std::string_view{} : s.substr(comma + 1);
  }
  return out;
}

bool parse_bool(std::string_view s)
{
  if (s == "1" || s == "on" || s == "true") { return true; }
  if (s == "0" || s == "off" || s == "false") { return false; }
  throw std::invalid_argument("expected 0|1, got " + std::string(s));
}

/// Short label of a byte count: "4K", "16M", "2G" when exact, else bytes.
std::string size_label(std::size_t bytes)
{
  if (bytes != 0 && bytes % GiB == 0) { return std::to_string(bytes / GiB) + "G"; }
  if (bytes != 0 && bytes % MiB == 0) { return std::to_string(bytes / MiB) + "M"; }
  if (bytes != 0 && bytes % KiB == 0) { return std::to_string(bytes / KiB) + "K"; }
  return std::to_string(bytes);
}

char const* class_name(io::request_class cls)
{
  switch (cls) {
    case io::request_class::latency: return "latency";
    case io::request_class::read: return "read";
    case io::request_class::write: return "write";
    case io::request_class::background: return "background";
    case io::request_class::automatic: break;
  }
  return "automatic";
}

void usage(char const* prog)
{
  std::cerr
    << "usage: " << prog << " [key=value ...]\n"
    << "  mode=write|mixed|readmix   (default write)\n"
    << "  backend=uring|kvikio       (default uring)\n"
    << "  source=host|device         write source memory (default host)\n"
    << "  size=<bytes>[K|M|G]        bytes written per repetition (default 4G)\n"
    << "  block=<bytes>[K|M|G]       bytes per write request / segment (default 16M)\n"
    << "  runners=N                  uring runner threads | kvikio pool threads (default 2)\n"
    << "  threads=N                  submitting threads (default 1)\n"
    << "  qd=N                       in-flight requests per submitting thread (default 32)\n"
    << "  odirect=0|1                O_DIRECT (uring use_odirect | kvikio auto_direct_io_write)\n"
    << "  durability=none|data_sync  per-request durability (default none)\n"
    << "  path=<dir>                 test file directory (default $TMPDIR or /tmp)\n"
    << "  reps=N                     repetitions (default 3)\n"
    << "  baseline=0|1               also run the pwrite baselines (default 1)\n"
    << "  baseline_threads=N         threads of the parallel pwrite baseline (default 4)\n"
    << "  pause_ms=N                 idle time between measurements (default 0)\n"
    << "  read_size=<bytes>          mixed: bytes per latency read (default 4K)\n"
    << "  read_file_size=<bytes>     mixed/readmix: size of the file read from (default 1G)\n"
    << "  idle_reads=N               mixed/readmix: reads in an idle phase (default 2000)\n"
    << "  slices_per_pass=N          uring: slices of one request planned per runner pass\n"
    << "                             (default 8, 0 = no cap)\n"
    << "  bg_groups=N                uring: scheduling.max_background_groups (default 2)\n"
    << "  bg_share=F                 uring: scheduling.background_slot_fraction (default 0.75)\n"
    << "  bg_reserve=N               uring: scheduling.reserved_background_slots (default 8)\n"
    << "  bg_size=<bytes>            readmix: background read size (default 2G)\n"
    << "  bg_slice=<bytes>           readmix: slice size of the background read (default 1M)\n"
    << "  bg_class=background|read   readmix: class of the background read (default\n"
    << "                             background; read = pre-tagging prefetch class)\n"
    << "  demand_sizes=<list>        readmix: demand read sizes (default 4K,1M,16M)\n"
    << "  demand_dst=host|device     readmix: demand read destination (default host)\n";
}

options parse_args(int argc, char** argv)
{
  options o;
  char const* tmp = std::getenv("TMPDIR");
  o.path          = tmp != nullptr && *tmp != '\0' ? tmp : "/tmp";
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    if (arg == "help" || arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    }
    auto const eq = arg.find('=');
    if (eq == std::string_view::npos) {
      throw std::invalid_argument("expected key=value, got " + std::string(arg));
    }
    auto const key = arg.substr(0, eq);
    auto const val = arg.substr(eq + 1);
    if (key == "mode") {
      o.mode = val;
    } else if (key == "backend") {
      o.backend = val;
    } else if (key == "source") {
      o.source = val;
    } else if (key == "size") {
      o.size = parse_size(val);
    } else if (key == "block") {
      o.block = parse_size(val);
    } else if (key == "runners") {
      o.runners = parse_size(val);
    } else if (key == "threads") {
      o.threads = parse_size(val);
    } else if (key == "qd") {
      o.qd = parse_size(val);
    } else if (key == "odirect") {
      o.odirect = parse_bool(val);
    } else if (key == "durability") {
      if (val == "none") {
        o.durability = io::write_durability::none;
      } else if (val == "data_sync") {
        o.durability = io::write_durability::data_sync;
      } else {
        throw std::invalid_argument("durability must be none|data_sync");
      }
    } else if (key == "path") {
      o.path = val;
    } else if (key == "reps") {
      o.reps = parse_size(val);
    } else if (key == "baseline") {
      o.baseline = parse_bool(val);
    } else if (key == "baseline_threads") {
      o.baseline_threads = parse_size(val);
    } else if (key == "pause_ms") {
      o.pause_ms = parse_size(val);
    } else if (key == "read_size") {
      o.read_size = parse_size(val);
    } else if (key == "read_file_size") {
      o.read_file_size = parse_size(val);
    } else if (key == "idle_reads") {
      o.idle_reads = parse_size(val);
    } else if (key == "slices_per_pass") {
      o.slices_per_pass = parse_size(val);
    } else if (key == "bg_groups") {
      o.scheduling.max_background_groups = parse_size(val);
    } else if (key == "bg_share") {
      o.scheduling.background_slot_fraction = std::stod(std::string(val));
    } else if (key == "bg_reserve") {
      o.scheduling.reserved_background_slots = parse_size(val);
    } else if (key == "bg_size") {
      o.bg_size = parse_size(val);
    } else if (key == "bg_slice") {
      o.bg_slice = parse_size(val);
    } else if (key == "bg_class") {
      if (val == "background") {
        o.bg_class = io::request_class::background;
      } else if (val == "read") {
        o.bg_class = io::request_class::read;
      } else {
        throw std::invalid_argument("bg_class must be background|read");
      }
    } else if (key == "demand_sizes") {
      o.demand_sizes = parse_size_list(val);
    } else if (key == "demand_dst") {
      o.demand_dst = val;
    } else {
      throw std::invalid_argument("unknown key: " + std::string(key));
    }
  }
  if (o.mode != "write" && o.mode != "mixed" && o.mode != "readmix") {
    throw std::invalid_argument("mode=write|mixed|readmix");
  }
  if (o.backend != "uring" && o.backend != "kvikio") {
    throw std::invalid_argument("backend=uring|kvikio");
  }
  if (o.source != "host" && o.source != "device") {
    throw std::invalid_argument("source=host|device");
  }
  if (o.size == 0 || o.block == 0 || o.runners == 0 || o.threads == 0 || o.qd == 0 || o.reps == 0 ||
      o.baseline_threads == 0) {
    throw std::invalid_argument("size, block, runners, threads, qd, reps must be > 0");
  }
  if (o.block % PAGE != 0) { throw std::invalid_argument("block must be a multiple of 4 KiB"); }
  if (o.read_size == 0 || o.read_size % PAGE != 0 || o.read_file_size < o.read_size) {
    throw std::invalid_argument("read_size must be a non-zero multiple of 4 KiB <= read_file_size");
  }
  if (o.mode == "readmix") {
    if (o.bg_slice == 0 || o.bg_slice % PAGE != 0 || o.bg_size < o.bg_slice) {
      throw std::invalid_argument("bg_slice must be a non-zero multiple of 4 KiB <= bg_size");
    }
    if (o.demand_sizes.empty()) { throw std::invalid_argument("demand_sizes is empty"); }
    if (o.demand_dst != "host" && o.demand_dst != "device") {
      throw std::invalid_argument("demand_dst=host|device");
    }
    for (auto const s : o.demand_sizes) {
      if (s == 0 || s % PAGE != 0 || s > o.read_file_size) {
        throw std::invalid_argument(
          "demand_sizes must be non-zero multiples of 4 KiB <= read_file_size");
      }
    }
  }
  return o;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

double gbps(std::size_t bytes, clock_type::duration d)
{
  return static_cast<double>(bytes) / std::chrono::duration<double>(d).count() / 1e9;
}

double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

/// Page-aligned host buffer filled with pseudo-random bytes (or zeroes, which
/// still faults every page in up front: read destinations).
struct aligned_buffer {
  explicit aligned_buffer(std::size_t bytes, bool random_fill = true)
    : _size((bytes + PAGE - 1) / PAGE * PAGE),
      _data(static_cast<std::uint8_t*>(std::aligned_alloc(PAGE, _size)), &std::free)
  {
    if (_data == nullptr) { throw std::bad_alloc(); }
    if (!random_fill) {
      std::memset(_data.get(), 0, _size);
      return;
    }
    std::mt19937_64 rng{42};
    for (std::size_t i = 0; i + sizeof(std::uint64_t) <= _size; i += sizeof(std::uint64_t)) {
      auto const v = rng();
      std::memcpy(_data.get() + i, &v, sizeof(v));
    }
  }
  [[nodiscard]] std::uint8_t* data() const noexcept { return _data.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

 private:
  std::size_t _size;
  std::unique_ptr<std::uint8_t, decltype(&std::free)> _data;
};

void check_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

/// Device copy of the host source buffer (modest: one block).
struct device_buffer {
  /// Uninitialised device memory (a read destination).
  explicit device_buffer(std::size_t bytes) : _size(bytes)
  {
    check_cuda(cudaMalloc(&_ptr, _size), "cudaMalloc");
  }
  explicit device_buffer(aligned_buffer const& src) : _size(src.size())
  {
    check_cuda(cudaMalloc(&_ptr, _size), "cudaMalloc (try source=host if the GPU is full)");
    check_cuda(cudaMemcpy(_ptr, src.data(), _size, cudaMemcpyHostToDevice), "cudaMemcpy");
  }
  ~device_buffer() { static_cast<void>(cudaFree(_ptr)); }
  device_buffer(device_buffer const&)            = delete;
  device_buffer& operator=(device_buffer const&) = delete;
  [[nodiscard]] std::uint8_t* data() const noexcept { return static_cast<std::uint8_t*>(_ptr); }

 private:
  void* _ptr{nullptr};
  std::size_t _size;
};

struct cuda_stream {
  cuda_stream() { check_cuda(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream"); }
  ~cuda_stream() { static_cast<void>(cudaStreamDestroy(s)); }
  cuda_stream(cuda_stream const&)            = delete;
  cuda_stream& operator=(cuda_stream const&) = delete;
  cudaStream_t s{};
};

std::string test_file(options const& o, std::string_view tag)
{
  return (std::filesystem::path(o.path) / ("cucascade_io_write_bench_" + std::string(tag) + "_" +
                                           std::to_string(::getpid()) + ".bin"))
    .string();
}

void remove_file(std::string const& file)
{
  std::error_code ec;
  std::filesystem::remove(file, ec);
}

/// Removes its file on scope exit (also when a measurement throws).
struct scoped_file {
  explicit scoped_file(std::string p) : path(std::move(p)) {}
  ~scoped_file() { remove_file(path); }
  scoped_file(scoped_file const&)            = delete;
  scoped_file& operator=(scoped_file const&) = delete;
  std::string path;
};

void evict_page_cache(std::string const& file)
{
  int const fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) { return; }
  static_cast<void>(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
  ::close(fd);
}

/// [lo, hi) block-index slice of submitting thread @p t out of @p n.
std::pair<std::size_t, std::size_t> slice_of(std::size_t n_blocks, std::size_t t, std::size_t n)
{
  std::size_t const base = n_blocks / n;
  std::size_t const rem  = n_blocks % n;
  std::size_t const lo   = t * base + std::min(t, rem);
  return {lo, lo + base + (t < rem ? 1 : 0)};
}

//===----------------------------------------------------------------------===//
// ioctx under test
//===----------------------------------------------------------------------===//

/// Owns the ioctx and (for uring) its pinned staging resource; members are
/// declared so the context is destroyed before the staging it borrows.
struct backend_holder {
  std::unique_ptr<cucascade::memory::numa_region_pinned_host_memory_resource> upstream;
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource> staging;
  std::shared_ptr<io::ioctx> ctx;

  ~backend_holder()
  {
    if (ctx != nullptr) { ctx->shutdown(); }
    ctx.reset();
  }
};

std::unique_ptr<backend_holder> make_backend(options const& o)
{
  auto h = std::make_unique<backend_holder>();
  if (o.backend == "uring") {
    // Same staging setup as parquet_io_benchmark: 1 MiB pinned blocks.
    constexpr std::size_t chunks_per_slab =
      cucascade::memory::fixed_size_host_memory_resource::default_pool_size;
    constexpr std::size_t capacity = 20 * chunks_per_slab * MiB;
    h->upstream =
      std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(0, true);
    h->staging = std::make_unique<cucascade::memory::fixed_size_host_memory_resource>(
      0, *h->upstream, capacity, capacity, MiB, chunks_per_slab, 1);
    io::uring::config cfg{};
    cfg.use_odirect     = o.odirect;
    cfg.slices_per_pass = o.slices_per_pass;
    cfg.scheduling      = o.scheduling;
    auto rctx = std::make_shared<io::uring::uring_reactor::reactor_context>(cfg, h->staging.get());
    h->ctx    = std::make_shared<io::uring::uring_ioctx>(o.runners, std::move(rctx));
  } else {
    // kvikIO writes are eager on the caller thread; a single pwrite is split
    // into task_size chunks over a pool of `runners` threads.  POSIX mode
    // (cuFile/GDS is not assumed on the benchmark filesystem).
    kvikio::defaults::set_auto_direct_io_write(o.odirect);
    io::kvikio_config cfg{};
    cfg.nthreads    = static_cast<unsigned int>(o.runners);
    cfg.compat_mode = kvikio::CompatMode::ON;
    h->ctx          = std::make_shared<io::kvikio_context>(cfg);
  }
  h->ctx->start();
  return h;
}

struct write_result {
  clock_type::duration write{};       ///< until every write future resolved
  clock_type::duration with_flush{};  ///< plus flush_async
};

/// Write `o.size` bytes to @p file through @p ctx (from @p dev_src when
/// non-null, else @p host_src), then flush it.
write_result ioctx_write(options const& o,
                         io::ioctx& ctx,
                         std::string const& file,
                         aligned_buffer const& host_src,
                         device_buffer const* dev_src)
{
  auto obj = ctx.open_io_object_for_write(
    file, io::write_open_options{.mode = io::write_mode::create_or_truncate, .size_hint = o.size});
  std::size_t const n_blocks = (o.size + o.block - 1) / o.block;
  io::write_options const wopts{.durability = o.durability};

  auto const t0 = clock_type::now();
  std::vector<std::thread> submitters;
  std::atomic<bool> failed{false};
  std::string error;
  std::mutex error_mutex;
  for (std::size_t t = 0; t < o.threads; ++t) {
    submitters.emplace_back([&, t] {
      try {
        auto const [lo, hi] = slice_of(n_blocks, t, o.threads);
        std::unique_ptr<cuda_stream> stream;
        if (dev_src != nullptr) { stream = std::make_unique<cuda_stream>(); }
        std::deque<cucascade::exec::semi_future<std::size_t>> window;
        for (std::size_t b = lo; b < hi; ++b) {
          if (window.size() >= o.qd) {
            static_cast<void>(std::move(window.front()).get());
            window.pop_front();
          }
          std::size_t const off = b * o.block;
          std::size_t const len = std::min(o.block, o.size - off);
          window.push_back(
            dev_src != nullptr
              ? ctx.device_write_async(
                  *obj, off, len, dev_src->data(), ::cuda::stream_ref{stream->s}, wopts)
              : ctx.host_write_async(*obj, off, len, host_src.data(), wopts));
        }
        while (!window.empty()) {
          static_cast<void>(std::move(window.front()).get());
          window.pop_front();
        }
      } catch (std::exception const& e) {
        std::lock_guard lock(error_mutex);
        failed = true;
        error  = e.what();
      }
    });
  }
  for (auto& s : submitters) {
    s.join();
  }
  if (failed) { throw std::runtime_error("write failed: " + error); }
  auto const t1 = clock_type::now();
  ctx.flush_async(*obj).get();
  auto const t2 = clock_type::now();
  obj.reset();
  return {t1 - t0, t2 - t0};
}

/// dd-equivalent baseline: `threads` threads, each a sequential pwrite loop of
/// `o.block` over its contiguous slice, then fdatasync.
write_result pwrite_write(options const& o,
                          std::string const& file,
                          aligned_buffer const& src,
                          std::size_t threads)
{
  int const fd =
    ::open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC | (o.odirect ? O_DIRECT : 0), 0644);
  if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open " + file); }
  std::size_t const n_blocks = (o.size + o.block - 1) / o.block;
  std::atomic<int> err{0};
  auto const t0 = clock_type::now();
  std::vector<std::thread> workers;
  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      auto const [lo, hi] = slice_of(n_blocks, t, threads);
      for (std::size_t b = lo; b < hi && err == 0; ++b) {
        std::size_t const off = b * o.block;
        std::size_t const len = std::min(o.block, o.size - off);
        std::size_t done      = 0;
        while (done < len) {
          auto const n =
            ::pwrite(fd, src.data() + done, len - done, static_cast<off_t>(off + done));
          if (n < 0) {
            if (errno == EINTR) { continue; }
            err = errno;
            return;
          }
          done += static_cast<std::size_t>(n);
        }
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  auto const t1 = clock_type::now();
  if (err == 0 && ::fdatasync(fd) != 0) { err = errno; }
  auto const t2 = clock_type::now();
  ::close(fd);
  if (err != 0) { throw std::system_error(err, std::generic_category(), "pwrite baseline"); }
  return {t1 - t0, t2 - t0};
}

std::string describe(options const& o)
{
  std::ostringstream s;
  s << o.backend << " source=" << o.source << " odirect=" << o.odirect
    << " durability=" << (o.durability == io::write_durability::none ? "none" : "data_sync")
    << " size=" << o.size / MiB << "MiB block=" << o.block / KiB << "KiB runners=" << o.runners
    << " threads=" << o.threads << " qd=" << o.qd;
  if (o.backend == "uring") {
    s << " slices_per_pass=" << o.slices_per_pass
      << " bg_groups=" << o.scheduling.max_background_groups
      << " bg_share=" << o.scheduling.background_slot_fraction
      << " bg_reserve=" << o.scheduling.reserved_background_slots;
  }
  return s.str();
}

void report(std::string const& label, std::size_t bytes, std::vector<write_result> const& results)
{
  std::vector<double> w;
  std::vector<double> f;
  std::cout << std::fixed << std::setprecision(2);
  for (std::size_t i = 0; i < results.size(); ++i) {
    w.push_back(gbps(bytes, results[i].write));
    f.push_back(gbps(bytes, results[i].with_flush));
    std::cout << "  " << label << " rep " << i + 1 << ": " << w.back()
              << " GB/s  (incl. flush/fdatasync: " << f.back() << " GB/s)\n";
  }
  std::cout << "  " << label << " MEDIAN: " << median(w) << " GB/s  (incl. flush: " << median(f)
            << " GB/s)\n\n";
}

//===----------------------------------------------------------------------===//
// mode=write
//===----------------------------------------------------------------------===//

int run_write_mode(options const& o)
{
  aligned_buffer host_src(o.block);
  std::unique_ptr<device_buffer> dev_src;
  if (o.source == "device") { dev_src = std::make_unique<device_buffer>(host_src); }

  std::cout << "== write: " << describe(o) << "\n   path=" << o.path << "\n\n";

  // Repetitions are interleaved (pwrite x1, pwrite xN, ioctx per round) so
  // that drive-level effects -- e.g. an SSD's SLC write cache filling up --
  // hit the baselines and the ioctx alike.
  auto backend = make_backend(o);
  std::vector<write_result> dd_results;
  std::vector<write_result> par_results;
  std::vector<write_result> results;
  auto const pfile = test_file(o, "pwrite");
  auto const file  = test_file(o, o.backend);
  auto const pause = [&] { std::this_thread::sleep_for(std::chrono::milliseconds(o.pause_ms)); };
  for (std::size_t r = 0; r < o.reps; ++r) {
    if (o.baseline) {
      dd_results.push_back(pwrite_write(o, pfile, host_src, 1));
      remove_file(pfile);
      pause();
      par_results.push_back(pwrite_write(o, pfile, host_src, o.baseline_threads));
      remove_file(pfile);
      pause();
    }
    results.push_back(ioctx_write(o, *backend->ctx, file, host_src, dev_src.get()));
    remove_file(file);
    pause();
  }
  if (o.baseline) {
    report("pwrite baseline (host, 1 thread, dd-equivalent)", o.size, dd_results);
    report("pwrite baseline (host, " + std::to_string(o.baseline_threads) + " threads)",
           o.size,
           par_results);
  }
  report(o.backend + " " + o.source, o.size, results);
  return 0;
}

//===----------------------------------------------------------------------===//
// mode=mixed
//===----------------------------------------------------------------------===//

struct latency_stats {
  std::size_t count{0};
  double p50_us{0};
  double p99_us{0};
  double max_us{0};
};

latency_stats summarize(std::vector<double> v)
{
  latency_stats s;
  if (v.empty()) { return s; }
  std::sort(v.begin(), v.end());
  auto const at = [&](double p) {
    return v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1))];
  };
  s.count  = v.size();
  s.p50_us = at(0.50);
  s.p99_us = at(0.99);
  s.max_us = v.back();
  return s;
}

void print_latency(std::string const& label, latency_stats const& s)
{
  std::cout << std::fixed << std::setprecision(1) << "  " << std::left << std::setw(44) << label
            << std::right << " reads=" << std::setw(7) << s.count << "  p50=" << std::setw(8)
            << s.p50_us << " us  p99=" << std::setw(8) << s.p99_us << " us  max=" << std::setw(9)
            << s.max_us << " us\n";
}

/// Issue one read at a time (random aligned offsets) until @p keep_going
/// returns false; returns per-read latency in microseconds.
std::vector<double> latency_reads(std::size_t file_size,
                                  std::size_t read_size,
                                  std::function<bool(std::size_t)> const& keep_going,
                                  std::function<void(std::size_t, std::uint8_t*)> const& read_one)
{
  aligned_buffer dst(read_size, /*random_fill=*/false);
  std::mt19937_64 rng{7};
  std::uniform_int_distribution<std::size_t> pick(0, (file_size - read_size) / PAGE);
  std::vector<double> lat;
  while (keep_going(lat.size())) {
    std::size_t const off = pick(rng) * PAGE;
    auto const t0         = clock_type::now();
    read_one(off, dst.data());
    lat.push_back(std::chrono::duration<double, std::micro>(clock_type::now() - t0).count());
  }
  return lat;
}

int run_mixed_mode(options const& o)
{
  aligned_buffer host_src(o.block);
  std::unique_ptr<device_buffer> dev_src;
  if (o.source == "device") { dev_src = std::make_unique<device_buffer>(host_src); }

  std::cout << "== mixed: " << describe(o) << " read_size=" << o.read_size / KiB
            << "KiB read_file=" << o.read_file_size / MiB << "MiB\n   path=" << o.path << "\n\n";

  // The file the latency reads target: written once, then evicted from the
  // page cache so buffered reads also reach the device.
  auto const read_file = test_file(o, "read");
  {
    options fill  = o;
    fill.size     = o.read_file_size;
    fill.odirect  = false;
    auto const fr = pwrite_write(fill, read_file, host_src, 1);
    static_cast<void>(fr);
    evict_page_cache(read_file);
  }
  auto const idle_done = [&](std::size_t n) { return n < o.idle_reads; };

  // -- pread/pwrite reference ------------------------------------------------
  if (o.baseline) {
    int const rfd = ::open(read_file.c_str(), O_RDONLY | O_CLOEXEC | (o.odirect ? O_DIRECT : 0));
    if (rfd < 0) { throw std::system_error(errno, std::generic_category(), "open " + read_file); }
    auto const pread_one = [&](std::size_t off, std::uint8_t* dst) {
      if (::pread(rfd, dst, o.read_size, static_cast<off_t>(off)) !=
          static_cast<ssize_t>(o.read_size)) {
        throw std::runtime_error("pread failed");
      }
    };
    print_latency("pread idle",
                  summarize(latency_reads(o.read_file_size, o.read_size, idle_done, pread_one)));
    std::atomic<bool> writing{true};
    write_result wr{};
    auto const wfile = test_file(o, "pwrite");
    std::thread writer([&] {
      wr      = pwrite_write(o, wfile, host_src, o.baseline_threads);
      writing = false;
    });
    auto const lat = latency_reads(
      o.read_file_size, o.read_size, [&](std::size_t) { return writing.load(); }, pread_one);
    writer.join();
    remove_file(wfile);
    print_latency("pread during pwrite (" + std::to_string(o.baseline_threads) + " threads)",
                  summarize(lat));
    std::cout << std::setprecision(2) << "    concurrent pwrite: " << gbps(o.size, wr.write)
              << " GB/s\n\n";
    ::close(rfd);
    evict_page_cache(read_file);
  }

  // -- ioctx -------------------------------------------------------------------
  auto backend              = make_backend(o);
  auto& ctx                 = *backend->ctx;
  auto robj                 = ctx.open_io_object(read_file);
  auto const ioctx_read_one = [&](std::size_t off, std::uint8_t* dst) {
    if (std::move(ctx.host_read_async(*robj, off, o.read_size, dst)).get() != o.read_size) {
      throw std::runtime_error("short read");
    }
  };
  print_latency(o.backend + " idle",
                summarize(latency_reads(o.read_file_size, o.read_size, idle_done, ioctx_read_one)));

  for (std::size_t r = 0; r < o.reps; ++r) {
    evict_page_cache(read_file);
    std::atomic<bool> writing{true};
    write_result wr{};
    std::exception_ptr werr;
    auto const wfile = test_file(o, o.backend);
    std::thread writer([&] {
      try {
        wr = ioctx_write(o, ctx, wfile, host_src, dev_src.get());
      } catch (...) {
        werr = std::current_exception();
      }
      writing = false;
    });
    auto const lat = latency_reads(
      o.read_file_size, o.read_size, [&](std::size_t) { return writing.load(); }, ioctx_read_one);
    writer.join();
    remove_file(wfile);
    if (werr) { std::rethrow_exception(werr); }
    print_latency(o.backend + " during " + o.source + " writes (rep " + std::to_string(r + 1) + ")",
                  summarize(lat));
    std::cout << std::setprecision(2) << "    concurrent " << o.backend
              << " write: " << gbps(o.size, wr.write) << " GB/s\n";
  }
  robj.reset();
  remove_file(read_file);
  return 0;
}

//===----------------------------------------------------------------------===//
// mode=readmix
//===----------------------------------------------------------------------===//

std::string kv_latency(latency_stats const& s)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(1) << "reads=" << s.count << " p50_us=" << s.p50_us
      << " p99_us=" << s.p99_us << " max_us=" << s.max_us;
  return out.str();
}

/// Run configuration repeated on every RESULT / STATS line, so the lines of
/// several invocations can be pooled and grouped by key.
std::string readmix_tag(options const& o)
{
  std::ostringstream s;
  s << "backend=" << o.backend << " runners=" << o.runners << " odirect=" << o.odirect
    << " bg_class=" << class_name(o.bg_class) << " bg_size=" << size_label(o.bg_size)
    << " bg_slice=" << size_label(o.bg_slice) << " slices_per_pass=" << o.slices_per_pass
    << " bg_groups=" << o.scheduling.max_background_groups
    << " bg_share=" << o.scheduling.background_slot_fraction
    << " bg_reserve=" << o.scheduling.reserved_background_slots << " demand_dst=" << o.demand_dst;
  return s.str();
}

double to_us(std::chrono::nanoseconds d)
{
  return std::chrono::duration<double, std::micro>(d).count();
}

/// Start a statistics window: clear the peaks, return the counters to diff.
io::queue_stats open_stats_window(io::ioctx& ctx)
{
  ctx.reset_stats_peaks();
  return ctx.stats();
}

/// One STATS line for the window opened by @p before.  Per class that retired
/// requests in the window: first-I/O count, mean and max delay, max queue wait
/// (absent class = no request).  Per runner, in registration order: peak
/// in-flight physical ops and MiB submitted in the window.
void print_stats(std::string const& prefix, io::ioctx const& ctx, io::queue_stats const& before)
{
  auto const after = ctx.stats();
  std::ostringstream s;
  s << std::fixed << std::setprecision(1) << "STATS " << prefix;
  for (auto const cls : {io::request_class::latency,
                         io::request_class::read,
                         io::request_class::write,
                         io::request_class::background}) {
    auto const& a = after.per_class[io::request_class_index(cls)];
    auto const& b = before.per_class[io::request_class_index(cls)];
    auto const n  = a.first_io_count - b.first_io_count;
    if (n == 0) { continue; }
    std::string const c = class_name(cls);
    s << " " << c << ".first_io_n=" << n << " " << c
      << ".first_io_mean_us=" << to_us(a.first_io_total - b.first_io_total) / static_cast<double>(n)
      << " " << c << ".first_io_max_us=" << to_us(a.first_io_max) << " " << c
      << ".max_queue_wait_us=" << to_us(a.max_queue_wait);
  }
  std::ostringstream inflight;
  std::ostringstream submitted;
  inflight << std::fixed << std::setprecision(1);
  submitted << std::fixed << std::setprecision(1);
  for (std::size_t i = 0; i < after.runners.size(); ++i) {
    auto const& r       = after.runners[i];
    std::uint64_t start = 0;
    for (auto const& b : before.runners) {
      if (b.id == r.id) { start = b.bytes_submitted; }
    }
    char const* sep = i == 0 ? "" : ",";
    inflight << sep << r.max_inflight_ops;
    submitted << sep << static_cast<double>(r.bytes_submitted - start) / static_cast<double>(MiB);
  }
  s << " runner_max_inflight_ops=" << inflight.str() << " runner_submitted_mib=" << submitted.str();
  std::cout << s.str() << "\n";
}

int run_readmix_mode(options const& o)
{
  auto const tag   = readmix_tag(o);
  auto const pause = [&] { std::this_thread::sleep_for(std::chrono::milliseconds(o.pause_ms)); };
  std::cout << std::fixed << std::setprecision(2) << "== readmix: " << tag
            << " read_file=" << size_label(o.read_file_size) << " idle_reads=" << o.idle_reads
            << " reps=" << o.reps << "\n   path=" << o.path << "\n\n";

  // Device demand reads need a GPU: without one the run is skipped, not failed.
  std::unique_ptr<device_buffer> demand_dev;
  std::unique_ptr<cuda_stream> demand_stream;
  if (o.demand_dst == "device") {
    try {
      demand_dev = std::make_unique<device_buffer>(
        *std::max_element(o.demand_sizes.begin(), o.demand_sizes.end()));
      demand_stream = std::make_unique<cuda_stream>();
    } catch (std::exception const& e) {
      std::cout << "SKIP readmix " << tag << " reason=\"no usable GPU: " << e.what() << "\"\n";
      return 0;
    }
  }

  // Built first: an invalid scheduling knob fails before the files are written.
  auto backend = make_backend(o);
  auto& ctx    = *backend->ctx;

  // The demand file and the background file: written once, then evicted from
  // the page cache before every measurement so buffered reads reach the device.
  aligned_buffer host_src(o.block);
  scoped_file const read_file{test_file(o, "read")};
  scoped_file const bg_file{test_file(o, "background")};
  auto const create = [&](std::string const& file, std::size_t bytes) {
    options fill = o;
    fill.size    = bytes;
    fill.odirect = false;
    static_cast<void>(pwrite_write(fill, file, host_src, 1));
    evict_page_cache(file);
  };
  create(read_file.path, o.read_file_size);
  create(bg_file.path, o.bg_size);
  auto const robj = ctx.open_io_object(read_file.path);
  auto const bobj = ctx.open_io_object(bg_file.path);

  // The background request: one vectored read of the whole file into one
  // buffer, like a prefetch of many chunks.
  aligned_buffer bg_dst(o.bg_size, /*random_fill=*/false);
  std::vector<io::slice> bg_slices;
  for (std::size_t off = 0; off < o.bg_size; off += o.bg_slice) {
    bg_slices.emplace_back(off, std::min(o.bg_slice, o.bg_size - off), bg_dst.data() + off);
  }
  io::io_options const bg_opts{o.bg_class};

  // -- (1) background read alone ----------------------------------------------
  std::vector<double> alone;
  for (std::size_t r = 0; r < o.reps; ++r) {
    evict_page_cache(bg_file.path);
    auto const before = open_stats_window(ctx);
    auto const t0     = clock_type::now();
    auto const got    = ctx.host_readv_async_io(*bobj, bg_slices, bg_opts).get();
    auto const t      = clock_type::now() - t0;
    if (got != o.bg_size) { throw std::runtime_error("short background read"); }
    alone.push_back(gbps(o.bg_size, t));
    auto const prefix = "readmix " + tag + " phase=bg_alone rep=" + std::to_string(r + 1);
    std::cout << "RESULT " << prefix << " bg_gbps=" << alone.back() << "\n";
    print_stats(prefix, ctx, before);
    pause();
  }
  double const alone_gbps = median(alone);
  std::cout << "RESULT readmix " << tag << " phase=bg_alone_median reps=" << o.reps
            << " bg_gbps=" << alone_gbps << "\n\n";

  // -- (2) per demand size: idle, then during the background read -------------
  for (auto const size : o.demand_sizes) {
    // Demand reads go through host_read_async / device_read_async with the
    // automatic class (no prefetch handle), i.e. latency or read by size --
    // never background.  A device read counts until its copy is on the GPU.
    auto const demand_class =
      io::resolve_request_class(io::request_class::automatic, io::io_kind::read, size);
    auto const dprefix = "readmix " + tag + " demand_size=" + size_label(size) +
                         " demand_class=" + class_name(demand_class);
    auto const read_one = [&](std::size_t off, std::uint8_t* host_dst) {
      std::size_t got = 0;
      if (demand_dev != nullptr) {
        got = ctx
                .device_read_async(
                  *robj, off, size, demand_dev->data(), ::cuda::stream_ref{demand_stream->s})
                .get();
        check_cuda(cudaStreamSynchronize(demand_stream->s), "cudaStreamSynchronize");
      } else {
        got = ctx.host_read_async(*robj, off, size, host_dst).get();
      }
      if (got != size) { throw std::runtime_error("short demand read"); }
    };

    evict_page_cache(read_file.path);
    auto const idle_before = open_stats_window(ctx);
    auto const idle        = summarize(latency_reads(
      o.read_file_size, size, [&](std::size_t n) { return n < o.idle_reads; }, read_one));
    std::cout << "RESULT " << dprefix << " phase=idle " << kv_latency(idle) << "\n";
    print_stats(dprefix + " phase=idle", ctx, idle_before);
    pause();

    std::vector<double> pooled;
    std::vector<double> bg_rates;
    for (std::size_t r = 0; r < o.reps; ++r) {
      evict_page_cache(read_file.path);
      evict_page_cache(bg_file.path);
      auto const before = open_stats_window(ctx);
      std::atomic<bool> bg_done{false};
      clock_type::time_point bg_end{};
      std::size_t bg_got{0};
      std::exception_ptr bg_err;
      auto const t0 = clock_type::now();
      // Submitted here (before the first demand read); the waiter only records
      // when it resolved.
      std::thread waiter([&, f = ctx.host_readv_async_io(*bobj, bg_slices, bg_opts)]() mutable {
        try {
          bg_got = std::move(f).get();
        } catch (...) {
          bg_err = std::current_exception();
        }
        bg_end = clock_type::now();
        bg_done.store(true, std::memory_order_release);
      });
      std::vector<double> lat;
      std::exception_ptr demand_err;
      try {
        lat = latency_reads(
          o.read_file_size,
          size,
          [&](std::size_t) { return !bg_done.load(std::memory_order_acquire); },
          read_one);
      } catch (...) {
        demand_err = std::current_exception();
      }
      waiter.join();
      if (bg_err) { std::rethrow_exception(bg_err); }
      if (demand_err) { std::rethrow_exception(demand_err); }
      if (bg_got != o.bg_size) { throw std::runtime_error("short background read"); }
      bg_rates.push_back(gbps(o.bg_size, bg_end - t0));
      auto const prefix = dprefix + " phase=concurrent rep=" + std::to_string(r + 1);
      std::cout << "RESULT " << prefix << " " << kv_latency(summarize(lat))
                << " bg_gbps=" << bg_rates.back() << "\n";
      print_stats(prefix, ctx, before);
      pooled.insert(pooled.end(), lat.begin(), lat.end());
      pause();
    }
    // All repetitions pooled: concurrent phases are short, so one repetition
    // may hold too few samples for a meaningful p99.
    auto const all      = summarize(std::move(pooled));
    double const bg_med = median(bg_rates);
    double const p99_x  = idle.p99_us > 0 ? all.p99_us / idle.p99_us : 0.0;
    double const bg_x   = alone_gbps > 0 ? bg_med / alone_gbps : 0.0;
    std::cout << "RESULT " << dprefix << " phase=concurrent_all reps=" << o.reps << " "
              << kv_latency(all) << " bg_gbps_median=" << bg_med << " p99_x_idle=" << p99_x
              << " bg_x_alone=" << bg_x << "\n\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const o = parse_args(argc, argv);
    if (o.mode == "readmix") { return run_readmix_mode(o); }
    return o.mode == "write" ? run_write_mode(o) : run_mixed_mode(o);
  } catch (std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
