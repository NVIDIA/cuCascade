/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cache_residency.hpp"
#include "transport_simulator.hpp"

#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/device_engine.hpp>
#include <cucascade/io/cufile/runtime.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace std::chrono_literals;
using clock_type          = std::chrono::steady_clock;
constexpr std::size_t mib = 1UL << 20;
constexpr double gib      = static_cast<double>(1UL << 30);

void check(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct options {
  std::string spill_mode{"new"};
  bool flush{};
  std::string file, engine{"legacy"}, workload{"sustained"}, distribution{"mixed"}, csv;
  std::size_t threads{1}, tasks{4}, pipeline{2}, task_bytes{2048 * mib}, single_bytes{1024 * mib};
  std::size_t file_bytes{4096 * mib}, chunk_min{mib}, chunk_max{8 * mib}, columns{64};
  std::size_t reactors{1}, repetitions{1}, min_seconds{1};
  unsigned qd{32};
  std::vector<unsigned> arena_classes{4};
  std::size_t op_bytes{4 * mib}, arena_bytes{256 * mib};
  int device{};
  bool create{}, verify{true}, sync_each{}, require_direct{true}, unsafe_arena{}, coalesce{true};
  bool drop_cache{}, guarded_destinations{}, posix_only{};
  double soak_minutes{}, recreate_seconds{60}, sample_seconds{5};
  unsigned slow_floor_us{100}, stuck_ms{2};
  // Internal epoch state: the driver recreates ALL context/engine resources between epochs.
  double epoch_seconds{};
  std::size_t epoch{};
  clock_type::time_point soak_start{};
  std::string soak_csv;
  std::vector<mount_override> overrides;
};
std::size_t number(std::string_view value)
{
  std::size_t n{};
  auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), n);
  if (ec != std::errc{} || end != value.data() + value.size() || value.empty())
    throw std::invalid_argument("invalid integer: " + std::string(value));
  return n;
}
std::size_t bytes(std::string_view value)
{
  auto n = number(value);
  if (!n || n > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) / mib)
    throw std::invalid_argument("MiB must be positive and fit off_t");
  return n * mib;
}
options parse(int argc, char** argv)
{
  options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout
        << "cucascade_cufile_engine_bench --file PATH [options]\n"
           "  --workload sustained|restore|single|spill (B-3/B-4/B-5/B-6; default sustained)\n"
           "  --engine legacy|async|simulated (default legacy)\n"
           "  --posix-only (legacy engine only; bypass cuFile)\n"
           "  --create-file (NEW deterministic pattern file(s); refuses overwrite)\n"
           "  --size-mib 4096 --threads 1 --reactors 1 --tasks 4 --pipeline 2 --sync-each\n"
           "  --min-seconds 1 --repetitions 1 (whole passes; setup/verification untimed)\n"
           "  --drop-cache (fsync + DONTNEED on measured files before EVERY timed pass)\n"
           "  --guarded-destinations (reverse destinations + 64-byte canaries)\n"
           "  --task-mib 2048 --chunk-min-mib 1 --chunk-max-mib 8\n"
           "  --columns 64 --col-dist kib|mixed|large --single-mib 1024\n"
           "  --qd 32 --op-mib 4 --arena-mib 256 --device 0 --coalesce 0|1\n"
           "  --arena-classes 4 (comma-separated MiB classes; production default {4})\n"
           "  --override fs_type:VALUE|mount_prefix:VALUE|fsid:VALUE (repeatable)\n"
           "  --registration slot|unsafe-arena (slot default; AM-9 diagnostic only)\n"
           "  --require-direct-evidence 0|1 (default 1; 0 TEST ONLY)\n"
           "  --verify 0|1 (default 1: separate verified pass BEFORE timing)\n"
           "  --spill-mode new|overwrite --flush 0|1 (spill; defaults new, 0)\n"
           "  --csv PATH (header + one row; '-' = stdout)\n"
           "  --slow-floor-us N --stuck-ms N (recorded per-host calibration)\n"
           "  --soak-minutes N --recreate-every-s K --sample-every-s S\n"
           "    Samples at drained pass boundaries (intervals are lower bounds); requires --csv.\n";
      std::exit(0);
    }
    if (key == "--posix-only") {
      o.posix_only = true;
      continue;
    }
    if (key == "--create-file") {
      o.create = true;
      continue;
    }
    if (key == "--guarded-destinations") {
      o.guarded_destinations = true;
      continue;
    }
    if (key == "--drop-cache") {
      o.drop_cache = true;
      continue;
    }
    if (key == "--sync-each") {
      o.sync_each = true;
      continue;
    }
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    std::string value = argv[i];
    if (key == "--file")
      o.file = value;
    else if (key == "--spill-mode")
      o.spill_mode = value;
    else if (key == "--flush") {
      if (value != "0" && value != "1") throw std::invalid_argument("--flush requires 0|1");
      o.flush = value == "1";
    } else if (key == "--engine")
      o.engine = value;
    else if (key == "--workload")
      o.workload = value;
    else if (key == "--col-dist")
      o.distribution = value;
    else if (key == "--csv")
      o.csv = value;
    else if (key == "--slow-floor-us")
      o.slow_floor_us = static_cast<unsigned>(std::stoul(value));
    else if (key == "--stuck-ms")
      o.stuck_ms = static_cast<unsigned>(std::stoul(value));
    else if (key == "--soak-minutes" || key == "--recreate-every-s" || key == "--sample-every-s") {
      std::size_t end{};
      auto n = std::stod(value, &end);
      if (end != value.size() || !std::isfinite(n) || n <= 0 || n > 86400)
        throw std::invalid_argument("invalid soak duration");
      if (key == "--soak-minutes")
        o.soak_minutes = n;
      else if (key == "--recreate-every-s")
        o.recreate_seconds = n;
      else
        o.sample_seconds = n;
    } else if (key == "--threads")
      o.threads = number(value);
    else if (key == "--reactors")
      o.reactors = number(value);
    else if (key == "--repetitions")
      o.repetitions = number(value);
    else if (key == "--min-seconds")
      o.min_seconds = number(value);
    else if (key == "--tasks")
      o.tasks = number(value);
    else if (key == "--pipeline")
      o.pipeline = number(value);
    else if (key == "--task-mib")
      o.task_bytes = bytes(value);
    else if (key == "--single-mib")
      o.single_bytes = bytes(value);
    else if (key == "--size-mib")
      o.file_bytes = bytes(value);
    else if (key == "--chunk-min-mib")
      o.chunk_min = bytes(value);
    else if (key == "--chunk-max-mib")
      o.chunk_max = bytes(value);
    else if (key == "--columns")
      o.columns = number(value);
    else if (key == "--op-mib")
      o.op_bytes = bytes(value);
    else if (key == "--arena-classes") {
      o.arena_classes.clear();
      std::istringstream list(value);
      std::string item;
      while (std::getline(list, item, ',')) {
        auto const n = number(item);
        if (n != 1 && n != 2 && n != 4 && n != 8)
          throw std::invalid_argument("arena classes must be 1,2,4,8");
        o.arena_classes.push_back(static_cast<unsigned>(n));
      }
      if (value.empty() || value.back() == ',') throw std::invalid_argument("empty arena class");
    } else if (key == "--arena-mib")
      o.arena_bytes = bytes(value);
    else if (key == "--qd") {
      auto n = number(value);
      if (!n || n > 128) throw std::invalid_argument("qd must be 1..128");
      o.qd = static_cast<unsigned>(n);
    } else if (key == "--device") {
      auto n = number(value);
      if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("device exceeds INT_MAX");
      o.device = static_cast<int>(n);
    } else if (key == "--registration") {
      if (value != "slot" && value != "unsafe-arena")
        throw std::invalid_argument("invalid registration");
      o.unsafe_arena = value == "unsafe-arena";
    } else if (key == "--verify" || key == "--require-direct-evidence" || key == "--coalesce") {
      if (value != "0" && value != "1") throw std::invalid_argument(key + " requires 0|1");
      if (key == "--verify")
        o.verify = value == "1";
      else if (key == "--coalesce")
        o.coalesce = value == "1";
      else
        o.require_direct = value == "1";
    } else if (key == "--override") {
      auto colon    = value.find(':');
      auto selector = value.substr(0, colon);
      mount_match match;
      if (selector == "fs_type")
        match = mount_match::fs_type;
      else if (selector == "mount_prefix")
        match = mount_match::mount_prefix;
      else if (selector == "fsid")
        match = mount_match::fsid;
      else
        throw std::invalid_argument("unknown override selector");
      if (colon == std::string::npos) throw std::invalid_argument("override needs selector:VALUE");
      o.overrides.push_back({match,
                             value.substr(colon + 1),
                             mount_override_state::operator_qualified,
                             o.qd,
                             o.op_bytes});
    } else
      throw std::invalid_argument("unknown option: " + key);
  }
  if (o.file.empty() || !o.threads || o.threads > 128 || !o.reactors || o.reactors > 128 ||
      !o.repetitions || o.repetitions > 100000 || o.min_seconds > 86400 || !o.tasks ||
      o.tasks > 100000 || !o.pipeline || o.pipeline > 64 || !o.columns || o.columns > 4096 ||
      o.chunk_min > o.chunk_max || o.chunk_max > o.file_bytes ||
      (o.op_bytes != mib && o.op_bytes != 2 * mib && o.op_bytes != 4 * mib &&
       o.op_bytes != 8 * mib) ||
      o.arena_bytes % (16 * mib))
    throw std::invalid_argument("invalid sizes/counts (see --help)");
  if (o.posix_only && o.engine != "legacy")
    throw std::invalid_argument("--posix-only requires --engine legacy");
  if (o.engine != "legacy" && o.engine != "async" && o.engine != "simulated")
    throw std::invalid_argument("invalid engine");
  if (o.workload != "sustained" && o.workload != "restore" && o.workload != "single" &&
      o.workload != "spill")
    throw std::invalid_argument("invalid workload");
  if (o.workload == "spill" && (!o.create || o.engine == "simulated" || o.guarded_destinations))
    throw std::invalid_argument(
      "spill requires --create-file, legacy|async, and no destination guards");
  if (o.spill_mode != "new" && o.spill_mode != "overwrite")
    throw std::invalid_argument("invalid spill-mode");
  if (o.distribution != "kib" && o.distribution != "mixed" && o.distribution != "large")
    throw std::invalid_argument("invalid col-dist");
  if (o.workload == "single" && o.single_bytes > o.file_bytes)
    throw std::invalid_argument("single-mib exceeds size-mib");
  for (auto& entry : o.overrides) {
    entry.qd           = o.qd;
    entry.op_bytes     = o.op_bytes;
    entry.allow_writes = o.workload == "spill";
  }
  if (o.soak_minutes && (o.csv.empty() || o.csv == "-" || !o.verify))
    throw std::invalid_argument("soak requires --csv PATH and --verify 1");
  async_engine_config validation;
  validation.mount_overrides    = o.overrides;
  validation.arena_slot_classes = o.arena_classes;
  validation.validate();  // Reject invalid selectors before --create-file can write anything.
  return o;
}

// Same offset-dependent SplitMix64 pattern as cufile_async_qd. No zero-file shortcuts.
void fill(std::vector<std::uint64_t>& words, std::size_t offset, std::uint64_t seed = 0)
{
  for (std::size_t i = 0; i < words.size(); ++i) {
    auto v = static_cast<std::uint64_t>(offset / 8 + i) + 0x9e3779b97f4a7c15ULL +
             seed * 0xd1342543de82ef95ULL;
    v        = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v        = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    words[i] = v ^ (v >> 31);
  }
}
void create_file(std::string const& path, std::size_t size)
{
  file_descriptor fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
  if (fd.get() < 0)
    throw std::runtime_error("create NEW file " + path + ": " + std::strerror(errno));
  std::vector<std::uint64_t> words(mib / 8);
  for (std::size_t offset = 0; offset < size; offset += mib) {
    fill(words, offset);
    auto count = std::min(mib, size - offset);
    std::size_t written{};
    while (written < count) {
      auto n = ::write(fd.get(), reinterpret_cast<char*>(words.data()) + written, count - written);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("pattern file write failed");
      written += static_cast<std::size_t>(n);
    }
  }
  if (fsync(fd.get())) throw std::runtime_error("pattern fsync failed");
}
void pread_exact(int fd, void* ptr, std::size_t size, std::size_t offset)
{
  std::size_t done{};
  while (done < size) {
    auto n =
      pread(fd, static_cast<char*>(ptr) + done, size - done, static_cast<off_t>(offset + done));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("reference pread failed or file too small");
    done += static_cast<std::size_t>(n);
  }
}
struct layout {
  std::string path;
  std::vector<range> ranges;
  std::size_t file_bytes{}, logical_bytes{};
};
layout make_layout(options const& o, std::size_t index)
{
  layout l;
  l.path = o.file + (o.workload == "restore" ? ".batch-" + std::to_string(index)
                     : o.workload == "spill" ? ".spill-" + std::to_string(index)
                                             : "");
  std::mt19937_64 rng(42 + index);
  if (o.workload == "restore") {
    for (std::size_t i = 0; i < o.columns; ++i) {
      auto size = (1 + rng() % 16) * 1024;
      auto kind = rng() % 100;
      if (o.distribution == "large")
        size = (64 + rng() % 961) * mib;
      else if (o.distribution == "mixed" && kind >= 70)
        size = kind < 95 ? (1 + rng() % 8) * mib : (64 + rng() % 65) * mib;
      l.ranges.push_back({l.file_bytes, size});
      l.file_bytes += (size + 4095) / 4096 * 4096;
      l.logical_bytes += size;
    }
  } else if (o.workload == "spill") {
    l.ranges.push_back({37, o.task_bytes - 74});
    l.logical_bytes = o.task_bytes - 74;
    l.file_bytes    = o.task_bytes;
  } else if (o.workload == "single") {
    l.ranges.push_back({0, o.single_bytes});
    l.logical_bytes = o.single_bytes;
    l.file_bytes    = o.file_bytes;
  } else {
    std::uniform_int_distribution<std::size_t> chunks(o.chunk_min / mib, o.chunk_max / mib);
    while (l.logical_bytes < o.task_bytes) {
      auto size = std::min(chunks(rng) * mib, o.task_bytes - l.logical_bytes);
      std::uniform_int_distribution<std::size_t> offsets(0, (o.file_bytes - size) / 4096);
      l.ranges.push_back({offsets(rng) * 4096, size});
      l.logical_bytes += size;
    }
    l.file_bytes = o.file_bytes;
  }
  return l;
}

struct task {
  cudaStream_t stream{};
  void* destination{};
  std::vector<std::vector<slice>> prepared;
  std::vector<std::vector<write_slice>> prepared_writes;
  cucascade::exec::semi_future<std::size_t> future;
  layout const* description{};
  std::span<layout const> layouts;
  std::size_t selected{}, allocation_bytes{};
  bool guarded{}, spilling{};

  explicit task(std::span<layout const> descriptions, bool guards = false, bool spill = false)
    : layouts(descriptions), guarded(guards), spilling(spill)
  {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      auto max_bytes =
        std::max_element(layouts.begin(), layouts.end(), [](auto const& a, auto const& b) {
          return a.logical_bytes < b.logical_bytes;
        })->logical_bytes;
      if (guarded) {
        max_bytes = 0;
        for (auto const& l : layouts) {
          if (l.ranges.size() > (std::numeric_limits<std::size_t>::max() - l.logical_bytes) / 128)
            throw std::overflow_error("guarded destination size overflow");
          max_bytes = std::max(max_bytes, l.logical_bytes + 128 * l.ranges.size());
        }
      }
      // A shifted view of a nonconstant source distinguishes each file without
      // allocating the whole disk working set on the GPU. Refill once per pass,
      // outside timing; a pipeline slot never has two active users.
      allocation_bytes = max_bytes + (spill ? layouts.size() * 4096 : 0);
      check(cudaMallocAsync(&destination, allocation_bytes, stream));
      prepared.reserve(layouts.size());
      prepared_writes.reserve(layouts.size());
      std::size_t layout_index{};
      for (auto const& l : layouts) {
        std::vector<slice> slices;
        slices.reserve(l.ranges.size());
        std::size_t offset = guarded ? allocation_bytes : spill ? layout_index * 4096 : 0;
        ++layout_index;
        for (auto rng : l.ranges) {
          if (guarded) offset -= rng.size + 128;
          slices.emplace_back(
            rng.offset,
            rng.size,
            static_cast<std::uint8_t*>(destination) + offset + (guarded ? 64 : 0));
          if (!guarded) offset += rng.size;
        }
        std::vector<write_slice> writes;
        for (auto const& s : slices)
          writes.emplace_back(s.offset(), s.size(), s.dst);
        prepared_writes.push_back(std::move(writes));
        prepared.push_back(std::move(slices));
      }
      check(cudaStreamSynchronize(stream));  // Pool growth finishes BEFORE timing.
    } catch (...) {
      if (destination) static_cast<void>(cudaFreeAsync(destination, stream));
      static_cast<void>(cudaStreamSynchronize(stream));
      static_cast<void>(cudaStreamDestroy(stream));
      throw;
    }
  }
  void prepare_spill(std::uint64_t seed)
  {
    std::vector<std::uint64_t> words((allocation_bytes + 7) / 8);
    fill(words, 0, seed);
    check(
      cudaMemcpyAsync(destination, words.data(), allocation_bytes, cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  void select(std::size_t index)
  {
    selected    = index;
    description = &layouts[index];
  }
  std::vector<slice> const& slices() const { return prepared[selected]; }
  ~task()
  {
    if (future.valid()) {
      try {
        static_cast<void>(std::move(future).get());
      } catch (...) {
      }
    }
    if (destination) static_cast<void>(cudaFreeAsync(destination, stream));
    static_cast<void>(cudaStreamSynchronize(stream));
    static_cast<void>(cudaStreamDestroy(stream));
  }
  void consume(bool verify)
  {
    if (std::move(future).get() != description->logical_bytes)
      throw std::runtime_error("short logical transfer");
    // AM-1: consume future then synchronize ONLY this task's caller stream.
    check(cudaStreamSynchronize(stream));
    if (verify) verify_contents();
  }
  void verify_contents()
  {
    file_descriptor reference(::open(description->path.c_str(), O_RDONLY | O_CLOEXEC));
    if (reference.get() < 0) throw std::runtime_error("verification open failed");
    std::vector<std::byte> expected(64 * 1024), actual(64 * 1024);
    for (auto const& s : slices()) {
      // Every spill byte is checked, including every physical body piece.
      std::vector<std::size_t> offsets;
      if (spilling) {
        for (std::size_t offset = 0; offset < s.size(); offset += actual.size())
          offsets.push_back(offset);
      } else {
        auto count = std::min<std::size_t>(s.size() <= 64 * 1024 ? s.size() : 256, s.size());
        offsets    = {0, (s.size() - count) / 2, s.size() - count};
      }
      for (auto relative : offsets) {
        auto count =
          std::min(spilling || s.size() <= 64 * 1024 ? actual.size() : 256, s.size() - relative);
        pread_exact(reference.get(), expected.data(), count, s.offset() + relative);
        check(
          cudaMemcpyAsync(actual.data(), s.dst + relative, count, cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        if (!std::equal(actual.begin(),
                        actual.begin() + static_cast<std::ptrdiff_t>(count),
                        expected.begin()))
          throw std::runtime_error("data mismatch at file offset " +
                                   std::to_string(s.offset() + relative));
      }
      if (guarded)
        for (auto* guard : {s.dst - 64, s.dst + s.size()}) {
          check(cudaMemcpyAsync(actual.data(), guard, 64, cudaMemcpyDeviceToHost, stream));
          check(cudaStreamSynchronize(stream));
          if (!std::all_of(actual.begin(), actual.begin() + 64, [](auto byte) {
                return byte == std::byte{0xcc};
              }))
            throw std::runtime_error("destination canary overwritten");
        }
    }
  }
};

double percentile(engine_histogram const& h, unsigned percent)
{
  if (!h.count) return 0;
  auto rank = (h.count * percent + 99) / 100;
  std::uint64_t n{};
  for (unsigned i = 0; i < 64; ++i) {
    n += h.buckets[i];
    if (n >= rank) return static_cast<double>(std::uint64_t{1} << i) / 1000.0;
  }
  return 0;
}

int run(options const& o)
{
  check(cudaSetDevice(o.device));
  cudaMemPool_t pool{};
  check(cudaDeviceGetDefaultMemPool(&pool, o.device));
  std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
  check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  std::vector<layout> layouts;
  for (std::size_t i = 0; i < o.threads * o.tasks; ++i)
    layouts.push_back(make_layout(o, i));
  if (o.create) {
    if (o.workload == "restore" || o.workload == "spill")
      for (auto const& l : layouts)
        create_file(l.path, l.file_bytes);
    else
      create_file(o.file, o.file_bytes);
  }
  config cfg;
  if (o.posix_only) cfg.compat_mode = compat::on;
  cfg.async_engine.mode = o.engine == "legacy" ? engine_mode::legacy
                          : o.engine == "async" && o.require_direct
                            ? engine_mode::async_auto
                            : engine_mode::async_simulated_for_testing;
  // Explicit TEST-ONLY platform bypass on compat hosts; async still uses real libcufile.
  if (o.engine == "async" && !o.require_direct)
    std::cerr << "test_only_compat_admission=1 transport=libcufile (not simulated)\n";
  cfg.async_engine.arena_bytes                   = o.arena_bytes;
  cfg.async_engine.arena_slot_classes            = o.arena_classes;
  cfg.async_engine.device_lane_cap               = static_cast<unsigned>(lanes(o.qd));
  cfg.async_engine.mount_overrides               = o.overrides;
  cfg.async_engine.probe.require_direct_evidence = o.require_direct;
  cfg.async_engine.coalesce                      = o.coalesce;
  cfg.async_engine.async_writes                  = o.workload == "spill";
  cfg.async_engine.demotion.slow_floor_us        = o.slow_floor_us;
  cfg.async_engine.demotion.stuck_ms             = o.stuck_ms;
  if (o.engine == "simulated") {
    // Synthetic CUDA/host scheduling is not a libcufile slow-submit qualification signal.
    cfg.async_engine.demotion.slow_floor_us = 100000;
    cfg.async_engine.demotion.stuck_ms      = 1000;
  }
  // Simulated routing is explicitly operator-qualified, never direct-path evidence.
  if (o.engine == "simulated" && cfg.async_engine.mount_overrides.empty())
    cfg.async_engine.mount_overrides.push_back(
      {mount_match::mount_prefix, "/", mount_override_state::operator_qualified, o.qd, o.op_bytes});
  cfg.async_engine.validate();
  auto shared = std::make_shared<cufile_reactor::reactor_context>(cfg);
  std::shared_ptr<transport_simulator> simulator;
  std::shared_ptr<cufile_transport> transport;
  std::unique_ptr<device_arena> arena;
  std::shared_ptr<lane_pool> lane_resources;
  std::unique_ptr<device_engine> engine;
  if (o.engine != "legacy") {
    static_cast<void>(runtime::instance().ensure_initialized());
    if (o.engine == "simulated") {
      auto largest =
        std::max_element(layouts.begin(), layouts.end(), [](auto const& a, auto const& b) {
          return a.file_bytes < b.file_bytes;
        });
      std::vector<std::byte> image(largest->file_bytes);
      file_descriptor fd(::open(largest->path.c_str(), O_RDONLY | O_CLOEXEC));
      pread_exact(fd.get(), image.data(), image.size(), 0);
      simulator = std::make_shared<transport_simulator>(
        std::move(image), transport_simulator::options{largest->file_bytes, 1024, true, true});
      transport = simulator;
    } else
      transport = make_cufile_transport();
    auto hooks         = simulator ? simulator->arena_hooks() : device_arena_hooks::defaults();
    auto arena_options = device_arena_options::from_config(cfg.async_engine);
    arena_options.registration =
      o.unsafe_arena ? registration_granularity::arena : registration_granularity::slot;
    arena_options.allow_shared_registration = o.unsafe_arena;
    arena       = std::make_unique<device_arena>(o.device, std::move(hooks));
    auto report = arena->warm_up(cfg.async_engine, arena_options);
    std::cerr << "arena_registered_bytes=" << report.registered_bytes
              << " arena_registered_class_mask=" << unsigned(arena->available_class_mask())
              << " registrations=" << arena->gauges().registrations << '\n';
    lane_resources = std::make_shared<lane_pool>(o.device, cfg.async_engine);
    device_engine_options opts;
    opts.arena      = arena.get();
    opts.lanes      = lane_resources;
    opts.transport  = transport;
    engine          = std::make_unique<device_engine>(o.device, cfg.async_engine, std::move(opts));
    auto binding    = std::make_shared<cufile_reactor::reactor_context::engine_binding>();
    binding->engine = engine.get();
    binding->arena  = arena.get();
    binding->transport   = transport;
    binding->device_uuid = "engine-benchmark-" + std::to_string(o.device);
    binding->generation  = 1;
    binding->simulated   = bool(simulator);
    shared->install_engine(o.device, std::move(binding));
  }
  auto context          = std::make_unique<cufile_ioctx>(o.reactors, shared);
  auto const open_start = clock_type::now();
  std::vector<std::shared_ptr<io_object>> objects;
  for (auto const& l : layouts) {
    auto object = o.workload == "spill"
                    ? context->open_io_object_for_write(l.path, {write_mode::open_existing})
                  : o.workload != "restore" && !objects.empty() ? objects.front()
                                                                : context->open_io_object(l.path);
    // Fresh spill passes deliberately truncate; their final unaligned edge can
    // leave EOF below the allocation size. Recreated writers must accept that.
    if (o.workload != "spill" && object->size() < l.file_bytes)
      throw std::runtime_error("input file too small");
    if (simulator && !static_cast<cufile_io_object const&>(*object).cufile_registered())
      throw std::runtime_error(
        "simulated public routing requires a registered file (compat libcufile suffices)");
    objects.push_back(std::move(object));
  }
  auto const open_ms =
    std::chrono::duration<double, std::milli>(clock_type::now() - open_start).count();
  // A dedicated file on the SAME mount warms policy/transport without reading the
  // restore working set. It is removed on all exits; input files are never modified.
  struct warm_file {
    std::string path;
    bool created{};
    ~warm_file()
    {
      if (created) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
      }
    }
  } warm_disk{o.file + ".warmup-" + std::to_string(getpid())};
  auto warm_bytes = std::min(
    o.op_bytes, std::max_element(layouts.begin(), layouts.end(), [](auto const& a, auto const& b) {
                  return a.file_bytes < b.file_bytes;
                })->file_bytes);
  create_file(warm_disk.path, warm_bytes);
  warm_disk.created = true;
  auto warm_object  = o.workload == "spill" ? context->open_io_object_for_write(
                                               warm_disk.path, {write_mode::open_existing})
                                            : context->open_io_object(warm_disk.path);
  if (engine) {
    // Prewarm every physical stream before the policy watchdog can mistake lazy
    // libcufile initialization for a slow/stuck transport. Hold leases to visit each once.
    std::vector<lane_lease> warm_lanes;
    for (std::size_t i = 0; i < lane_resources->capacity(); ++i) {
      auto lane = lane_resources->try_lease();
      auto slot = arena->try_lease_probe(o.op_bytes);
      if (!lane || !slot) throw std::runtime_error("physical warmup resources unavailable");
      auto const& file = static_cast<cufile_io_object const&>(*warm_object);
      async_args args{warm_bytes, 0, slot->buffer_offset_to_pass(), pending_sentinel};
      auto rc = transport->read_async(file.cufile_handle(),
                                      slot->base_ptr_to_pass(),
                                      &args.size,
                                      &args.file_offset,
                                      &args.buffer_offset,
                                      &args.bytes_done,
                                      (*lane)->stream());
      // Even submit refusal can have queued work; synchronize before releasing warm-up leases.
      auto sync = cudaStreamSynchronize((*lane)->stream());
      check(sync);
      if (!rc.ok() || args.bytes_done != static_cast<ssize_t>(args.size))
        throw std::runtime_error("physical warmup read failed");
      if (o.verify) {
        std::vector<std::byte> actual(args.size), expected(args.size);
        check(cudaMemcpy(actual.data(), slot->data(), args.size, cudaMemcpyDeviceToHost));
        pread_exact(file.buffered_fd(), expected.data(), expected.size(), 0);
        if (actual != expected) throw std::runtime_error("physical warmup data mismatch");
      }
      if (o.workload == "spill") {
        args.bytes_done = pending_sentinel;
        auto write_rc   = transport->write_async(file.cufile_handle(),
                                               slot->base_ptr_to_pass(),
                                               &args.size,
                                               &args.file_offset,
                                               &args.buffer_offset,
                                               &args.bytes_done,
                                               (*lane)->stream());
        check(cudaStreamSynchronize((*lane)->stream()));
        if (!write_rc.ok() || args.bytes_done != static_cast<ssize_t>(args.size))
          throw std::runtime_error("physical write warmup failed");
      }
      warm_lanes.push_back(std::move(*lane));
    }
  }
  context->start();
  layout warm_layout{warm_disk.path, {{0, 4096}}, warm_bytes, 4096};
  {
    task warm(std::span{&warm_layout, 1});
    warm.select(0);
    warm.future =
      context->device_readv_async_io(*warm_object, warm.slices(), ::cuda::stream_ref{warm.stream});
    warm.consume(o.verify);
  }
  // Pre-create the full pipeline's streams, destinations and per-layout slice metadata.
  auto depth = std::min(o.tasks, o.sync_each ? std::size_t{1} : o.pipeline);
  std::vector<std::vector<std::unique_ptr<task>>> pipelines(o.threads);
  for (std::size_t t = 0; t < o.threads; ++t)
    for (std::size_t p = 0; p < depth; ++p)
      pipelines[t].push_back(
        std::make_unique<task>(std::span<layout const>{layouts}.subspan(t * o.tasks, o.tasks),
                               o.guarded_destinations,
                               o.workload == "spill"));
  auto execute = [&](std::size_t thread, bool verify) {
    auto& slots = pipelines[thread];
    auto finish = [&](std::size_t i) { slots[i % depth]->consume(verify); };
    for (std::size_t i = 0; i < o.tasks; ++i) {
      if (i >= depth) finish(i - depth);
      auto& next = *slots[i % depth];
      next.select(i);
      if (verify && next.guarded)
        check(cudaMemsetAsync(next.destination, 0xcc, next.allocation_bytes, next.stream));
      if (o.workload == "spill")
        next.future = context->device_writev_async(*objects[thread * o.tasks + i],
                                                   next.prepared_writes[next.selected],
                                                   ::cuda::stream_ref{next.stream});
      else
        next.future = context->device_readv_async_io(
          *objects[thread * o.tasks + i], next.slices(), ::cuda::stream_ref{next.stream});
    }
    for (std::size_t i = o.tasks - std::min(depth, o.tasks); i < o.tasks; ++i)
      finish(i);
    if (o.workload == "spill" && o.flush)
      for (std::size_t i = 0; i < o.tasks; ++i)
        context->flush_async(*objects[thread * o.tasks + i]).get();
  };
  auto prepare_spill = [&](std::size_t pass) {
    if (o.workload != "spill") return;
    for (std::size_t t = 0; t < o.threads; ++t)
      for (std::size_t p = 0; p < depth; ++p)
        pipelines[t][p]->prepare_spill(1 + pass * (o.threads * depth) + t * depth + p);
    if (o.spill_mode == "new")
      for (auto const& object : objects) {
        auto const& file = static_cast<cufile_io_object const&>(*object);
        if (ftruncate(file.buffered_fd(), 0)) throw std::runtime_error("spill truncate failed");
      }
  };
  prepare_spill(0);
  // Verification is a separate, untimed pass. No pread/D2H comparisons in timing.
  if (o.verify)
    for (std::size_t t = 0; t < o.threads; ++t)
      execute(t, true);
  if (engine && !engine->drain(30s)) throw std::runtime_error("warmup engine drain incomplete");
  auto snapshot = [&] {
    if (!engine) return engine_gauges{};
    auto previous = engine->gauges().accounted_ns;
    engine->notify();
    auto deadline = clock_type::now() + 5s;
    for (;;) {
      auto g = engine->gauges();
      if (g.accounted_ns > previous && !g.outstanding_slices && !g.live_operations) return g;
      if (clock_type::now() >= deadline) throw std::runtime_error("diagnostic snapshot timeout");
      std::this_thread::sleep_for(100us);
    }
  };
  std::size_t max_resident_pages{};
  auto evict = [&] {
    if (!o.drop_cache) return;
    for (std::size_t i = 0; i < objects.size(); ++i) {
      if (o.workload != "restore" && o.workload != "spill" && i != 0) break;
      auto const& file = static_cast<cufile_io_object const&>(*objects[i]);
      if (fsync(file.buffered_fd())) throw std::runtime_error("cache-drop fsync failed");
      auto error = posix_fadvise(file.buffered_fd(), 0, 0, POSIX_FADV_DONTNEED);
      if (error) throw std::system_error(error, std::generic_category(), "cache-drop fadvise");
      auto resident      = resident_pages(file.buffered_fd());
      max_resident_pages = std::max(max_resident_pages, resident);
      if (resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE)) > layouts[i].file_bytes / 100)
        std::cerr << "WARNING: >1% file pages resident after eviction\n";
    }
  };
  auto policy_gauges = [&] {
    auto const& file = static_cast<cufile_io_object const&>(*objects.front());
    auto policy      = mount_policy_registry::instance().find(
      file.policy_key("engine-benchmark-" + std::to_string(o.device), 1));
    return policy ? policy->gauges() : mount_policy_gauges{};
  };
  std::uint64_t logged_demotions{};
  auto policy_csv = [&] {
    auto p = policy_gauges();
    if (p.demotions > logged_demotions) {
      auto reason = p.last_trigger == circuit_trigger::slow_of_eight ? "slow-of-eight"
                    : p.last_trigger == circuit_trigger::stuck       ? "stuck"
                                                                     : "failure";
      std::cerr << "mount circuit trigger=" << reason
                << " submit_elapsed_us=" << p.trigger_elapsed.count() / 1000
                << " healthy_p99_us=" << p.healthy_p99.count() / 1000
                << " demotions=" << p.demotions << '\n';
      logged_demotions = p.demotions;
    }
    std::ostringstream out;
    out << ',' << static_cast<int>(p.state) << ',' << p.demotions << ',' << p.cooldowns << ','
        << p.cooldown.count() << ',' << static_cast<int>(p.last_trigger) << ','
        << p.trigger_elapsed.count() / 1000 << ',' << p.healthy_p99.count() / 1000;
    return out.str();
  };
  auto epoch_start          = clock_type::now();
  auto last_sample          = epoch_start;
  std::uint64_t sampled_ops = std::numeric_limits<std::uint64_t>::max();
  auto soak_sample          = [&](bool force) {
    if (o.soak_csv.empty()) return;
    auto now = clock_type::now();
    if (!force && std::chrono::duration<double>(now - last_sample).count() < o.sample_seconds)
      return;
    last_sample = now;
    auto g      = snapshot();  // Quiescent samples distinguish resource leaks from varying backlog.
    if (force && sampled_ops == g.completed_ops) return;
    sampled_ops = g.completed_ops;
    auto a      = arena ? arena->gauges() : arena_gauges{};
    std::size_t free_device{}, total_device{}, fd_count{};
    check(cudaMemGetInfo(&free_device, &total_device));
    for ([[maybe_unused]] auto const& entry : std::filesystem::directory_iterator("/proc/self/fd"))
      ++fd_count;
    std::ifstream status("/proc/self/status");
    std::string line;
    std::size_t thread_count{}, rss_kib{};
    while (std::getline(status, line)) {
      std::istringstream fields(line);
      std::string key;
      fields >> key;
      if (key == "Threads:") fields >> thread_count;
      if (key == "VmRSS:") fields >> rss_kib;
    }
    std::ofstream csv(o.soak_csv, std::ios::app);
    csv << std::chrono::duration<double>(now - o.soak_start).count() << ',' << o.epoch
        << ",quiescent," << a.total_bytes << ',' << a.registered_bytes << ','
        << a.unregistered_bytes << ',' << g.live_operations << ','
        << (lane_resources ? lane_resources->capacity() : 0) << ',' << thread_count << ','
        << rss_kib * 1024 << ',' << g.pinned_write_scratch_pool_bytes << ','
        << g.pinned_write_scratch_bytes << ',' << g.storage_qd << ',' << g.outstanding_slices << ','
        << g.queries << ',' << g.completed_ops << ',' << g.quarantines << ','
        << g.late_copy_failures << ',' << g.accounted_ns << ',' << g.storage_qd_ns << ','
        << g.backlog_ns << ',' << g.backlog_storage_qd_ns << ',' << g.slot_waits << ','
        << g.lane_waits << ',' << g.refill_gap_ns << ',' << g.reaper_active_ns << ','
        << g.async_registered_bytes << ',' << g.async_unregistered_bytes << ',' << g.sync_bytes
        << ',' << percentile(g.residence, 99) << policy_csv() << ',' << fd_count << ','
        << total_device - free_device << ',' << o.verify << '\n';
    if (!csv) throw std::runtime_error("soak CSV write failed");
  };
  auto before = snapshot();
  std::atomic<unsigned> effective_qd{}, min_qd{std::numeric_limits<unsigned>::max()};
  std::atomic<std::size_t> effective_bytes{}, min_bytes{std::numeric_limits<std::size_t>::max()};
  auto sample_effective = [&] {
    if (engine)
      for (auto const& m : engine->gauges().mounts)
        if (m.effective.op_bytes) {
          effective_qd.store(std::max(effective_qd.load(), m.effective.qd));
          min_qd.store(std::min(min_qd.load(), m.effective.qd));
          effective_bytes.store(std::max(effective_bytes.load(), m.effective.op_bytes));
          min_bytes.store(std::min(min_bytes.load(), m.effective.op_bytes));
        }
  };
  std::jthread sampler([&](std::stop_token stop) {
    static_cast<void>(cudaSetDevice(o.device));
    while (!stop.stop_requested()) {
      sample_effective();
      std::this_thread::sleep_for(5ms);
    }
  });
  std::mutex errors_mutex;
  std::exception_ptr error;
  std::barrier gate(static_cast<std::ptrdiff_t>(o.threads + 1));
  std::atomic<bool> done{};
  std::vector<std::jthread> threads;
  for (std::size_t thread = 0; thread < o.threads; ++thread)
    threads.emplace_back([&, thread] {
      for (;;) {
        gate.arrive_and_wait();
        if (done.load()) break;
        try {
          check(cudaSetDevice(o.device));
          execute(thread, false);
        } catch (...) {
          std::lock_guard lock(errors_mutex);
          if (!error) error = std::current_exception();
        }
        gate.arrive_and_wait();
      }
    });
  double seconds{};
  std::size_t passes{}, completed_bytes{};
  std::uint64_t measured_qd_ns{}, measured_accounted_ns{}, measured_backlog_qd_ns{},
    measured_backlog_ns{};
  try {
    do {
      prepare_spill(passes + 1);  // Untimed; unique source seed and fresh extents per pass.
      evict();  // Untimed, repeated for EVERY pass, including after verification/creation.
      auto pass_before = snapshot();  // Exclude eviction/idle gaps from engine QD integrals too.
      auto start       = clock_type::now();
      gate.arrive_and_wait();
      gate.arrive_and_wait();
      seconds += std::chrono::duration<double>(clock_type::now() - start).count();
      if (error) std::rethrow_exception(error);
      ++passes;
      for (auto const& l : layouts)
        completed_bytes += l.logical_bytes;
      if (engine && !engine->drain(30s)) throw std::runtime_error("engine drain incomplete");
      auto pass_after = snapshot();
      measured_qd_ns += pass_after.storage_qd_ns - pass_before.storage_qd_ns;
      measured_accounted_ns += pass_after.accounted_ns - pass_before.accounted_ns;
      measured_backlog_qd_ns +=
        pass_after.backlog_storage_qd_ns - pass_before.backlog_storage_qd_ns;
      measured_backlog_ns += pass_after.backlog_ns - pass_before.backlog_ns;
      if (!o.soak_csv.empty() &&
          std::chrono::duration<double>(clock_type::now() - last_sample).count() >=
            o.sample_seconds) {
        for (auto const& pipeline : pipelines)
          for (auto const& slot : pipeline)
            slot->verify_contents();
        soak_sample(false);
      }
    } while (o.epoch_seconds > 0
               ? std::chrono::duration<double>(clock_type::now() - epoch_start).count() <
                   o.epoch_seconds
               : passes < o.repetitions || seconds < static_cast<double>(o.min_seconds));
    soak_sample(true);
  } catch (...) {
    done.store(true);
    gate.arrive_and_wait();
    threads.clear();
    throw;
  }
  done.store(true);
  gate.arrive_and_wait();
  threads.clear();
  sampler.request_stop();
  sampler.join();
  sample_effective();
  auto after = snapshot();
  if (o.verify && o.workload == "spill")
    for (std::size_t t = 0; t < o.threads; ++t)
      for (std::size_t i = 0; i < o.tasks; ++i) {
        auto& slot = *pipelines[t][i % depth];
        slot.select(i);
        slot.verify_contents();  // Final timed pass, not another write pass.
      }
  // Histograms are additive; remove the untimed warm-up samples.
  auto subtract = [](engine_histogram a, engine_histogram const& b) {
    for (std::size_t i = 0; i < a.buckets.size(); ++i)
      a.buckets[i] -= b.buckets[i];
    a.count -= b.count;
    a.total_ns -= b.total_ns;
    return a;
  };
  auto submit            = subtract(after.submit, before.submit);
  auto completion        = subtract(after.completion, before.completion);
  auto ops               = after.completed_ops - before.completed_ops;
  auto const write_bytes = after.async_write_bytes - before.async_write_bytes +
                           after.direct_write_bytes - before.direct_write_bytes +
                           after.buffered_write_bytes - before.buffered_write_bytes;
  auto transport_bytes = write_bytes + after.async_registered_bytes -
                         before.async_registered_bytes + after.async_unregistered_bytes -
                         before.async_unregistered_bytes + after.sync_bytes - before.sync_bytes;
  auto mean = [](std::uint64_t integral, std::uint64_t duration) {
    return duration ? static_cast<double>(integral) / static_cast<double>(duration) : 0;
  };
  auto qd_mean      = mean(measured_qd_ns, measured_accounted_ns);
  auto backlog_mean = mean(measured_backlog_qd_ns, measured_backlog_ns);
  auto quote        = [](std::string const& value) {
    std::string escaped = "\"";
    for (auto ch : value) {
      if (ch == '"') escaped += '"';
      escaped += ch;
    }
    return escaped + '"';
  };
  std::string overrides;
  for (auto const& entry : o.overrides) {
    if (!overrides.empty()) overrides += ';';
    overrides += std::to_string(static_cast<unsigned>(entry.match)) + ":" + entry.value;
  }
  auto const& platform = probe_platform();
  auto admission       = o.engine == "simulated" ? "simulated"
                         : o.engine == "legacy"  ? "legacy"
                         : !o.require_direct     ? "test_only_compat"
                         : !o.overrides.empty()  ? "operator_override"
                                                 : "direct_policy";
  std::cerr << "residence_p50_us=" << percentile(subtract(after.residence, before.residence), 50)
            << " residence_p99_us=" << percentile(subtract(after.residence, before.residence), 99)
            << '\n';
  auto cache = o.drop_cache ? "dontneed_before_each_pass"
               : o.verify   ? "verified_warm_uncontrolled"
                            : "uncontrolled";
  std::string header =
    "engine,workload,threads,tasks,verify,registration,logical_bytes,seconds,gib_s,mean_qd,backlog_"
    "mean_qd,peak_qd,Q_eff_max,S_eff_max,slot_waits,lane_waits,slot_wait_ns,lane_wait_ns,refill_"
    "gap_ns,"
    "coalesced_ops,coalesced_targets,gap_bytes,async_registered_bytes,async_unregistered_bytes,"
    "sync_bytes,engine_logical_bytes,legacy_bytes,submit_p50_us,submit_p99_us,completion_p50_us,"
    "completion_p99_us,queries_per_op,io_queries_per_op,copy_queries_per_op,stage_queries_per_op,"
    "fence_hint_queries_per_op,open_register_ms,admission,transport,coalesce,qd,op_bytes,"
    "arena_bytes,pipeline,sync_each,columns,col_dist,cache_state,reactors,passes,min_seconds,"
    "repetitions,chunk_min_bytes,chunk_max_bytes,task_bytes,single_bytes,file_bytes,file,"
    "overrides,libcufile_version,gds_available,compat_only,forced_compat,Q_eff_min,S_eff_min,"
    "guarded_destinations,async_write_bytes,direct_pwrite_bytes,buffered_pwrite_bytes,spill_mode,"
    "flush,NOT_GDS,residence_p50_us,residence_p99_us,backlog_ns,storage_qd_ns,"
    "accounted_ns,completed_ops,policy_state,policy_demotions,policy_cooldowns,policy_cooldown_s,"
    "policy_trigger,policy_trigger_us,policy_healthy_p99_us,resident_pages,slow_floor_us,stuck_ms,"
    "residence_mean_us,arena_slot_classes,arena_registered_class_mask,registrations";
  std::string classes;
  for (auto cls : o.arena_classes) {
    if (!classes.empty()) classes += ',';
    classes += std::to_string(cls);
  }
  auto const arena_state = arena ? arena->gauges() : arena_gauges{};
  std::ostringstream row;
  row << std::fixed << std::setprecision(3) << o.engine << ',' << o.workload << ',' << o.threads
      << ',' << o.tasks << ',' << o.verify << ',' << (o.unsafe_arena ? "unsafe-arena" : "slot")
      << ',' << completed_bytes << ',' << seconds << ','
      << static_cast<double>(completed_bytes) / gib / seconds << ',' << qd_mean << ','
      << backlog_mean << ',' << after.max_storage_qd << ',' << effective_qd.load() << ','
      << effective_bytes.load() << ',' << after.slot_waits - before.slot_waits << ','
      << after.lane_waits - before.lane_waits << ',' << after.slot_wait_ns - before.slot_wait_ns
      << ',' << after.lane_wait_ns - before.lane_wait_ns << ','
      << after.refill_gap_ns - before.refill_gap_ns << ','
      << after.coalesced_ops - before.coalesced_ops << ','
      << after.coalesced_targets - before.coalesced_targets << ','
      << after.gap_bytes - before.gap_bytes << ','
      << after.async_registered_bytes - before.async_registered_bytes << ','
      << after.async_unregistered_bytes - before.async_unregistered_bytes << ','
      << after.sync_bytes - before.sync_bytes << ','
      << after.enqueued_logical_bytes - before.enqueued_logical_bytes << ','
      << completed_bytes - (after.enqueued_logical_bytes - before.enqueued_logical_bytes) << ','
      << percentile(submit, 50) << ',' << percentile(submit, 99) << ','
      << percentile(completion, 50) << ',' << percentile(completion, 99) << ','
      << (ops ? static_cast<double>(after.queries - before.queries) / static_cast<double>(ops) : 0)
      << ','
      << (ops ? static_cast<double>(after.io_queries - before.io_queries) / static_cast<double>(ops)
              : 0)
      << ','
      << (ops ? static_cast<double>(after.copy_queries - before.copy_queries) /
                  static_cast<double>(ops)
              : 0)
      << ','
      << (ops ? static_cast<double>(after.stage_queries - before.stage_queries) /
                  static_cast<double>(ops)
              : 0)
      << ','
      << (ops ? static_cast<double>(after.fence_hint_queries - before.fence_hint_queries) /
                  static_cast<double>(ops)
              : 0)
      << ',' << open_ms << ',' << admission << ','
      << (o.engine == "simulated" ? "simulator"
          : o.posix_only          ? "posix"
          : o.engine == "legacy"  ? "legacy"
                                  : "libcufile")
      << ',' << o.coalesce << ',' << o.qd << ',' << o.op_bytes << ',' << o.arena_bytes << ','
      << depth << ',' << o.sync_each << ',' << o.columns << ',' << o.distribution << ',' << cache
      << ',' << o.reactors << ',' << passes << ',' << o.min_seconds << ',' << o.repetitions << ','
      << o.chunk_min << ',' << o.chunk_max << ',' << o.task_bytes << ',' << o.single_bytes << ','
      << o.file_bytes << ',' << quote(o.file) << ',' << quote(overrides) << ',' << platform.version
      << ',' << gds_available() << ',' << platform.compat_only << ',' << platform.forced_compat
      << ',' << (min_qd.load() == std::numeric_limits<unsigned>::max() ? 0 : min_qd.load()) << ','
      << (min_bytes.load() == std::numeric_limits<std::size_t>::max() ? 0 : min_bytes.load()) << ','
      << o.guarded_destinations << ',' << after.async_write_bytes - before.async_write_bytes << ','
      << after.direct_write_bytes - before.direct_write_bytes << ','
      << after.buffered_write_bytes - before.buffered_write_bytes << ',' << o.spill_mode << ','
      << o.flush << ",1," << percentile(subtract(after.residence, before.residence), 50) << ','
      << percentile(subtract(after.residence, before.residence), 99) << ',' << measured_backlog_ns
      << ',' << measured_qd_ns << ',' << measured_accounted_ns << ',' << ops << policy_csv() << ','
      << max_resident_pages << ',' << o.slow_floor_us << ',' << o.stuck_ms << ','
      << (after.residence.count > before.residence.count
            ? static_cast<double>(after.residence.total_ns - before.residence.total_ns) /
                static_cast<double>(after.residence.count - before.residence.count) / 1000.0
            : 0.0)
      << ',' << quote(classes) << ',' << unsigned(arena_state.registered_class_mask) << ','
      << arena_state.registrations;
  std::cout << header << '\n' << row.str() << '\n';
  if (!o.csv.empty() && o.csv != "-") {
    std::ofstream csv(o.csv);
    csv << header << '\n' << row.str() << '\n';
    if (!csv) throw std::runtime_error("CSV write failed");
  }
  if (engine && !transport_bytes)
    std::cerr
      << "WARNING: measured requests stayed on legacy routing; no engine transport evidence\n";
  if (o.engine == "simulated")
    std::cerr << "SIMULATED storage, NOT GDS numbers\n";
  else
    std::cerr << "No direct-path evidence collected: NOT GDS qualification\n";
  context->shutdown();
  context.reset();  // Destroy the facade before its injected engine binding becomes invalid.
  engine.reset();
  lane_resources.reset();
  if (arena) arena->release_for_maintenance();
  return 0;
}
}  // namespace
int main(int argc, char** argv)
{
  try {
    auto o = parse(argc, argv);
    if (!o.soak_minutes) return run(o);
    o.soak_csv = o.csv + ".soak.csv";
    if (std::filesystem::exists(o.soak_csv) || std::filesystem::exists(o.csv + ".ok"))
      throw std::runtime_error("use a fresh soak output path (no stale completion marker)");
    std::ofstream samples(o.soak_csv);
    samples << "elapsed_s,epoch,phase,arena_bytes,registered_bytes,unregistered_bytes,"
               "op_pool_live,lane_count,thread_count,vmrss_bytes,pinned_host_bytes,"
               "pinned_write_scratch_bytes,storage_qd,outstanding_slices,queries,completed_ops,"
               "quarantines,late_copy_failures,accounted_ns,storage_qd_ns,backlog_ns,"
               "backlog_storage_qd_ns,slot_waits,lane_waits,refill_gap_ns,reaper_active_ns,"
               "async_registered_bytes,async_unregistered_bytes,sync_bytes,residence_p99_us,"
               "policy_state,policy_demotions,policy_cooldowns,policy_cooldown_s,policy_trigger,"
               "policy_trigger_us,policy_healthy_p99_us,fd_count,device_used_bytes,verify\n";
    samples.close();
    if (!samples) throw std::runtime_error("cannot create soak CSV");
    auto base    = o.csv;
    o.soak_start = clock_type::now();
    do {
      auto remaining = o.soak_minutes * 60 -
                       std::chrono::duration<double>(clock_type::now() - o.soak_start).count();
      o.epoch_seconds = std::min(o.recreate_seconds, remaining);
      o.csv           = base + ".epoch-" + std::to_string(o.epoch) + ".csv";
      run(o);
      o.create = false;
      ++o.epoch;
    } while (std::chrono::duration<double>(clock_type::now() - o.soak_start).count() <
             o.soak_minutes * 60);
    std::ofstream complete(base + ".ok");
    complete << "verified complete\n";
    if (!complete) throw std::runtime_error("cannot write completion marker");
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "engine benchmark: " << e.what() << '\n';
    return 1;
  }
}
