/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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
 * The cuFile tuning sweep (hidden; run explicitly with "[.cufile-tune]").
 *
 * Run it on the deployment host against a large file on the storage SiriusDB
 * reads (an existing file; it is only read).  It prints what the host's GDS setup
 * is, then measures what the backend's knobs need:
 *
 *   A. latency by read size, direct vs staged, aligned vs not, one read in flight
 *      -- the per-read latency Little's law needs (in flight = rate x latency);
 *   B. on one GPU, Parquet-shaped reads (1-8 MiB at arbitrary offsets into
 *      arbitrary device addresses) for every reactors-per-GPU depth x staging
 *      piece size x staging mode -- where throughput stops growing with depth;
 *   C. every GPU at once with the best setting -- whether GPUs share a bottleneck
 *      (NIC, PCIe switch, storage).
 *
 * and ends with the configuration to use.  Environment:
 *
 *   CUCASCADE_IO_TUNE_FILE        file to read (required), ideally several times
 *                                 larger than any cache in the path
 *   CUCASCADE_IO_TUNE_SECONDS     seconds per measured point (default 4)
 *   CUCASCADE_IO_TUNE_DEPTHS      reactors-per-GPU values (default 1,2,4,8,16,32,64)
 *   CUCASCADE_IO_TUNE_PIECES_MIB  staging piece sizes in MiB (default 4,8,16)
 *   CUCASCADE_IO_TUNE_MODES       staging modes to compare (default auto,never)
 *   CUCASCADE_IO_TUNE_SIZES_KIB   phase A read sizes in KiB (default 64,256,1024,4096,8192,16384)
 *   CUCASCADE_IO_TUNE_CHUNK_KIB   Parquet chunk size range "min,max" in KiB (default 1024,8192)
 *   CUCASCADE_IO_TUNE_GPUS        CUDA devices for phase C (default all)
 *   CUCASCADE_IO_TUNE_SUBMITTERS  submitting threads per GPU (default 4)
 *   CUCASCADE_IO_TUNE_SKIP        phases to skip, e.g. "A" or "A,C"
 */

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/gds_topology.hpp>
#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cufile = cucascade::io::cufile;
namespace memory = cucascade::memory;

namespace {

using namespace std::chrono_literals;
using clock_type = std::chrono::steady_clock;

constexpr std::size_t block = cucascade::io::IO_BLOCK_SIZE;

[[nodiscard]] std::vector<std::size_t> env_list(char const* name, std::vector<std::size_t> fallback)
{
  char const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  std::vector<std::size_t> out;
  std::stringstream stream{value};
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) out.push_back(std::stoull(item));
  }
  return out.empty() ? fallback : out;
}

[[nodiscard]] std::vector<std::string> env_words(char const* name,
                                                 std::vector<std::string> fallback)
{
  char const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  std::vector<std::string> out;
  std::stringstream stream{value};
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out.empty() ? fallback : out;
}

[[nodiscard]] bool skipped(char phase)
{
  char const* value = std::getenv("CUCASCADE_IO_TUNE_SKIP");
  return value != nullptr && std::string{value}.find(phase) != std::string::npos;
}

[[nodiscard]] cufile::staging mode_of(std::string const& word)
{
  if (word == "always") return cufile::staging::always;
  if (word == "never") return cufile::staging::never;
  return cufile::staging::automatic;
}

[[nodiscard]] char const* name_of(cufile::staging mode)
{
  switch (mode) {
    case cufile::staging::always: return "always";
    case cufile::staging::never: return "never";
    case cufile::staging::automatic: break;
  }
  return "auto";
}

/// xorshift64: deterministic, cheap, independent per thread.
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

void drop_cache(std::string const& path)
{
  int const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;
  std::ignore = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
}

[[nodiscard]] double gbps(std::size_t bytes, std::chrono::duration<double> seconds)
{
  return seconds.count() > 0 ? static_cast<double>(bytes) / seconds.count() / 1e9 : 0.0;
}

// ---------------------------------------------------------------------------
// Environment report
// ---------------------------------------------------------------------------

void print_environment(std::string const& path,
                       std::optional<memory::system_topology_info> const& topology)
{
  auto const& platform = cufile::probe_platform();
  std::printf("== cuFile platform\n");
  std::printf("library %s, driver %s, version %d, nvidia-fs %u.%u\n",
              platform.library_available ? "loaded" : "MISSING",
              platform.driver_opened ? "open" : "NOT OPEN",
              platform.version,
              platform.nvfs_major_version,
              platform.nvfs_minor_version);
  std::printf(
    "dstatusflags 0x%x (compat-only %s, forced compat %s), fflags 0x%x, dcontrolflags 0x%x\n",
    platform.dstatusflags,
    platform.compat_only ? "YES" : "no",
    platform.forced_compat ? "YES" : "no",
    platform.fflags,
    platform.dcontrolflags);
  std::printf(
    "max_direct_io %zu KiB, bounce cache %zu KiB/GPU in %zu KiB buffers, pinned limit %zu KiB/GPU, "
    "poll threshold %zu KiB, batch %u\n",
    platform.max_direct_io_size_kb,
    platform.max_device_cache_size_kb,
    platform.per_buffer_cache_size_kb,
    platform.max_device_pinned_mem_size_kb,
    platform.poll_threshold_size_kb,
    platform.max_batch_io_size);
  auto const& p = platform.parameters;
  auto opt_size = [](std::optional<std::size_t> const& v) {
    return v ? std::to_string(*v) : std::string{"?"};
  };
  auto opt_bool = [](std::optional<bool> const& v) {
    return v ? (*v ? std::string{"on"} : std::string{"off"}) : std::string{"?"};
  };
  std::printf(
    "parameters%s: parallel_io %s, max_io_threads %s, max_io_queue_depth %s, "
    "min_io_threshold %s KiB, max_request_parallelism %s, poll %s, compat %s, rdma transport %s\n",
    p.available ? "" : " (API unavailable)",
    opt_bool(p.parallel_io).c_str(),
    opt_size(p.max_io_threads).c_str(),
    opt_size(p.max_io_queue_depth).c_str(),
    opt_size(p.min_io_threshold_size_kb).c_str(),
    opt_size(p.max_request_parallelism).c_str(),
    opt_bool(p.use_poll_mode).c_str(),
    opt_bool(p.allow_compat_mode).c_str(),
    p.rdma_transport ? p.rdma_transport->c_str() : "?");

  auto const verdict = cufile::probe_path(path);
  std::printf("== %s\n%s on %s (%s), block size %zu, preferred I/O %zu: %s -- %s\n",
              path.c_str(),
              verdict.storage.filesystem_type.c_str(),
              verdict.storage.mount_point.c_str(),
              verdict.storage.mount_source.c_str(),
              verdict.storage.filesystem_block_size,
              verdict.storage.preferred_io_size,
              verdict.capability == cufile::path_capability::direct_candidate ? "DIRECT CANDIDATE"
              : verdict.capability == cufile::path_capability::compatibility_only ? "COMPAT ONLY"
              : verdict.capability == cufile::path_capability::unavailable        ? "UNAVAILABLE"
                                                                                  : "UNKNOWN",
              verdict.reason.c_str());

  if (!topology) {
    std::printf("== topology: not discovered\n");
    return;
  }
  auto const& gds = topology->gds;
  std::printf(
    "== GDS setup: nvidia_fs %s %s (PeerDirect %s), nvidia_peermem %s, ib_core %s, GPFS %s, "
    "Lustre %s, BeeGFS %s, WekaFS %s, NFS/RDMA %s, CUFILE_FORCE_COMPAT_MODE %s\n",
    gds.nvidia_fs.loaded ? "loaded" : "absent",
    gds.nvidia_fs.version.c_str(),
    gds.nvidia_fs.peer_direct_supported ? (*gds.nvidia_fs.peer_direct_supported ? "yes" : "no")
                                        : "?",
    gds.peer_memory_loaded ? "loaded" : "absent",
    gds.rdma_core_loaded ? "loaded" : "absent",
    gds.gpfs_client_loaded ? "loaded" : "absent",
    gds.lustre_client_loaded ? "loaded" : "absent",
    gds.beegfs_client_loaded ? "loaded" : "absent",
    gds.wekafs_client_loaded ? "loaded" : "absent",
    gds.nfs_rdma_loaded ? "loaded" : "absent",
    gds.force_compat_env ? "SET" : "unset");
  auto const& json = gds.cufile_json;
  std::printf("cufile.json %s: %s", json.path.c_str(), json.parsed ? "parsed" : json.error.c_str());
  if (json.parsed) {
    auto const& s = json.settings;
    std::string addresses;
    for (auto const& address : s.rdma_dev_addr_list) {
      addresses += (addresses.empty() ? "" : " ") + address;
    }
    std::printf(
      "; max_direct_io %s KiB, bounce %s KiB / %s KiB, pinned %s KiB, poll %s, parallel_io %s, "
      "max_io_threads %s, rdma_dev_addr_list [%s] policy %s",
      opt_size(s.max_direct_io_size_kb).c_str(),
      opt_size(s.max_device_cache_size_kb).c_str(),
      opt_size(s.per_buffer_cache_size_kb).c_str(),
      opt_size(s.max_device_pinned_mem_size_kb).c_str(),
      opt_bool(s.use_poll_mode).c_str(),
      opt_bool(s.parallel_io).c_str(),
      opt_size(s.max_io_threads).c_str(),
      addresses.c_str(),
      s.rdma_load_balancing_policy ? s.rdma_load_balancing_policy->c_str() : "?");
  }
  std::printf("\n");
  for (auto const& gpu : topology->gpus) {
    std::string nics;
    for (auto const& name : gpu.network_devices) {
      nics += (nics.empty() ? "" : ",") + name;
    }
    std::printf(
      "GPU %u %s %s: NUMA %d, CPUs %s, memory %zu MiB, BAR1 %zu/%zu MiB free, NICs [%s] path %d\n",
      gpu.id,
      gpu.name.c_str(),
      gpu.pci_bus_id.c_str(),
      gpu.numa_node,
      gpu.cpu_affinity_list.c_str(),
      gpu.memory_total_bytes >> 20,
      gpu.bar1_free_bytes >> 20,
      gpu.bar1_total_bytes >> 20,
      nics.c_str(),
      gpu.network_device_path ? static_cast<int>(*gpu.network_device_path) : -1);
  }
  for (auto const& nic : topology->network_devices) {
    std::printf("NIC %s %s: NUMA %d, %s, %u active port(s), %u Gb/s, if %s\n",
                nic.name.c_str(),
                nic.pci_bus_id.c_str(),
                nic.numa_node,
                nic.link_layer.c_str(),
                nic.active_ports,
                nic.rate_gbps,
                nic.net_interface.c_str());
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Pools and workloads
// ---------------------------------------------------------------------------

/// A started pool planned for @p devices only (exactly as the datasource
/// factory would build it), shut down on scope exit; its staging is released
/// afterwards so the next point can use another slot size.
class tuned_pool {
 public:
  tuned_pool(cufile::config cfg, std::vector<int> const& devices) : _devices(devices)
  {
    cfg.devices = devices;
    _ioctx      = cufile::make_cufile_ioctx(0, cfg);
    for (auto const& group : _ioctx->plan().groups) {
      if (cfg.staging_mode != cufile::staging::never &&
          group.staging_slots < group.n_reactors * cfg.staging_slots_per_reactor) {
        std::printf("  note: GPU %d gets %zu of %zu staging slots (%s)\n",
                    group.device,
                    group.staging_slots,
                    group.n_reactors * cfg.staging_slots_per_reactor,
                    group.reason.c_str());
      }
    }
    _piece = _ioctx->plan().staging_piece_bytes != 0 ? _ioctx->plan().staging_piece_bytes
                                                     : cfg.staging_piece_bytes;
    _ioctx->start();
  }
  ~tuned_pool()
  {
    _ioctx->shutdown();
    _ioctx.reset();
    for (int device : _devices) {
      std::ignore = cufile::release_staging(device);
    }
  }
  tuned_pool(tuned_pool const&)            = delete;
  tuned_pool& operator=(tuned_pool const&) = delete;

  [[nodiscard]] cufile::cufile_ioctx& ioctx() noexcept { return *_ioctx; }
  [[nodiscard]] std::size_t piece() const noexcept { return _piece; }

 private:
  std::vector<int> _devices;
  std::shared_ptr<cufile::cufile_ioctx> _ioctx;
  std::size_t _piece{0};
};

struct workload {
  std::size_t chunk_min{1UL << 20};
  std::size_t chunk_max{8UL << 20};
  std::size_t per_call{32};
  std::size_t submitters{4};
};

/// Parquet-shaped reads on @p devices for @p seconds: each submitter keeps one
/// vectored read of @c per_call chunks in flight on its own stream into its own
/// buffer.  Returns the bytes read per device and the elapsed time.
[[nodiscard]] std::pair<std::vector<std::size_t>, std::chrono::duration<double>> run_parquet(
  cufile::cufile_ioctx& ioctx,
  cucascade::io::io_object const& object,
  std::vector<int> const& devices,
  workload const& load,
  std::chrono::duration<double> seconds)
{
  auto const file_size = object.size();
  std::vector<std::size_t> per_device(devices.size(), 0);
  std::atomic<bool> failed{false};
  std::string failure;
  std::mutex failure_mutex;
  std::vector<std::thread> threads;
  auto const start    = clock_type::now();
  auto const deadline = start + std::chrono::duration_cast<clock_type::duration>(seconds);
  std::vector<std::atomic<std::size_t>> bytes(devices.size());
  for (std::size_t d = 0; d < devices.size(); ++d) {
    for (std::size_t t = 0; t < load.submitters; ++t) {
      threads.emplace_back([&, d, t] {
        try {
          rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{devices[d]}};
          rmm::cuda_stream stream;
          rmm::device_buffer buffer(load.per_call * (load.chunk_max + 512), stream);
          stream.synchronize();
          auto* const base = static_cast<std::uint8_t*>(buffer.data());
          rng next{0x9E3779B97F4A7C15ULL * (d * 131 + t + 1)};
          while (clock_type::now() < deadline && !failed.load()) {
            std::vector<cucascade::io::slice> slices;
            std::size_t expected = 0;
            std::size_t placed   = 0;
            for (std::size_t i = 0; i < load.per_call; ++i) {
              auto const size   = load.chunk_min + next() % (load.chunk_max - load.chunk_min + 1);
              auto const offset = next() % (file_size - size);
              auto const where  = (placed + 255) / 256 * 256 + (next() & 1);
              slices.emplace_back(offset, size, base + where);
              placed = where + size;
              expected += size;
            }
            auto got = ioctx.device_readv_async_io(object, slices, stream).get(600s);
            if (got != expected) throw std::runtime_error("short vectored read");
            bytes[d].fetch_add(got);
          }
        } catch (std::exception const& e) {
          failed = true;
          std::lock_guard lock(failure_mutex);
          failure = e.what();
        }
      });
    }
  }
  for (auto& thread : threads) {
    thread.join();
  }
  auto const elapsed = clock_type::now() - start;
  if (failed) throw std::runtime_error(failure);
  for (std::size_t d = 0; d < devices.size(); ++d) {
    per_device[d] = bytes[d].load();
  }
  return {per_device, std::chrono::duration<double>(elapsed)};
}

void print_paths(cufile::reactor_stats const& stats)
{
  std::printf(" [staged %.1f GB (%llu early), direct %.1f GB, posix %.1f GB, cuFile failures %llu]",
              static_cast<double>(stats.staged_bytes) / 1e9,
              static_cast<unsigned long long>(stats.early_staged_reads),
              static_cast<double>(stats.direct_bytes) / 1e9,
              static_cast<double>(stats.posix_device_bytes) / 1e9,
              static_cast<unsigned long long>(stats.cufile_failures));
}

struct point {
  std::size_t depth{0};
  std::size_t piece{0};
  cufile::staging mode{cufile::staging::automatic};
  double gbps{0.0};
};

}  // namespace

TEST_CASE("cufile tuning sweep", "[.cufile-tune]")
{
  char const* path_env = std::getenv("CUCASCADE_IO_TUNE_FILE");
  int device_count     = 0;
  if (path_env == nullptr || cudaGetDeviceCount(&device_count) != cudaSuccess ||
      device_count == 0) {
    SKIP("set CUCASCADE_IO_TUNE_FILE (and have a GPU) to run the sweep");
  }
  std::string const path = path_env;
  std::error_code ec;
  REQUIRE(std::filesystem::is_regular_file(path, ec));
  auto const file_bytes = std::filesystem::file_size(path, ec);
  REQUIRE(file_bytes >= (256UL << 20));

  auto const seconds = std::chrono::duration<double>(
    static_cast<double>(env_list("CUCASCADE_IO_TUNE_SECONDS", {4}).front()));
  auto const depths = env_list("CUCASCADE_IO_TUNE_DEPTHS", {1, 2, 4, 8, 16, 32, 64});
  auto const pieces = env_list("CUCASCADE_IO_TUNE_PIECES_MIB", {4, 8, 16});
  std::vector<cufile::staging> modes;
  for (auto const& word : env_words("CUCASCADE_IO_TUNE_MODES", {"auto", "never"})) {
    modes.push_back(mode_of(word));
  }
  auto const sizes = env_list("CUCASCADE_IO_TUNE_SIZES_KIB", {64, 256, 1024, 4096, 8192, 16384});
  auto const chunk = env_list("CUCASCADE_IO_TUNE_CHUNK_KIB", {1024, 8192});
  workload load;
  load.chunk_min  = chunk.front() << 10;
  load.chunk_max  = chunk.back() << 10;
  load.submitters = env_list("CUCASCADE_IO_TUNE_SUBMITTERS", {4}).front();
  std::vector<int> gpus;
  for (auto const id : env_list("CUCASCADE_IO_TUNE_GPUS", {})) {
    gpus.push_back(static_cast<int>(id));
  }
  if (gpus.empty()) {
    for (int d = 0; d < device_count; ++d) {
      gpus.push_back(d);
    }
  }

  std::optional<memory::system_topology_info> topology;
  {
    memory::topology_discovery discovery;
    if (discovery.discover(memory::NetworkDeviceVerification::EXISTS_ACTIVE)) {
      topology = discovery.get_topology();
    }
  }
  print_environment(path, topology);

  cufile::config base;
  base.storage_hint_path = path;
  base.gds_threshold     = 64UL << 10;
  {
    auto const plan =
      cufile::plan_cufile_pool(base, topology ? &*topology : nullptr, cufile::probe_platform());
    std::printf("== default plan: %s\n", cufile::describe(plan).c_str());
    for (auto const& warning : plan.warnings) {
      std::printf("WARNING: %s\n", warning.c_str());
    }
  }

  // -- A: latency by read size, one read in flight -----------------------------
  if (!skipped('A')) {
    std::printf("\n== A. latency, 1 reactor, 1 read in flight (GPU %d)\n", gpus.front());
    std::printf(
      "%10s %-17s %9s %9s %9s %9s\n", "size", "path", "p50 ms", "p90 ms", "p99 ms", "GB/s");
    struct variant {
      char const* name;
      cufile::staging mode;
      bool aligned;
    };
    for (auto const kib : sizes) {
      auto const size = kib << 10;
      if (size * 4 > file_bytes) continue;
      for (auto const& v : {variant{"direct aligned", cufile::staging::never, true},
                            variant{"direct unaligned", cufile::staging::never, false},
                            variant{"staged unaligned", cufile::staging::always, false}}) {
        try {
          auto cfg                = base;
          cfg.reactors_per_device = 1;
          cfg.staging_mode        = v.mode;
          cfg.staging_piece_bytes = std::max<std::size_t>(size, 1UL << 20);
          drop_cache(path);
          tuned_pool pool{cfg, {gpus.front()}};
          auto object = pool.ioctx().open_io_object(path);
          rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{gpus.front()}};
          rmm::cuda_stream stream;
          rmm::device_buffer buffer(size + 512, stream);
          stream.synchronize();
          auto* const dst = static_cast<std::uint8_t*>(buffer.data()) + (v.aligned ? 0 : 13);
          rng next{size};
          std::vector<double> latencies;
          std::size_t total = 0;
          auto const begin  = clock_type::now();
          while (clock_type::now() - begin < seconds / 2 || latencies.size() < 8) {
            auto offset = next() % (file_bytes - size);
            if (v.aligned) offset &= ~(block - 1);
            auto const t0 = clock_type::now();
            total +=
              pool.ioctx().device_read_async_io(*object, offset, size, dst, stream).get(600s);
            latencies.push_back(
              std::chrono::duration<double, std::milli>(clock_type::now() - t0).count());
            if (latencies.size() >= 100000) break;
          }
          auto const elapsed = clock_type::now() - begin;
          std::sort(latencies.begin(), latencies.end());
          auto pct = [&latencies](double q) {
            return latencies[std::min(
              latencies.size() - 1,
              static_cast<std::size_t>(q * static_cast<double>(latencies.size())))];
          };
          std::printf("%7zu KiB %-17s %9.3f %9.3f %9.3f %9.2f",
                      kib,
                      v.name,
                      pct(0.5),
                      pct(0.9),
                      pct(0.99),
                      gbps(total, elapsed));
          print_paths(pool.ioctx().stats());
          std::printf("\n");
        } catch (std::exception const& e) {
          std::printf("%7zu KiB %-17s failed: %s\n", kib, v.name, e.what());
        }
        std::fflush(stdout);
      }
    }
  }

  // -- B: depth x piece x staging on one GPU -----------------------------------
  std::vector<point> results;
  if (!skipped('B')) {
    std::printf(
      "\n== B. Parquet-shaped reads (%zu-%zu KiB, unaligned), GPU %d, %zu submitters x %zu "
      "ranges\n",
      load.chunk_min >> 10,
      load.chunk_max >> 10,
      gpus.front(),
      load.submitters,
      load.per_call);
    std::printf("%7s %8s %7s %9s\n", "depth", "piece", "staging", "GB/s");
    for (auto const mode : modes) {
      for (auto const mib : pieces) {
        for (auto const depth : depths) {
          auto cfg                         = base;
          cfg.reactors_per_device          = depth;
          cfg.staging_mode                 = mode;
          cfg.staging_piece_bytes          = mib << 20;
          cfg.max_staging_bytes_per_device = 0;  // only the device limits bound the sweep
          auto run_load                    = load;
          run_load.per_call = std::max<std::size_t>(load.per_call, 2 * depth / load.submitters);
          try {
            drop_cache(path);
            tuned_pool pool{cfg, {gpus.front()}};
            auto object = pool.ioctx().open_io_object(path);
            auto [bytes, elapsed] =
              run_parquet(pool.ioctx(), *object, {gpus.front()}, run_load, seconds);
            point p{depth, pool.piece(), mode, gbps(bytes.front(), elapsed)};
            results.push_back(p);
            std::printf("%7zu %5zu MiB %7s %9.2f", depth, p.piece >> 20, name_of(mode), p.gbps);
            print_paths(pool.ioctx().stats());
            std::printf("\n");
          } catch (std::exception const& e) {
            std::printf("%7zu %5zu MiB %7s failed: %s\n", depth, mib, name_of(mode), e.what());
          }
          std::fflush(stdout);
        }
      }
    }
  }

  // -- the best setting: the fastest point, at the shallowest depth within 95% --
  point best;
  for (auto const& p : results) {
    if (p.gbps > best.gbps) best = p;
  }
  point chosen = best;
  for (auto const& p : results) {
    if (p.mode == best.mode && p.piece == best.piece && p.gbps >= 0.95 * best.gbps &&
        p.depth < chosen.depth) {
      chosen = p;
    }
  }

  // -- C: every GPU at once ---------------------------------------------------
  if (!skipped('C') && chosen.depth != 0) {
    std::printf("\n== C. every GPU (%zu) at depth %zu, piece %zu MiB, staging %s\n",
                gpus.size(),
                chosen.depth,
                chosen.piece >> 20,
                name_of(chosen.mode));
    try {
      auto cfg                         = base;
      cfg.reactors_per_device          = chosen.depth;
      cfg.staging_mode                 = chosen.mode;
      cfg.staging_piece_bytes          = chosen.piece;
      cfg.max_staging_bytes_per_device = 0;
      drop_cache(path);
      tuned_pool pool{cfg, gpus};
      auto object       = pool.ioctx().open_io_object(path);
      auto run_load     = load;
      run_load.per_call = std::max<std::size_t>(load.per_call, 2 * chosen.depth / load.submitters);
      auto [bytes, elapsed] = run_parquet(pool.ioctx(), *object, gpus, run_load, seconds);
      std::size_t total     = 0;
      for (std::size_t d = 0; d < gpus.size(); ++d) {
        std::printf("GPU %d: %.2f GB/s\n", gpus[d], gbps(bytes[d], elapsed));
        total += bytes[d];
      }
      std::printf("node: %.2f GB/s (one GPU alone: %.2f GB/s)", gbps(total, elapsed), chosen.gbps);
      print_paths(pool.ioctx().stats());
      std::printf("\n");
    } catch (std::exception const& e) {
      std::printf("failed: %s\n", e.what());
    }
  }

  // -- recommendation -----------------------------------------------------------
  if (chosen.depth != 0) {
    std::printf(
      "\n== recommended (fastest %.2f GB/s at depth %zu; depth %zu reaches %.2f GB/s, within 5%%)\n"
      "io:\n  backend: cufile\n  cufile_n_reactors: 0\n  cufile:\n"
      "    storage_hint_path: %s\n    reactors_per_device: %zu\n    staging_mode: %s\n"
      "    staging_piece_bytes: %zuMiB\n",
      best.gbps,
      best.depth,
      chosen.depth,
      chosen.gbps,
      path.c_str(),
      chosen.depth,
      name_of(chosen.mode),
      chosen.piece >> 20);
    std::printf(
      "(staging memory per GPU: about %zu MiB = depth x %zu slots x %zu MiB; set "
      "max_staging_bytes_per_device at least that, and leave it out of the RMM pool)\n",
      chosen.depth * cufile::config{}.staging_slots_per_reactor * ((chosen.piece >> 20) + 1),
      cufile::config{}.staging_slots_per_reactor,
      (chosen.piece >> 20) + 1);
  }
  std::fflush(stdout);
}
