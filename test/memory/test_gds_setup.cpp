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

/**
 * Test Tags:
 * [gds_setup] - Passive GPUDirect Storage setup discovery (fixture procfs /
 *               sysfs trees and cufile.json files, plus smoke checks on the host)
 */

#include "memory/detail/gds_setup.hpp"

#include <cucascade/memory/gds_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace cucascade::memory;
namespace fs = std::filesystem;

namespace {

/// A throwaway directory holding fake proc/ and sys/ trees and files.
class fake_host {
 public:
  fake_host()
    : _root(fs::temp_directory_path() /
            ("cucascade_gds_setup_" + std::to_string(::getpid()) + "_" + std::to_string(_next++)))
  {
    fs::create_directories(_root);
  }
  ~fake_host()
  {
    std::error_code ec;
    fs::remove_all(_root, ec);
  }
  fake_host(fake_host const&)            = delete;
  fake_host& operator=(fake_host const&) = delete;

  [[nodiscard]] discovery_roots roots() const { return {_root, _root}; }
  [[nodiscard]] std::string path(std::string const& rel) const { return (_root / rel).string(); }

  void write(std::string const& rel, std::string const& content) const
  {
    auto const file = _root / rel;
    fs::create_directories(file.parent_path());
    std::ofstream(file) << content;
  }

 private:
  static inline int _next = 0;
  fs::path _root;
};

/// The shape of the cufile.json NVIDIA ships, comments included.
constexpr char const* stock_cufile_json = R"({
    // NOTE : Application can override custom configuration via export CUFILE_ENV_PATH_JSON=<filepath>
    // e.g : export CUFILE_ENV_PATH_JSON="/home/<xxx>/cufile.json"
    "logging": {
        // log directory, if not enabled will create log file under current working directory
        //"dir": "/home/<xxxx>",
        // NOTICE|ERROR|WARN|INFO|DEBUG|TRACE (in decreasing order of severity)
        "level": "ERROR"
    },
    "profile": {
        // nvtx profiling on/off
        "nvtx": false,
        // cufile stats level(0-3)
        "cufile_stats": 3
    },
    "execution" : {
        // max number of workitems in the queue;
        "max_io_queue_depth": 128,
        // max number of host threads per gpu to spawn for parallel IO
        "max_io_threads" : 4,
        // enable support for parallel IO
        "parallel_io" : true,
        // minimum IO threshold before splitting the IO
        "min_io_threshold_size_kb" : 8192,
        // maximum parallelism for a single request
        "max_request_parallelism" : 4
    },
    "properties": {
        /* max IO chunk size (parameter should be multiples of 64K) used by cuFileRead/Write
           internally per IO request */
        "max_direct_io_size_kb" : 16384,
        // device memory size for reserving bounce buffers for the entire GPU
        "max_device_cache_size_kb" : 131072,
        // per-io bounce-buffer size (parameter should be multiples of 64K) ranging from 1024kb to 16384kb
        "per_buffer_cache_size_kb": 1024,
        // limit on maximum device memory size that can be pinned for a given process
        "max_device_pinned_mem_size_kb" : 33554432,
        // true or false (true will enable asynchronous io submission to nvidia-fs driver)
        "use_poll_mode" : false,
        // maximum IO request size (in KB) to use polling on
        "poll_mode_max_size_kb": 4,
        // allow compat mode, this will enable use of cuFile posix read/writes
        "allow_compat_mode": true,
        "use_pci_p2pdma": false,
        // client-side rdma addr list for user-space file-systems(e.g ["10.0.1.0", "10.0.2.0"])
        "rdma_dev_addr_list": [ "192.168.0.12", "192.168.1.12" ],
        "rdma_load_balancing_policy": "RoundRobin",
        "rdma_dynamic_routing": false,
    },
    "fs": {
        "generic": {
            // for unaligned writes, setting it to true will use posix writes
            "posix_unaligned_writes" : false
        },
        "lustre": {
            "posix_gds_min_kb" : 0
        },
        "beegfs": {
            "posix_gds_min_kb" : 4
        },
        "gpfs": {
            "gds_write_support": false,
            "gds_async_support": true
        },
    },
    "denylist": {
        // specify list of vendor driver modules to deny for nvidia-fs (e.g. ["nvme" , "nvme_rdma"])
        "drivers":  [ ],
        // specify list of block devices to prevent IO using cuFile (e.g. [ "/dev/nvme0n1" ])
        "devices": [ "/dev/nvme9n1" ],
        // specify list of mount points to prevent IO using cuFile (e.g. ["/mnt/test"])
        "mounts": [ "/scratch" ],
        // specify list of file-systems to prevent IO using cuFile (e.g ["lustre", "wekafs"])
        "filesystems": [ ]
    },
    "miscellaneous": {
        // enable only for enforcing strict checks at API level for debugging
        "api_check_aggressive": false
    }
}
)";

constexpr char const* nvfs_stats = R"(GDS Version: 1.14.1.1
NVFS statistics(ver: 4.0)
NVFS Driver(version: 2.25.6)
Mellanox PeerDirect Supported: True
IO stats: Disabled, peer IO stats: Disabled
Logging level: info

Active Shadow-Buffer (MiB): 0
Active Process: 0
Reads                           : err=0 io_state_err=0
Sparse Reads                    : n=6 io=0 holes=0 pages=0
Writes                          : err=0 io_state_err=0 pg-cache=0 pg-cache-fail=0 pg-cache-eio=0
Mmap                            : n=27 ok=27 err=0 munmap=27
Bar1-map                        : n=27 ok=27 err=0 free=27 callbacks=0 active=0
Error                           : cpu-gpu-pages=0 sg-ext=0 dma-map=0 dma-ref=0
Ops                             : Read=0 Write=0
GPU 0000:1b:00.0  uuid:87e5c586-88ed-583b-df45-fcee0f1e7917 : Registered_MiB=512 Cache_MiB=128 max_pinned_MiB=640 cross_root_port(%)=0
GPU 0000:43:00.0  uuid:5b2a1f04-1c2b-4f0e-9a51-2a8e9f0c7d11 : Registered_MiB=0 Cache_MiB=0 max_pinned_MiB=1 cross_root_port(%)=12
)";

constexpr char const* proc_modules =
  "nvidia_fs 262144 0 - Live 0x0000000000000000 (OE)\n"
  "nvidia_peermem 16384 0 - Live 0x0000000000000000 (OE)\n"
  "mmfs26 3612672 13 mmfslinux, Live 0x0000000000000000 (OE)\n"
  "mmfslinux 581632 14 mmfs26, Live 0x0000000000000000 (OE)\n"
  "ib_core 491520 9 rdma_ucm,ib_ipoib,mlx5_ib, Live 0x0000000000000000 (OE)\n"
  "rpcrdma 405504 0 - Unloading 0x0000000000000000\n"
  "lustre 1000000 0 - Loading 0x0000000000000000\n";

}  // namespace

TEST_CASE("gds setup: the stock cufile.json shape parses, comments and all", "[gds_setup]")
{
  cufile_json_settings settings;
  std::string error;
  REQUIRE(detail::parse_cufile_json(stock_cufile_json, settings, error));
  CHECK(error.empty());

  CHECK(settings.max_direct_io_size_kb == 16384U);
  CHECK(settings.max_device_cache_size_kb == 131072U);
  CHECK(settings.per_buffer_cache_size_kb == 1024U);
  CHECK(settings.max_device_pinned_mem_size_kb == 33554432U);
  CHECK(settings.use_poll_mode == false);
  CHECK(settings.poll_mode_max_size_kb == 4U);
  CHECK(settings.allow_compat_mode == true);
  CHECK(settings.use_pci_p2pdma == false);
  CHECK(settings.rdma_dev_addr_list == std::vector<std::string>{"192.168.0.12", "192.168.1.12"});
  CHECK(settings.rdma_load_balancing_policy == std::string{"RoundRobin"});
  CHECK(settings.rdma_dynamic_routing == false);

  CHECK(settings.parallel_io == true);
  CHECK(settings.max_io_threads == 4U);
  CHECK(settings.max_io_queue_depth == 128U);
  CHECK(settings.min_io_threshold_size_kb == 8192U);
  CHECK(settings.max_request_parallelism == 4U);

  CHECK(settings.gpfs_gds_write_support == false);
  CHECK(settings.gpfs_gds_async_support == true);
  CHECK(settings.lustre_posix_gds_min_kb == 0U);
  CHECK(settings.beegfs_posix_gds_min_kb == 4U);

  CHECK(settings.denied_drivers.empty());
  CHECK(settings.denied_devices == std::vector<std::string>{"/dev/nvme9n1"});
  CHECK(settings.denied_mounts == std::vector<std::string>{"/scratch"});
  CHECK(settings.denied_filesystems.empty());

  CHECK(settings.cufile_stats == 3U);
  CHECK(settings.logging_level == std::string{"ERROR"});
}

TEST_CASE("gds setup: unset, unknown and mistyped cufile.json keys stay unset", "[gds_setup]")
{
  cufile_json_settings settings;
  std::string error;
  REQUIRE(detail::parse_cufile_json(R"({
      "properties": { "max_direct_io_size_kb": "16384", "use_poll_mode": 1,
                      "per_buffer_cache_size_kb": -4, "max_device_cache_size_kb": 1.5,
                      "something_new": { "nested": [1, 2, {"deep": null}] } },
      "execution": [],
      "unrelated": "é\n\"quoted\""
    })",
                                    settings,
                                    error));
  CHECK_FALSE(settings.max_direct_io_size_kb.has_value());
  CHECK_FALSE(settings.use_poll_mode.has_value());
  CHECK_FALSE(settings.per_buffer_cache_size_kb.has_value());
  CHECK_FALSE(settings.max_device_cache_size_kb.has_value());
  CHECK_FALSE(settings.parallel_io.has_value());
  CHECK(settings.rdma_dev_addr_list.empty());
}

TEST_CASE("gds setup: a repeated cufile.json key takes its last value", "[gds_setup]")
{
  cufile_json_settings settings;
  std::string error;
  REQUIRE(detail::parse_cufile_json(
    R"({"properties": {"max_direct_io_size_kb": 1024, "max_direct_io_size_kb": 32768}})",
    settings,
    error));
  CHECK(settings.max_direct_io_size_kb == 32768U);
}

TEST_CASE("gds setup: malformed cufile.json is reported with its position", "[gds_setup]")
{
  struct malformed {
    char const* text;
    char const* what;
  };
  auto const sample = GENERATE(malformed{R"({"properties": {"a": 1)", "expected ',' or '}'"},
                               malformed{R"({"properties": {"a" 1}})", "expected ':'"},
                               malformed{"{\n  \"a\": tRue\n}", "invalid literal"},
                               malformed{R"({"a": 1} extra)", "unexpected trailing text"},
                               malformed{R"({"a": "unterminated})", "unterminated string"},
                               malformed{"{ /* never closed ", "unterminated comment"},
                               malformed{R"([1, 2, 3])", "not a JSON object"},
                               malformed{"", "unexpected end of text"},
                               malformed{R"({"a": 01e})", "invalid number"});
  INFO(sample.text);
  cufile_json_settings settings;
  settings.max_io_threads = 7;  // untouched on failure
  std::string error;
  CHECK_FALSE(detail::parse_cufile_json(sample.text, settings, error));
  CHECK(error.find(sample.what) != std::string::npos);
  CHECK(settings.max_io_threads == 7U);
}

TEST_CASE("gds setup: deeply nested cufile.json is refused, not recursed into", "[gds_setup]")
{
  std::string text = "{\"a\": ";
  for (int i = 0; i < 10000; ++i) {
    text += '[';
  }
  cufile_json_settings settings;
  std::string error;
  CHECK_FALSE(detail::parse_cufile_json(text, settings, error));
  CHECK(error.find("nesting too deep") != std::string::npos);
}

TEST_CASE("gds setup: nvidia-fs stats, version and module rows parse", "[gds_setup]")
{
  nvidia_fs_info info;
  detail::parse_nvidia_fs_stats(nvfs_stats, info);
  CHECK(info.gds_version == "1.14.1.1");
  CHECK(info.version == "2.25.6");
  CHECK(info.peer_direct_supported == true);
  CHECK(info.io_stats_enabled == false);
  REQUIRE(info.gpus.size() == 2);
  CHECK(info.gpus[0].pci_bus_id == "0000:1b:00.0");
  CHECK(info.gpus[0].registered_mib == 512U);
  CHECK(info.gpus[0].cache_mib == 128U);
  CHECK(info.gpus[0].max_pinned_mib == 640U);
  CHECK(info.gpus[1].cross_root_port_pct == 12U);

  // A version learned elsewhere (sysfs) is kept.
  nvidia_fs_info known;
  known.version = "9.9.9";
  detail::parse_nvidia_fs_stats(nvfs_stats, known);
  CHECK(known.version == "9.9.9");

  CHECK(detail::parse_nvidia_fs_version("version: 2.20.5\n") == "2.20.5");
  CHECK(detail::parse_nvidia_fs_version("2.17.3\n") == "2.17.3");
  CHECK(detail::parse_nvidia_fs_version("") == "");

  auto const live = detail::live_modules(proc_modules);
  CHECK(live ==
        std::vector<std::string>{"nvidia_fs", "nvidia_peermem", "mmfs26", "mmfslinux", "ib_core"});

  CHECK(detail::force_compat_value("true"));
  CHECK(detail::force_compat_value("TRUE"));
  CHECK(detail::force_compat_value("True1"));
  CHECK_FALSE(detail::force_compat_value("1"));
  CHECK_FALSE(detail::force_compat_value("tr"));
  CHECK_FALSE(detail::force_compat_value(nullptr));
}

TEST_CASE("gds setup: discovery reads a fixture host", "[gds_setup]")
{
  fake_host host;
  host.write("proc/modules", proc_modules);
  host.write("sys/module/nvidia_fs/version", "2.26.1\n");
  host.write("proc/driver/nvidia-fs/stats", nvfs_stats);
  host.write("etc/cufile.json", stock_cufile_json);

  auto const info = discover_gds_setup(host.roots(), host.path("etc/cufile.json"), std::string{});
  CHECK(info.nvidia_fs.loaded);
  CHECK(info.kernel_path_present());
  CHECK(info.nvidia_fs.version == "2.26.1");  // sysfs wins over the stats file
  CHECK(info.nvidia_fs.gds_version == "1.14.1.1");
  CHECK(info.nvidia_fs.peer_direct_supported == true);
  CHECK(info.nvidia_fs.gpus.size() == 2);
  CHECK(info.peer_memory_loaded);
  CHECK(info.rdma_core_loaded);
  CHECK(info.gpfs_client_loaded);
  CHECK_FALSE(info.lustre_client_loaded);  // still loading
  CHECK_FALSE(info.nfs_rdma_loaded);       // unloading
  CHECK_FALSE(info.beegfs_client_loaded);
  CHECK_FALSE(info.wekafs_client_loaded);

  CHECK(info.cufile_json.path == host.path("etc/cufile.json"));
  CHECK_FALSE(info.cufile_json.from_environment);
  CHECK(info.cufile_json.found);
  REQUIRE(info.cufile_json.parsed);
  CHECK(info.cufile_json.error.empty());
  CHECK(info.cufile_json.settings.max_direct_io_size_kb == 16384U);
}

TEST_CASE("gds setup: a host without GDS reports nothing loaded", "[gds_setup]")
{
  fake_host host;
  host.write("proc/modules", "ena 155648 0 - Live 0x0\n");

  auto const info = discover_gds_setup(host.roots(), host.path("etc/cufile.json"), std::string{});
  CHECK_FALSE(info.nvidia_fs.loaded);
  CHECK_FALSE(info.kernel_path_present());
  CHECK(info.nvidia_fs.version.empty());
  CHECK_FALSE(info.nvidia_fs.peer_direct_supported.has_value());
  CHECK_FALSE(info.peer_memory_loaded);
  CHECK_FALSE(info.gpfs_client_loaded);
  // No file at all: cuFile runs on its defaults, which the (empty) settings are.
  CHECK(info.cufile_json.examined);
  CHECK_FALSE(info.cufile_json.found);
  CHECK(info.cufile_json.parsed);
  CHECK(info.cufile_json.settings.rdma_dev_addr_list.empty());
  CHECK(info.cufile_json.error.find("not found") != std::string::npos);
}

TEST_CASE("gds setup: the override file is merged over the system file key by key", "[gds_setup]")
{
  fake_host host;
  host.write("etc/cufile.json", R"({
      "properties": { "max_direct_io_size_kb": 16384, "rdma_dev_addr_list": ["10.0.0.11"] },
      "denylist": { "mounts": ["/scratch"] },
      "logging": { "level": "ERROR" } })");
  host.write("override.json", R"({
      "properties": { "max_direct_io_size_kb": 4096 },
      "logging": { "level": "WARN" } })");

  auto const merged = read_cufile_config(host.path("etc/cufile.json"), host.path("override.json"));
  REQUIRE(merged.parsed);
  CHECK(merged.found);
  CHECK(merged.override_path == host.path("override.json"));
  CHECK(merged.settings.max_direct_io_size_kb == 4096U);  // the override wins
  CHECK(merged.settings.rdma_dev_addr_list == std::vector<std::string>{"10.0.0.11"});  // kept
  CHECK(merged.settings.denied_mounts == std::vector<std::string>{"/scratch"});
  CHECK(merged.settings.logging_level == std::string{"WARN"});

  // A missing override is skipped, like cuFile does.
  auto const base_only =
    read_cufile_config(host.path("etc/cufile.json"), host.path("missing.json"));
  REQUIRE(base_only.parsed);
  CHECK(base_only.settings.max_direct_io_size_kb == 16384U);

  // Discovery takes the override path from the caller when given one.
  auto const info =
    discover_gds_setup(host.roots(), host.path("etc/cufile.json"), host.path("override.json"));
  CHECK(info.cufile_json.settings.max_direct_io_size_kb == 4096U);
  CHECK_FALSE(info.cufile_json.from_environment);
}

TEST_CASE("gds setup: an unparsable cufile.json is found but not parsed", "[gds_setup]")
{
  fake_host host;
  host.write("cufile.json", "{ \"properties\": { \"max_direct_io_size_kb\": 16384, }");
  auto const file = read_cufile_config(host.path("cufile.json"));
  CHECK(file.found);
  CHECK_FALSE(file.parsed);
  CHECK(file.error.find("not valid JSON") != std::string::npos);
  CHECK_FALSE(file.settings.max_direct_io_size_kb.has_value());
}

TEST_CASE("gds setup: discovery on this host never throws", "[gds_setup]")
{
  gds_setup_info info;
  REQUIRE_NOTHROW(info = discover_gds_setup());
  // Whatever the host has, the configuration file is one cuFile would read.
  CHECK_FALSE(info.cufile_json.path.empty());
  CHECK(info.cufile_json.examined);
  // No file at all means cuFile's defaults, which always "parse".
  if (!info.cufile_json.found) { CHECK(info.cufile_json.parsed); }
  INFO("nvidia-fs loaded " << info.nvidia_fs.loaded << ", cufile.json " << info.cufile_json.path
                           << (info.cufile_json.found ? "" : " (absent)"));
}

TEST_CASE("gds setup: path discovery reports the file's I/O and block sizes", "[gds_setup]")
{
  fake_host host;
  host.write("data.bin", std::string(8192, 'x'));
  auto const info = topology_discovery::discover_storage_for_path(host.path("data.bin"));
  REQUIRE(info.status != path_storage_status::not_found);
  CHECK(info.preferred_io_size > 0);
  CHECK(info.filesystem_block_size > 0);
}
