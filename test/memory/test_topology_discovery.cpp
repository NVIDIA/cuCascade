/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * [memory_space] - Basic memory space functionality tests
 * [threading] - Multi-threaded tests
 * [gpu] - GPU-specific tests requiring CUDA
 * [.multi-device] - Tests requiring multiple GPU devices (hidden by default)
 *
 * Running tests:
 * - Default (includes single GPU): ./test_executable
 * - Include multi-device tests: ./test_executable "[.multi-device]"
 * - Exclude multi-device tests: ./test_executable "~[.multi-device]"
 * - Run all tests: ./test_executable "[memory_space]"
 */

#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <ifaddrs.h>
#include <nvml.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace cucascade::memory;
namespace fs = std::filesystem;

namespace {
struct ScopedEnvVar {
  explicit ScopedEnvVar(std::string name, std::string value) : name_(std::move(name))
  {
    const char* existing = std::getenv(name_.c_str());
    if (existing) {
      had_value_ = true;
      old_value_ = existing;
    }
    setenv(name_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar()
  {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(ScopedEnvVar const&)            = delete;
  ScopedEnvVar& operator=(ScopedEnvVar const&) = delete;

 private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};
}  // namespace

// Test topology discovery
TEST_CASE("Topology Discovery", "[hw_topology]")
{
  topology_discovery discovery;

  // Call discover() method
  bool success = discovery.discover();

  // Verify discovery was successful
  REQUIRE(success == true);
  REQUIRE(discovery.is_discovered() == true);

  // Get the topology information
  const auto& topology = discovery.get_topology();

  // Verify basic topology information
  REQUIRE(!topology.hostname.empty());
  REQUIRE(topology.num_gpus >= 0);
  REQUIRE(topology.num_numa_nodes >= 0);
  REQUIRE(topology.num_network_devices >= 0);

  // Verify GPU information consistency
  REQUIRE(topology.gpus.size() == topology.num_gpus);

  // If GPUs are present, verify their information
  if (topology.num_gpus > 0) {
    for (const auto& gpu : topology.gpus) {
      REQUIRE(!gpu.name.empty());
      REQUIRE(!gpu.pci_bus_id.empty());
      REQUIRE(!gpu.uuid.empty());
      // NUMA node may be -1 if unknown
      REQUIRE(gpu.numa_node >= -1);
    }
  }

  // Verify network device information consistency
  REQUIRE(topology.network_devices.size() == static_cast<size_t>(topology.num_network_devices));

  // If network devices are present, verify their information
  for (const auto& net_dev : topology.network_devices) {
    REQUIRE(!net_dev.name.empty());
    // NUMA node may be -1 if unknown
    REQUIRE(net_dev.numa_node >= -1);
  }
}

TEST_CASE("Topology Discovery with all verification levels", "[hw_topology]")
{
  topology_discovery disc_ip;
  topology_discovery disc_active;
  topology_discovery disc_exists;

  REQUIRE(disc_ip.discover(NetworkDeviceVerification::EXISTS_ACTIVE_IP));
  REQUIRE(disc_active.discover(NetworkDeviceVerification::EXISTS_ACTIVE));
  REQUIRE(disc_exists.discover(NetworkDeviceVerification::EXISTS));

  auto const& topo_ip     = disc_ip.get_topology();
  auto const& topo_active = disc_active.get_topology();
  auto const& topo_exists = disc_exists.get_topology();

  // Non-network fields must be identical across all levels.
  REQUIRE(topo_ip.hostname == topo_active.hostname);
  REQUIRE(topo_ip.hostname == topo_exists.hostname);
  REQUIRE(topo_ip.num_gpus == topo_active.num_gpus);
  REQUIRE(topo_ip.num_gpus == topo_exists.num_gpus);
  REQUIRE(topo_ip.num_numa_nodes == topo_active.num_numa_nodes);
  REQUIRE(topo_ip.num_numa_nodes == topo_exists.num_numa_nodes);

  // Monotonicity: stricter levels yield fewer or equal network devices.
  REQUIRE(topo_ip.num_network_devices <= topo_active.num_network_devices);
  REQUIRE(topo_active.num_network_devices <= topo_exists.num_network_devices);

  // Subset property: every device in a stricter set must appear in the more permissive set.
  auto contains_device = [](std::vector<network_device_info> const& haystack,
                            std::string const& name) {
    return std::find_if(haystack.begin(), haystack.end(), [&](network_device_info const& d) {
             return d.name == name;
           }) != haystack.end();
  };

  for (auto const& dev : topo_ip.network_devices) {
    REQUIRE(contains_device(topo_active.network_devices, dev.name));
  }
  for (auto const& dev : topo_active.network_devices) {
    REQUIRE(contains_device(topo_exists.network_devices, dev.name));
  }

  // Per-GPU network devices must also respect monotonicity.
  REQUIRE(topo_ip.gpus.size() == topo_exists.gpus.size());
  for (size_t i = 0; i < topo_ip.gpus.size(); ++i) {
    REQUIRE(topo_ip.gpus[i].network_devices.size() <= topo_active.gpus[i].network_devices.size());
    REQUIRE(topo_active.gpus[i].network_devices.size() <=
            topo_exists.gpus[i].network_devices.size());
  }
}

TEST_CASE("Topology Discovery default uses strictest verification", "[hw_topology]")
{
  topology_discovery disc_default;
  topology_discovery disc_explicit;

  REQUIRE(disc_default.discover());
  REQUIRE(disc_explicit.discover(NetworkDeviceVerification::EXISTS_ACTIVE_IP));

  auto const& topo_default  = disc_default.get_topology();
  auto const& topo_explicit = disc_explicit.get_topology();

  REQUIRE(topo_default.num_network_devices == topo_explicit.num_network_devices);
  REQUIRE(topo_default.network_devices.size() == topo_explicit.network_devices.size());
}

// Invariant: when the host advertises NUMA topology and GPUs are present, every discovered
// GPU must resolve to a valid NUMA node.
TEST_CASE("Topology Discovery resolves GPU NUMA node on NUMA-aware hosts", "[hw_topology]")
{
  topology_discovery discovery;
  REQUIRE(discovery.discover());

  auto const& topology = discovery.get_topology();

  if (topology.num_gpus == 0) {
    SUCCEED("Skipped: requires at least one GPU");
    return;
  }

  for (auto const& gpu : topology.gpus) {
    INFO("GPU " << gpu.id << " (" << gpu.name << " @ " << gpu.pci_bus_id << ")");
    REQUIRE(gpu.numa_node >= 0);
    REQUIRE(gpu.numa_node < topology.num_numa_nodes);
    // memory_binding is populated from numa_node whenever the latter is known.
    REQUIRE(!gpu.memory_binding.empty());
    REQUIRE(gpu.memory_binding.front() == gpu.numa_node);
  }
}

TEST_CASE("Topology Discovery reports NUMA node capacities", "[hw_topology]")
{
  topology_discovery discovery;
  REQUIRE(discovery.discover());

  auto const& topology = discovery.get_topology();

  // num_numa_nodes is derived from the discovered node list.
  REQUIRE(topology.numa_nodes.size() == static_cast<size_t>(topology.num_numa_nodes));

  if (topology.numa_nodes.empty()) {
    SUCCEED("Skipped: host does not expose NUMA topology");
    return;
  }

  size_t summed_host_capacity = 0;
  int previous_id             = -1;
  for (auto const& node : topology.numa_nodes) {
    INFO("NUMA node " << node.id);
    REQUIRE(node.id >= 0);
    // Nodes are reported sorted by id and each id appears once.
    REQUIRE(node.id > previous_id);
    previous_id = node.id;

    // /sys/devices/system/node/node<id>/meminfo always reports MemTotal and MemFree.
    REQUIRE(node.memory_capacity > 0);
    REQUIRE(node.free_memory <= node.memory_capacity);

    // A node with memory but no CPUs is device memory (GPU HBM, CXL), not host memory.
    REQUIRE(node.is_device_memory == !node.has_cpus);

    REQUIRE(topology.find_numa_node(node.id) != nullptr);
    REQUIRE(topology.get_numa_memory_capacity(node.id).value() == node.memory_capacity);
    REQUIRE(topology.get_numa_free_memory(node.id).value_or(0) == node.free_memory);
    if (!node.is_device_memory) { summed_host_capacity += node.memory_capacity; }
  }

  REQUIRE(topology.get_total_numa_memory_capacity() == summed_host_capacity);

  // Unknown node ids are reported as such rather than silently resolving to 0.
  REQUIRE_FALSE(topology.get_numa_memory_capacity(-1).has_value());
  REQUIRE_FALSE(topology.get_numa_memory_capacity(topology.numa_nodes.back().id + 1).has_value());
  REQUIRE_FALSE(topology.get_numa_free_memory(-1).has_value());
  REQUIRE(topology.find_numa_node(-1) == nullptr);
}

// Every GPU's NUMA node must be one of the discovered NUMA nodes with a known capacity,
// so host memory spaces can be sized from it.
TEST_CASE("Topology Discovery maps GPUs to NUMA nodes with known capacity", "[hw_topology]")
{
  topology_discovery discovery;
  REQUIRE(discovery.discover());

  auto const& topology = discovery.get_topology();

  if (topology.num_gpus == 0 || topology.numa_nodes.empty()) {
    SUCCEED("Skipped: requires at least one GPU and NUMA node");
    return;
  }

  for (auto const& gpu : topology.gpus) {
    INFO("GPU " << gpu.id << " on NUMA node " << gpu.numa_node);
    auto const capacity = topology.get_numa_memory_capacity(gpu.numa_node);
    REQUIRE(capacity.has_value());
    REQUIRE(*capacity > 0);
  }
}

TEST_CASE("Topology Discovery rejects out-of-range CUDA_VISIBLE_DEVICES", "[hw_topology]")
{
  ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "99999999");

  topology_discovery discovery;
  REQUIRE_THROWS_WITH(discovery.discover(), "CUDA_VISIBLE_DEVICES entry 99999999 is out of range");
}

TEST_CASE("Topology Discovery rejects overflow CUDA_VISIBLE_DEVICES", "[hw_topology]")
{
  ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "999999999999999999999999999999");

  topology_discovery discovery;
  REQUIRE_THROWS_WITH(discovery.discover(),
                      "Invalid numeric CUDA_VISIBLE_DEVICES entry: 999999999999999999999999999999");
}

TEST_CASE("Default Topology Discovery does not initialize CUDA while visibility is widened",
          "[hw_topology][cuda-context]")
{
  auto const child = fork();
  REQUIRE(child >= 0);

  if (child == 0) {
    // NVML is used only to make this test meaningful on a multi-GPU host. It
    // does not create a CUDA context or initialize the CUDA runtime.
    if (nvmlInit_v2() != NVML_SUCCESS) { _exit(77); }
    unsigned int device_count = 0;
    auto const count_status   = nvmlDeviceGetCount_v2(&device_count);
    if (count_status != NVML_SUCCESS || device_count < 2) { _exit(77); }

    // This mirrors callers that temporarily widen visibility to collect
    // physical topology, then restore the process's assigned GPU. If
    // default discover() initializes the CUDA runtime while CVD is unset, the
    // runtime permanently sees all GPUs and the final cudaGetDeviceCount() is wrong.
    setenv("CUDA_VISIBLE_DEVICES", "1", 1);
    unsetenv("CUDA_VISIBLE_DEVICES");
    topology_discovery discovery;
    bool const discovered = discovery.discover();
    setenv("CUDA_VISIBLE_DEVICES", "1", 1);
    if (!discovered) { _exit(1); }

    CUcontext context = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuCtxGetCurrent(&context) != CUDA_SUCCESS ||
        context != nullptr) {
      _exit(1);
    }

    int visible_count = 0;
    if (cudaGetDeviceCount(&visible_count) != cudaSuccess || visible_count != 1) { _exit(1); }
    _exit(0);
  }

  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  if (WIFEXITED(status) && WEXITSTATUS(status) == 77) {
    SUCCEED("Skipped: requires NVML and at least two visible GPUs");
    return;
  }
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

namespace {
/// Interfaces with at least one IPv4/IPv6 address, as reported by getifaddrs().
std::set<std::string> interfaces_with_ip()
{
  std::set<std::string> names;
  struct ifaddrs* ifa_list = nullptr;
  if (getifaddrs(&ifa_list) != 0) { return names; }
  for (struct ifaddrs* ifa = ifa_list; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) { continue; }
    int const family = ifa->ifa_addr->sa_family;
    if (family == AF_INET || family == AF_INET6) { names.insert(ifa->ifa_name); }
  }
  freeifaddrs(ifa_list);
  return names;
}

/// Interfaces under /sys/class/infiniband/<device>/device/net, sorted by name.
std::vector<std::string> sysfs_net_interfaces(std::string const& device)
{
  std::vector<std::string> names;
  std::error_code ec;
  fs::path const net_dir = fs::path("/sys/class/infiniband") / device / "device" / "net";
  for (auto const& entry : fs::directory_iterator(net_dir, ec)) {
    if (entry.is_directory()) { names.push_back(entry.path().filename().string()); }
  }
  std::sort(names.begin(), names.end());
  return names;
}

/// Contents of /sys/class/net/<iface>/mtu, or -1 if unreadable.
int sysfs_mtu(std::string const& iface)
{
  std::ifstream file(fs::path("/sys/class/net") / iface / "mtu");
  int mtu = -1;
  if (!(file >> mtu)) { return -1; }
  return mtu;
}
}  // namespace

TEST_CASE("Topology Discovery reports the MTU of each network device's interface", "[hw_topology]")
{
  auto const verification = GENERATE(NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                                     NetworkDeviceVerification::EXISTS_ACTIVE,
                                     NetworkDeviceVerification::EXISTS);
  INFO("verification level " << static_cast<int>(verification));

  topology_discovery discovery;
  REQUIRE(discovery.discover(verification));
  auto const& topology = discovery.get_topology();

  if (topology.network_devices.empty()) {
    SUCCEED("Skipped: host has no InfiniBand/RoCE devices at this verification level");
    return;
  }

  auto const with_ip = interfaces_with_ip();
  for (auto const& dev : topology.network_devices) {
    INFO("network device " << dev.name << " -> '" << dev.net_interface << "' mtu " << dev.mtu);
    auto const candidates = sysfs_net_interfaces(dev.name);

    if (candidates.empty()) {
      // InfiniBand without IPoIB: nothing to report.
      REQUIRE(dev.net_interface.empty());
      REQUIRE(dev.mtu == 0);
      continue;
    }

    // The reported interface is one of the device's own, chosen as the first by name
    // with an IP address, else the first by name.
    REQUIRE(std::find(candidates.begin(), candidates.end(), dev.net_interface) != candidates.end());
    auto const first_with_ip =
      std::find_if(candidates.begin(), candidates.end(), [&](auto const& name) {
        return with_ip.contains(name);
      });
    std::string const expected =
      first_with_ip != candidates.end() ? *first_with_ip : candidates.front();
    REQUIRE(dev.net_interface == expected);

    // An IP is what admitted the device at the strictest level, so the chosen interface
    // must carry one.
    if (verification == NetworkDeviceVerification::EXISTS_ACTIVE_IP) {
      REQUIRE(with_ip.contains(dev.net_interface));
    }

    // The MTU matches sysfs and the per-interface NIC topology.
    REQUIRE(dev.mtu > 0);
    REQUIRE(dev.mtu == sysfs_mtu(dev.net_interface));
    auto const nic = discover_nic(dev.net_interface);
    REQUIRE(nic.has_value());
    REQUIRE(nic->mtu == dev.mtu);
  }
}

TEST_CASE("Topology Discovery MTU does not depend on the verification level", "[hw_topology]")
{
  topology_discovery disc_ip;
  topology_discovery disc_exists;
  REQUIRE(disc_ip.discover(NetworkDeviceVerification::EXISTS_ACTIVE_IP));
  REQUIRE(disc_exists.discover(NetworkDeviceVerification::EXISTS));

  // A device kept at both levels reports the same interface and MTU at both.
  for (auto const& strict : disc_ip.get_topology().network_devices) {
    auto const& permissive = disc_exists.get_topology().network_devices;
    auto const it =
      std::find_if(permissive.begin(), permissive.end(), [&](network_device_info const& d) {
        return d.name == strict.name;
      });
    REQUIRE(it != permissive.end());
    INFO("network device " << strict.name);
    REQUIRE(it->net_interface == strict.net_interface);
    REQUIRE(it->mtu == strict.mtu);
  }
}

namespace {
/// A fake procfs: a temporary directory holding a @c proc/ subtree.
class fake_procfs {
 public:
  fake_procfs()
  {
    static std::atomic<int> counter{0};
    _root = fs::temp_directory_path() / ("cucascade_ktls_" + std::to_string(::getpid()) + "_" +
                                         std::to_string(counter.fetch_add(1)));
    fs::remove_all(_root);
    fs::create_directories(_root / "proc");
  }
  ~fake_procfs()
  {
    std::error_code ec;
    fs::remove_all(_root, ec);
  }
  fake_procfs(fake_procfs const&)            = delete;
  fake_procfs& operator=(fake_procfs const&) = delete;

  [[nodiscard]] discovery_roots roots() const { return {_root, _root}; }

  void modules(std::string const& content) const { write("proc/modules", content); }
  void ulps(std::string const& content) const
  {
    write("proc/sys/net/ipv4/tcp_available_ulp", content);
  }

 private:
  void write(std::string const& rel, std::string const& content) const
  {
    auto const path = _root / rel;
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
  }

  fs::path _root;
};

constexpr char const* other_modules =
  "nvidia_uvm 2076672 4 - Live 0x0000000000000000 (POE)\n"
  "ena 155648 0 - Live 0x0000000000000000\n";
constexpr char const* tls_live = "tls 167936 0 - Live 0x0000000000000000\n";
}  // namespace

TEST_CASE("kTLS: a live tls module with the ULP registered is loaded, not built in",
          "[hw_topology][ktls]")
{
  fake_procfs host;
  host.modules(std::string{other_modules} + tls_live);
  host.ulps("espintcp mptcp tls\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: a registered ULP without a module row is built in", "[hw_topology][ktls]")
{
  fake_procfs host;
  host.modules(other_modules);
  host.ulps("tls\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE(ktls.loaded);
  REQUIRE(ktls.built_in);
}

TEST_CASE("kTLS: neither a module row nor a registered ULP is not loaded", "[hw_topology][ktls]")
{
  fake_procfs host;
  host.modules(other_modules);
  host.ulps("espintcp mptcp\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE_FALSE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: a module that is still loading or unloading is not loaded", "[hw_topology][ktls]")
{
  auto const state = GENERATE(std::string{"Loading"}, std::string{"Unloading"});
  INFO("module state " << state);

  fake_procfs host;
  host.modules(std::string{other_modules} + "tls 167936 0 - " + state + " 0x0000000000000000\n");
  host.ulps("mptcp\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE_FALSE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: only exact names count as the tls module and ULP", "[hw_topology][ktls]")
{
  fake_procfs host;
  host.modules(std::string{other_modules} + "tls_foo 4096 0 - Live 0x0\nktls 4096 0 - Live 0x0\n");
  host.ulps("tlsx ktls mptcp\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE_FALSE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: the ULP list is authoritative over the module row", "[hw_topology][ktls]")
{
  // A Live row with no registered ULP (e.g. a racing unload): not usable.
  fake_procfs host;
  host.modules(tls_live);
  host.ulps("mptcp\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE_FALSE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: an unreadable ULP list falls back to the module row", "[hw_topology][ktls]")
{
  SECTION("live module")
  {
    fake_procfs host;
    host.modules(std::string{other_modules} + tls_live);
    auto const ktls = discover_kernel_tls(host.roots());
    REQUIRE(ktls.loaded);
    REQUIRE_FALSE(ktls.built_in);
  }
  SECTION("no module row")
  {
    fake_procfs host;
    host.modules(other_modules);
    auto const ktls = discover_kernel_tls(host.roots());
    REQUIRE_FALSE(ktls.loaded);
    REQUIRE_FALSE(ktls.built_in);
  }
}

TEST_CASE("kTLS: an unreadable module list cannot report built in", "[hw_topology][ktls]")
{
  fake_procfs host;
  host.ulps("espintcp mptcp tls\n");

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: an empty procfs reports nothing", "[hw_topology][ktls]")
{
  fake_procfs host;

  auto const ktls = discover_kernel_tls(host.roots());
  REQUIRE_FALSE(ktls.loaded);
  REQUIRE_FALSE(ktls.built_in);
}

TEST_CASE("kTLS: discover() reports this host's kTLS state", "[hw_topology][ktls]")
{
  topology_discovery discovery;
  REQUIRE(discovery.discover());
  auto const& ktls = discovery.get_topology().kernel_tls;

  auto const direct = discover_kernel_tls();
  REQUIRE(ktls.loaded == direct.loaded);
  REQUIRE(ktls.built_in == direct.built_in);
  REQUIRE((!ktls.built_in || ktls.loaded));

  // The tls module creates /proc/net/tls_stat in its init, whether loaded or built in.
  std::error_code ec;
  bool const has_tls_stat = fs::exists("/proc/net/tls_stat", ec);
  INFO("loaded=" << ktls.loaded << " built_in=" << ktls.built_in
                 << " /proc/net/tls_stat=" << has_tls_stat);
  REQUIRE(ktls.loaded == has_tls_stat);
}
