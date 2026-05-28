/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/memory/topology_discovery.hpp>

#include <dlfcn.h>
#include <ifaddrs.h>
#include <nvml.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace cucascade::memory {

namespace {
/**
 * @brief Network device with topology information.
 */
struct NetworkDeviceWithTopology {
  std::string name;
  int numa_node;
  std::string pci_bus_id;
};

/**
 * @brief Emit a standardized warning for a failed NVML call.
 *
 * Formats the message as "Warning: <context>: <nvmlErrorString(result)>" and writes
 * it to stderr. Callers retain control of flow (skip, continue, use defaults) — this
 * helper only standardizes message formatting so every NVML failure surfaces the same
 * way and always includes the decoded NVML error string.
 *
 * @param result NVML return code that did not equal NVML_SUCCESS.
 * @param context Description of the operation that failed (e.g. "Failed to get
 * device count" or "Failed to get handle for GPU 0").
 */
void report_nvml_error(nvmlReturn_t result, std::string const& context)
{
  std::cerr << "Warning: " << context << ": " << nvmlErrorString(result) << std::endl;
}

/**
 * @brief Read a file and return its content.
 *
 * Attempts to open and read the file at @p path. If the file cannot be opened or
 * read, returns an empty string. On success, returns the full contents with a
 * single trailing '\n' removed if present.
 *
 * @param path File to read.
 * @return File content on success; empty string on failure.
 */
std::string read_file_content(std::string const& path)
{
  std::ifstream file(path);
  if (!file.is_open()) { return ""; }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();
  // Trim trailing newline
  if (!content.empty() && content.back() == '\n') { content.pop_back(); }
  return content;
}

/**
 * @brief Parse a CPU list string into a vector of core ids.
 *
 * Accepts formats like "0-31,128-159" or comma-separated single cores. If @p cpulist
 * is empty, returns an empty vector.
 *
 * @param cpulist CPU list string (e.g., "0-31,128-159").
 * @return Vector of CPU core ids; empty if @p cpulist is empty.
 * @throw std::invalid_argument or std::out_of_range if tokens are malformed and cannot be
 * parsed with std::stoi.
 */
std::vector<int> parse_cpu_list(std::string const& cpulist)
{
  std::vector<int> cores;
  if (cpulist.empty()) { return cores; }

  std::istringstream iss(cpulist);
  std::string token;
  while (std::getline(iss, token, ',')) {
    size_t dash_pos = token.find('-');
    if (dash_pos != std::string::npos) {
      // Range, e.g., "0-31"
      int start = std::stoi(token.substr(0, dash_pos));
      int end   = std::stoi(token.substr(dash_pos + 1));
      for (int i = start; i <= end; ++i) {
        cores.push_back(i);
      }
    } else {
      // Single core, e.g., "5"
      cores.push_back(std::stoi(token));
    }
  }
  return cores;
}

/**
 * @brief Normalize PCI bus ID to standard format.
 *
 * Converts the domain to 4 hex digits and lowercases the entire string. If the input
 * does not contain a colon (unexpected format), the input is returned unchanged.
 *
 * @param pci_bus_id PCI bus ID to normalize.
 * @return Normalized PCI bus ID in format (0000:06:00.0); unchanged on unrecognized
 * input.
 */
std::string normalize_pci_bus_id(std::string const& pci_bus_id)
{
  // NVML may return format like "00000000:0A:00.0" but /sys uses "0000:0a:00.0"
  // Find the first colon and take the last 4 hex digits before it as domain
  size_t colon_pos = pci_bus_id.find(':');
  if (colon_pos == std::string::npos) { return pci_bus_id; }
  std::string domain = pci_bus_id.substr(0, colon_pos);
  if (domain.length() > 4) { domain = domain.substr(domain.length() - 4); }

  // Convert to lowercase
  std::string normalized_id = domain + pci_bus_id.substr(colon_pos);
  std::ranges::transform(normalized_id, normalized_id.begin(), ::tolower);

  return normalized_id;
}

std::string trim_copy(std::string const& input)
{
  size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
    ++start;
  }
  size_t end = input.size();
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }
  return input.substr(start, end - start);
}

std::vector<std::string> split_csv(std::string const& input)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(input);
  while (std::getline(iss, token, ',')) {
    tokens.push_back(trim_copy(token));
  }
  return tokens;
}

bool is_numeric_token(std::string const& token)
{
  return !token.empty() &&
         std::ranges::all_of(token, [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::vector<size_t> resolve_visible_gpu_indices(
  std::vector<gpu_topology_info> const& nvml_gpus,
  std::unordered_map<std::string, size_t> const& index_by_pci,
  std::unordered_map<std::string, size_t> const& index_by_uuid)
{
  std::vector<size_t> indices;
  std::unordered_set<size_t> seen;

  char const* env_value = std::getenv("CUDA_VISIBLE_DEVICES");
  if (!env_value) {
    indices.reserve(nvml_gpus.size());
    for (size_t i = 0; i < nvml_gpus.size(); ++i) {
      indices.push_back(i);
    }
    return indices;
  }

  std::string env_str(env_value);
  auto tokens = split_csv(env_str);
  for (auto const& token : tokens) {
    if (token.empty()) { continue; }

    bool matched = false;
    if (is_numeric_token(token)) {
      size_t idx = 0;
      try {
        idx = static_cast<size_t>(std::stoul(token));
      } catch (std::exception const& e) {
        throw std::invalid_argument("Invalid numeric CUDA_VISIBLE_DEVICES entry: " + token);
      }
      if (idx < nvml_gpus.size()) {
        if (seen.insert(idx).second) { indices.push_back(idx); }
        matched = true;
      } else {
        throw std::invalid_argument("CUDA_VISIBLE_DEVICES entry " + token + " is out of range");
      }
    } else if (token.starts_with("GPU-") || token.starts_with("MIG-")) {
      // Direct UUID lookup first. With MIG enumeration each MIG instance is
      // registered under its own MIG-* UUID, so this matches both forms.
      auto uuid_it = index_by_uuid.find(token);
      if (uuid_it != index_by_uuid.end()) {
        if (seen.insert(uuid_it->second).second) { indices.push_back(uuid_it->second); }
        matched = true;
      }

      // Fallback: resolve the UUID via NVML to a parent PCI bus and match by PCI.
      // Reachable only when MIG instances were not enumerated (e.g. NVML denied
      // access to MIG mode); a MIG UUID then maps to its parent physical GPU.
      if (!matched) {
        nvmlDevice_t handle;
        if (nvmlDeviceGetHandleByUUID(token.c_str(), &handle) == NVML_SUCCESS) {
          unsigned int is_mig = 0;
          if (nvmlDeviceIsMigDeviceHandle(handle, &is_mig) == NVML_SUCCESS && is_mig) {
            nvmlDevice_t parent_handle;
            if (nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle, &parent_handle) ==
                NVML_SUCCESS) {
              handle = parent_handle;
            }
          }

          nvmlPciInfo_t pci_info;
          if (nvmlDeviceGetPciInfo_v3(handle, &pci_info) == NVML_SUCCESS) {
            std::string normalized = normalize_pci_bus_id(pci_info.busId);
            auto it                = index_by_pci.find(normalized);
            if (it != index_by_pci.end()) {
              if (seen.insert(it->second).second) { indices.push_back(it->second); }
              matched = true;
            }
          }
        }
      }
    }

    if (!matched) {
      std::cerr << "Warning: CUDA_VISIBLE_DEVICES entry '" << token
                << "' does not map to an NVML device" << std::endl;
    }
  }

  return indices;
}

/**
 * @brief Get the host NUMA node id with the best memory affinity to a GPU.
 *
 * Queries NVML's `nvmlDeviceGetMemoryAffinity` with `NVML_AFFINITY_SCOPE_NODE` and
 * returns the lowest-numbered NUMA node in the resulting bitmask. Same source as
 * `nvidia-smi topo -m`.
 *
 * @param device NVML device handle.
 * @return NUMA node id on success; -1 if NVML reports no affinity (empty bitmask
 * or query failure).
 *
 * @note `nodeSetSize = 1` (one `unsigned long` = 64 NUMA bits) is sufficient for
 * any realistic system.
 */
int get_numa_node_from_nvml(nvmlDevice_t device)
{
  unsigned long nodeset = 0;
  if (nvmlDeviceGetMemoryAffinity(device, 1, &nodeset, NVML_AFFINITY_SCOPE_NODE) == NVML_SUCCESS &&
      nodeset != 0) {
    return std::countr_zero(nodeset);
  }
  return -1;
}

/**
 * @brief Get CPU affinity list from /sys for a PCI device.
 *
 * Reads /sys/bus/pci/devices/<pci>/local_cpulist and returns the file content as-is
 * (with any trailing newline trimmed). If the file is missing or unreadable, returns
 * an empty string.
 *
 * @param pci_bus_id PCI bus ID of the device.
 * @return CPU affinity list string; empty on failure.
 */
std::string get_cpu_affinity_from_sys(std::string const& pci_bus_id)
{
  std::string normalized_id = normalize_pci_bus_id(pci_bus_id);
  std::string path          = "/sys/bus/pci/devices/" + normalized_id + "/local_cpulist";
  return read_file_content(path);
}

/**
 * @brief Get PCI bus ID from a device directory in /sys.
 *
 * Resolves <device_path>/device symlink and returns its basename (e.g., "0000:06:00.0").
 * Returns an empty string if the symlink does not exist or cannot be resolved.
 *
 * @param device_path Path to the device in /sys.
 * @return PCI bus ID of the device; empty string if not available.
 */
std::string get_pci_bus_id_from_device(std::string const& device_path)
{
  fs::path device_link = fs::path(device_path) / "device";
  if (!fs::exists(device_link)) { return ""; }

  try {
    fs::path real_path = fs::canonical(device_link);
    return real_path.filename().string();
  } catch (...) {
    return "";
  }
}

/**
 * @brief Parse PCI bus number from a PCI ID string.
 *
 * Expects format domain:bus:device.function (e.g., "0000:06:00.0"). If parsing fails
 * or the expected separators are missing, returns -1.
 *
 * @param pci_id PCI ID string.
 * @return PCI bus number (integer) on success; -1 on failure.
 */
int get_pci_bus_number(std::string const& pci_id)
{
  size_t first_colon = pci_id.find(':');
  if (first_colon == std::string::npos) { return -1; }

  size_t second_colon = pci_id.find(':', first_colon + 1);
  if (second_colon == std::string::npos) { return -1; }

  std::string bus_str = pci_id.substr(first_colon + 1, second_colon - first_colon - 1);
  try {
    return std::stoi(bus_str, nullptr, 16);
  } catch (...) {
    return -1;
  }
}

/**
 * @brief Get PCIe path type between two devices by analyzing /sys topology.
 *
 * Uses NUMA node comparison and PCI bus number proximity as a heuristic. If either
 * device's bus cannot be parsed, falls back to PHB. If NUMA nodes differ (and are
 * known), returns SYS.
 *
 * @param gpu_pci_id PCI bus ID of the GPU device.
 * @param nic_pci_id PCI bus ID of the NIC device.
 * @return Path type (PIX, PXB, PHB, NODE, or SYS); PHB when indeterminate.
 */
PciePathType get_pcie_path_type(std::string const& gpu_pci_id, std::string const& nic_pci_id)
{
  std::string gpu_norm = normalize_pci_bus_id(gpu_pci_id);
  std::string nic_norm = normalize_pci_bus_id(nic_pci_id);

  // Read NUMA nodes
  int gpu_numa = -1, nic_numa = -1;
  std::string gpu_numa_str = read_file_content("/sys/bus/pci/devices/" + gpu_norm + "/numa_node");
  std::string nic_numa_str = read_file_content("/sys/bus/pci/devices/" + nic_norm + "/numa_node");

  if (!gpu_numa_str.empty()) { gpu_numa = std::stoi(gpu_numa_str); }
  if (!nic_numa_str.empty()) { nic_numa = std::stoi(nic_numa_str); }

  // If different NUMA nodes, it's a SYS connection
  if (gpu_numa != nic_numa && gpu_numa >= 0 && nic_numa >= 0) { return PciePathType::SYS; }

  // Use PCI bus number proximity as a heuristic for connection quality
  // Devices on nearby PCI buses are typically on the same PCIe root complex
  int gpu_bus = get_pci_bus_number(gpu_pci_id);
  int nic_bus = get_pci_bus_number(nic_pci_id);

  if (gpu_bus < 0 || nic_bus < 0) {
    // Can't determine, assume PHB
    return PciePathType::PHB;
  }

  int bus_distance = std::abs(gpu_bus - nic_bus);

  // Heuristic based on PCI bus proximity:
  // - Very close buses (distance <= 2): likely PIX (single bridge)
  // - Moderate distance (3-10): likely PHB (host bridge)
  // - Large distance (>10): likely NODE or worse

  if (bus_distance <= 2) {
    return PciePathType::PIX;
  } else if (bus_distance <= 10) {
    return PciePathType::PHB;
  } else {
    return PciePathType::NODE;
  }
}

/**
 * @brief Check whether an InfiniBand/RoCE device has at least one active port.
 *
 * Iterates over /sys/class/infiniband/<device>/ports/<N>/state for every port.
 * A port is considered active when its state string contains "ACTIVE".
 *
 * @param device_path Sysfs path to the InfiniBand device directory.
 * @return true if at least one port is in the ACTIVE state.
 */
bool has_active_port(std::string const& device_path)
{
  fs::path ports_dir = fs::path(device_path) / "ports";
  if (!fs::exists(ports_dir)) { return false; }

  try {
    for (auto const& port_entry : fs::directory_iterator(ports_dir)) {
      if (!port_entry.is_directory()) { continue; }

      std::string state = read_file_content((port_entry.path() / "state").string());
      // The state file format is "<number>: <STATE_NAME>" (e.g. "4: ACTIVE").
      if (state.find("ACTIVE") != std::string::npos) { return true; }
    }
  } catch (...) {
  }
  return false;
}

/**
 * @brief Check whether an InfiniBand device has an accessible userspace verbs device node.
 *
 * Looks up the uverbs device name from
 * /sys/class/infiniband/<device>/device/infiniband_verbs/ and then checks whether
 * the corresponding /dev/infiniband/<uverbs> character device exists.  In containerized
 * environments the sysfs entries may be mounted from the host while the /dev nodes
 * are not passed through, leaving libibverbs unable to open the device.
 *
 * @param device_path Sysfs path to the InfiniBand device directory.
 * @return true if the uverbs device node exists under /dev/infiniband/.
 */
bool has_uverbs_device(std::string const& device_path)
{
  fs::path verbs_dir = fs::path(device_path) / "device" / "infiniband_verbs";
  if (!fs::exists(verbs_dir)) { return false; }

  try {
    for (auto const& entry : fs::directory_iterator(verbs_dir)) {
      if (!entry.is_directory()) { continue; }
      fs::path dev_node = fs::path("/dev/infiniband") / entry.path().filename();
      if (fs::exists(dev_node)) { return true; }
    }
  } catch (...) {
  }
  return false;
}

/**
 * @brief Check whether a network interface has at least one IP address (v4 or v6).
 *
 * Uses getifaddrs() to query all interface addresses and returns true as soon as
 * an AF_INET or AF_INET6 entry matching @p iface_name is found.
 *
 * @param iface_name Name of the network interface (e.g. "ib0", "ibp26s0").
 * @return true if the interface has at least one IP address assigned.
 */
bool interface_has_ip(std::string const& iface_name)
{
  struct ifaddrs* ifa_list = nullptr;
  if (getifaddrs(&ifa_list) != 0) { return false; }

  bool found = false;
  for (struct ifaddrs* ifa = ifa_list; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) { continue; }
    int family = ifa->ifa_addr->sa_family;
    if ((family == AF_INET || family == AF_INET6) && iface_name == ifa->ifa_name) {
      found = true;
      break;
    }
  }

  freeifaddrs(ifa_list);
  return found;
}

/**
 * @brief Check whether an InfiniBand device's associated net interface has an IP address.
 *
 * Looks up the net interface name from /sys/class/infiniband/<dev>/device/net/ and
 * then checks for a configured IP address via getifaddrs().  Devices without IPoIB
 * or RoCE networking configured (no IP) are unusable for IP-based transports.
 *
 * @param device_path Sysfs path to the InfiniBand device directory.
 * @return true if the device has a net interface with at least one IP address.
 */
bool has_net_interface_with_ip(std::string const& device_path)
{
  fs::path net_dir = fs::path(device_path) / "device" / "net";
  if (!fs::exists(net_dir)) { return false; }

  try {
    for (auto const& entry : fs::directory_iterator(net_dir)) {
      if (!entry.is_directory()) { continue; }
      if (interface_has_ip(entry.path().filename().string())) { return true; }
    }
  } catch (...) {
  }
  return false;
}

/**
 * @brief Discover network devices (InfiniBand/RoCE).
 *
 * Scans /sys/class/infiniband and collects device name, NUMA node, and PCI bus ID.
 * Depending on @p verification, additional checks filter out unusable devices:
 *   - EXISTS: device must be present in /sys/class/infiniband.
 *   - EXISTS_ACTIVE: additionally, at least one port must be in the ACTIVE state
 *     and the uverbs device node must exist under /dev/infiniband/.
 *   - EXISTS_ACTIVE_IP: additionally, the net interface must have an IP address.
 * If the directory does not exist or an error occurs during iteration, returns an
 * empty vector (a warning is logged for iteration errors).
 *
 * @param verification Verification level for network devices.
 * @return Vector of discovered network devices; empty if none or on failure.
 */
std::vector<NetworkDeviceWithTopology> discover_network_devices_with_topology(
  NetworkDeviceVerification verification)
{
  std::vector<NetworkDeviceWithTopology> devices;
  std::string ib_path = "/sys/class/infiniband";

  if (!fs::exists(ib_path)) { return devices; }

  try {
    for (auto const& entry : fs::directory_iterator(ib_path)) {
      if (!entry.is_directory()) { continue; }

      if (verification <= NetworkDeviceVerification::EXISTS_ACTIVE) {
        if (!has_active_port(entry.path().string())) { continue; }
        if (!has_uverbs_device(entry.path().string())) { continue; }
      }
      if (verification <= NetworkDeviceVerification::EXISTS_ACTIVE_IP) {
        if (!has_net_interface_with_ip(entry.path().string())) { continue; }
      }

      NetworkDeviceWithTopology dev;
      dev.name = entry.path().filename().string();

      // Get device's NUMA node and PCI bus ID
      std::string numa_path = entry.path().string() + "/device/numa_node";
      std::string numa_str  = read_file_content(numa_path);
      dev.numa_node         = numa_str.empty() ? -1 : std::stoi(numa_str);
      dev.pci_bus_id        = get_pci_bus_id_from_device(entry.path().string());

      devices.push_back(dev);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering network devices: " << e.what() << std::endl;
  }

  return devices;
}

std::vector<storage_device_info> discover_storage_devices_with_topology()
{
  std::vector<storage_device_info> devices;
  std::string nvme_path = "/sys/class/nvme";

  if (!fs::exists(nvme_path)) { return devices; }

  try {
    for (auto const& entry : fs::directory_iterator(nvme_path)) {
      if (!entry.is_directory()) { continue; }

      storage_device_info dev;
      dev.name = entry.path().filename().string();
      dev.type = StorageDriveType::NVME;

      // Get device's NUMA node and PCI bus ID
      std::string numa_path = entry.path().string() + "/device/numa_node";
      std::string numa_str  = read_file_content(numa_path);
      dev.numa_node         = numa_str.empty() ? -1 : std::stoi(numa_str);
      dev.pci_bus_id        = get_pci_bus_id_from_device(entry.path().string());

      devices.push_back(dev);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering NVMe devices: " << e.what() << std::endl;
  }

  return devices;
}

/**
 * @brief Map network devices to a GPU based on PCIe topology and NUMA proximity.
 *
 * Prefers NICs with the best (lowest) PCIe path type to the GPU. If no PCI topology
 * information is available, falls back to matching by NUMA node. If still ambiguous,
 * returns all devices. May return an empty vector when @p network_devices is empty
 * or none have PCI info.
 *
 * @param gpu_numa_node NUMA node to query.
 * @param network_devices Network devices on the system.
 * @return Vector of device names selected for the GPU; possibly empty.
 */
std::vector<std::string> map_network_devices_to_gpu(
  std::string const& gpu_pci_id,
  int gpu_numa_node,
  std::vector<network_device_info> const& network_devices)
{
  std::vector<std::string> mapped_devices;

  // Structure to hold NIC with its topology path type
  struct NicWithPath {
    std::string name;
    PciePathType path_type;
  };

  std::vector<NicWithPath> nics_with_paths;

  // Query topology distance for each NIC
  for (auto const& dev : network_devices) {
    if (dev.pci_bus_id.empty()) {
      continue;  // Skip devices without PCI info
    }

    NicWithPath nic;
    nic.name      = dev.name;
    nic.path_type = get_pcie_path_type(gpu_pci_id, dev.pci_bus_id);

    nics_with_paths.push_back(nic);
  }

  // Find the best (lowest) path type
  if (nics_with_paths.empty()) { return mapped_devices; }

  PciePathType best_path_type = PciePathType::SYS;
  for (auto const& nic : nics_with_paths) {
    if (nic.path_type < best_path_type) { best_path_type = nic.path_type; }
  }

  // Return all NICs with the best path type
  for (auto const& nic : nics_with_paths) {
    if (nic.path_type == best_path_type) { mapped_devices.push_back(nic.name); }
  }

  // If no devices found, fall back to NUMA-based mapping
  if (mapped_devices.empty()) {
    for (auto const& dev : network_devices) {
      if (dev.numa_node == gpu_numa_node) { mapped_devices.push_back(dev.name); }
    }
  }

  // Last resort: return all devices
  if (mapped_devices.empty() && !network_devices.empty()) {
    for (auto const& dev : network_devices) {
      mapped_devices.push_back(dev.name);
    }
  }

  return mapped_devices;
}

/**
 * @brief Get system hostname.
 *
 * Returns the system hostname on success; returns an empty string if the
 * hostname cannot be determined.
 *
 * @return Hostname string, or an empty string if unavailable.
 */
std::string get_hostname()
{
  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) == 0) { return std::string(hostname.data()); }
  return "";
}

/**
 * @brief Count NUMA nodes on the system.
 *
 * Counts subdirectories named "node*" under /sys/devices/system/node. Returns 0 if
 * the directory does not exist or cannot be iterated.
 *
 * @return Number of NUMA nodes; 0 if unavailable.
 */
int count_numa_nodes()
{
  std::string numa_path = "/sys/devices/system/node";
  int count             = 0;

  if (!fs::exists(numa_path)) { return 0; }

  try {
    for (auto const& entry : fs::directory_iterator(numa_path)) {
      std::string name = entry.path().filename().string();
      if (name.starts_with("node")) {  // starts with "node"
        count++;
      }
    }
  } catch (...) {
    return 0;
  }

  return count;
}

}  // namespace

bool topology_discovery::discover(NetworkDeviceVerification net_verification)
{
  system_topology_info topology;
  // NVML is initialized exactly once per process via this static-local. Calling
  // nvmlInit_v2 + nvmlShutdown in sequence (which discover() used to do on every
  // call) SEGVs on NVIDIA driver 595.58.03 — the second nvmlInit_v2 after a
  // shutdown lands on a stale internal function pointer in libnvidia-ml.
  // The driver releases NVML resources at process exit via its own atexit hook,
  // so we never need to call nvmlShutdown explicitly.
  static const nvmlReturn_t init_result = nvmlInit_v2();
  nvmlReturn_t result                   = init_result;
  if (result != NVML_SUCCESS) {
    report_nvml_error(result, "Failed to initialize NVML");
    // Continue anyway to report system info even without GPUs
  }

  // Get GPU count
  unsigned int device_count = 0;
  bool nvml_available       = false;
  if (result == NVML_SUCCESS) {
    result = nvmlDeviceGetCount_v2(&device_count);
    if (result != NVML_SUCCESS) {
      report_nvml_error(result, "Failed to get device count");
      device_count = 0;
    } else {
      nvml_available = true;
    }
  }

  // Discover network devices
  std::vector<NetworkDeviceWithTopology> network_devices_with_topology =
    discover_network_devices_with_topology(net_verification);

  // Get system information
  topology.hostname            = get_hostname();
  topology.num_numa_nodes      = count_numa_nodes();
  topology.num_gpus            = device_count;
  topology.num_network_devices = static_cast<int>(network_devices_with_topology.size());

  // Convert network devices to public format
  topology.network_devices.clear();
  for (auto const& dev : network_devices_with_topology) {
    network_device_info info;
    info.name       = dev.name;
    info.numa_node  = dev.numa_node;
    info.pci_bus_id = dev.pci_bus_id;
    topology.network_devices.push_back(info);
  }

  topology.storage_devices = discover_storage_devices_with_topology();

  // Collect GPU information
  topology.gpus.clear();

  std::vector<gpu_topology_info> nvml_gpus;
  std::unordered_map<std::string, size_t> nvml_index_by_pci;
  std::unordered_map<std::string, size_t> nvml_index_by_uuid;
  if (nvml_available) {
    // Emit one gpu_topology_info from an NVML device handle (either a physical GPU
    // or a MIG instance). Topology fields (PCI, NUMA, CPU affinity, NICs) come from
    // the physical parent — MIG instances share their parent's physical hardware.
    auto emit_gpu = [&](nvmlDevice_t handle,
                        std::string const& parent_pci,
                        int parent_numa,
                        std::string const& parent_cpu_affinity,
                        std::vector<int> const& parent_cpu_cores,
                        std::vector<std::string> const& parent_nics) {
      gpu_topology_info gpu;

      std::array<char, NVML_DEVICE_NAME_BUFFER_SIZE> name{};
      nvmlReturn_t r = nvmlDeviceGetName(handle, name.data(), NVML_DEVICE_NAME_BUFFER_SIZE);
      gpu.name       = (r == NVML_SUCCESS) ? std::string(name.data()) : "Unknown";

      std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE> uuid{};
      r        = nvmlDeviceGetUUID(handle, uuid.data(), NVML_DEVICE_UUID_BUFFER_SIZE);
      gpu.uuid = (r == NVML_SUCCESS) ? std::string(uuid.data()) : "Unknown";

      gpu.pci_bus_id        = parent_pci;
      gpu.numa_node         = parent_numa;
      gpu.cpu_affinity_list = parent_cpu_affinity;
      gpu.cpu_cores         = parent_cpu_cores;
      if (parent_numa >= 0) { gpu.memory_binding.push_back(parent_numa); }
      gpu.network_devices = parent_nics;

      nvml_gpus.push_back(std::move(gpu));
      if (!nvml_gpus.back().uuid.empty()) {
        nvml_index_by_uuid.emplace(nvml_gpus.back().uuid, nvml_gpus.size() - 1);
      }
    };

    for (unsigned int i = 0; i < device_count; ++i) {
      nvmlDevice_t device;
      result = nvmlDeviceGetHandleByIndex_v2(i, &device);
      if (result != NVML_SUCCESS) {
        report_nvml_error(result, "Failed to get handle for GPU " + std::to_string(i));
        continue;
      }

      // Resolve the physical device's topology once; reused for every MIG instance.
      nvmlPciInfo_t pci_info;
      result = nvmlDeviceGetPciInfo_v3(device, &pci_info);
      if (result != NVML_SUCCESS) {
        report_nvml_error(result, "Failed to get PCI info for GPU " + std::to_string(i));
        continue;
      }
      std::string parent_pci      = std::string(pci_info.busId);
      int parent_numa             = get_numa_node_from_nvml(device);
      std::string parent_cpu_aff  = get_cpu_affinity_from_sys(parent_pci);
      std::vector<int> parent_cpu = parse_cpu_list(parent_cpu_aff);
      std::vector<std::string> parent_nics =
        map_network_devices_to_gpu(parent_pci, parent_numa, topology.network_devices);

      unsigned int mig_current = NVML_DEVICE_MIG_DISABLE;
      unsigned int mig_pending = NVML_DEVICE_MIG_DISABLE;
      nvmlReturn_t mig_rc      = nvmlDeviceGetMigMode(device, &mig_current, &mig_pending);
      // NVML_ERROR_NOT_SUPPORTED on non-Ampere/non-MIG-capable GPUs — treat as disabled.
      bool mig_enabled = (mig_rc == NVML_SUCCESS && mig_current == NVML_DEVICE_MIG_ENABLE);

      if (!mig_enabled) {
        emit_gpu(device, parent_pci, parent_numa, parent_cpu_aff, parent_cpu, parent_nics);
        // PCI-based index entries are unambiguous only for physical GPUs.
        nvml_index_by_pci.emplace(normalize_pci_bus_id(parent_pci), nvml_gpus.size() - 1);
        continue;
      }

      unsigned int max_mig = 0;
      nvmlReturn_t mc_rc   = nvmlDeviceGetMaxMigDeviceCount(device, &max_mig);
      if (mc_rc != NVML_SUCCESS) {
        report_nvml_error(
          mc_rc,
          "MIG enabled on GPU " + std::to_string(i) + " but failed to query MIG device count");
        max_mig = 0;
      }

      unsigned int emitted = 0;
      for (unsigned int mig_idx = 0; mig_idx < max_mig; ++mig_idx) {
        nvmlDevice_t mig_device;
        nvmlReturn_t r = nvmlDeviceGetMigDeviceHandleByIndex(device, mig_idx, &mig_device);
        if (r == NVML_ERROR_NOT_FOUND) { continue; }
        if (r != NVML_SUCCESS) {
          report_nvml_error(r,
                            "Failed to get MIG handle for GPU " + std::to_string(i) + " slot " +
                              std::to_string(mig_idx));
          continue;
        }
        emit_gpu(mig_device, parent_pci, parent_numa, parent_cpu_aff, parent_cpu, parent_nics);
        ++emitted;
      }

      if (emitted == 0) {
        std::cerr << "Warning: MIG enabled on GPU " << i << " but no MIG instances were enumerated"
                  << std::endl;
      }
    }
  }

  auto visible_indices =
    resolve_visible_gpu_indices(nvml_gpus, nvml_index_by_pci, nvml_index_by_uuid);
  topology.num_gpus = static_cast<unsigned int>(visible_indices.size());
  for (size_t visible_idx = 0; visible_idx < visible_indices.size(); ++visible_idx) {
    size_t nvml_idx = visible_indices[visible_idx];
    if (nvml_idx >= nvml_gpus.size()) { continue; }
    auto gpu = nvml_gpus[nvml_idx];
    gpu.id   = static_cast<unsigned int>(visible_idx);
    topology.gpus.push_back(std::move(gpu));
  }
  std::cerr << "num_gpus: " << topology.num_gpus << " vs gpu count: " << topology.gpus.size()
            << " device count: " << device_count << std::endl;

  // Do not call nvmlShutdown here — NVML is initialized once per process via
  // the static-local in this function. See the comment at the top of discover().

  _topology = std::move(topology);
  return true;
}

}  // namespace cucascade::memory
