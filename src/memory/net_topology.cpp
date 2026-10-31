/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/memory/net_topology.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace cucascade::memory {

namespace {

namespace fs = std::filesystem;

constexpr unsigned rtf_up     = 0x0001;
constexpr unsigned rtf_reject = 0x0200;

fs::path sys_path(discovery_roots const& roots, std::string_view rel)
{
  return roots.sysfs_root / "sys" / fs::path(rel);
}

fs::path proc_path(discovery_roots const& roots, std::string_view rel)
{
  return roots.procfs_root / "proc" / fs::path(rel);
}

std::string_view trim(std::string_view s)
{
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) {
    s.remove_prefix(1);
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.remove_suffix(1);
  }
  return s;
}

/// The first line of @p path, trimmed; @c nullopt when the file is unreadable.
std::optional<std::string> read_line(fs::path const& path)
{
  std::ifstream in(path);
  if (!in) { return std::nullopt; }
  std::string line;
  std::getline(in, line);
  if (in.bad()) { return std::nullopt; }
  return std::string{trim(line)};
}

template <typename T>
std::optional<T> parse_number(std::string_view s, int base = 10)
{
  s = trim(s);
  if (s.empty()) { return std::nullopt; }
  T value{};
  auto const* first = s.data();
  auto const* last  = s.data() + s.size();
  auto [ptr, ec]    = std::from_chars(first, last, value, base);
  if (ec != std::errc{} || ptr != last) { return std::nullopt; }
  return value;
}

std::optional<int> read_int(fs::path const& path)
{
  auto line = read_line(path);
  if (!line) { return std::nullopt; }
  return parse_number<int>(*line);
}

std::vector<int> read_cpu_list(fs::path const& path)
{
  auto line = read_line(path);
  if (!line) { return {}; }
  return parse_cpu_list(*line);
}

std::vector<std::string_view> split_ws(std::string_view s)
{
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0) {
      ++i;
    }
    auto const start = i;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) == 0) {
      ++i;
    }
    if (i > start) { out.push_back(s.substr(start, i - start)); }
  }
  return out;
}

/// @c 0000:00:05.0 / @c 0003:02:00.1 (domain:bus:device.function, hex).
bool looks_like_pci_address(std::string_view s)
{
  if (s.size() != 12) { return false; }
  for (std::size_t i = 0; i < s.size(); ++i) {
    char const c = s[i];
    if (i == 4 || i == 7) {
      if (c != ':') { return false; }
    } else if (i == 10) {
      if (c != '.') { return false; }
    } else if (std::isxdigit(static_cast<unsigned char>(c)) == 0) {
      return false;
    }
  }
  return true;
}

/// Names of a device's control vectors: never a queue, whatever else they say.
bool is_control_vector_name(std::string_view name)
{
  static constexpr std::array<std::string_view, 6> control_tokens{
    "async", "cmd", "mgmnt", "mgmt", "ctrl", "aenq"};
  std::string lower(name);
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return std::any_of(control_tokens.begin(), control_tokens.end(), [&](std::string_view t) {
    return lower.find(t) != std::string::npos;
  });
}

/// The trailing decimal of @p s when it is a whole token (@c "TxRx-3" -> 3,
/// @c "3" -> 3, @c "mlx5_comp12" -> 12); @c nullopt otherwise.
std::optional<int> trailing_index(std::string_view s, bool require_separator)
{
  std::size_t digits = 0;
  while (digits < s.size() &&
         std::isdigit(static_cast<unsigned char>(s[s.size() - 1 - digits])) != 0) {
    ++digits;
  }
  if (digits == 0) { return std::nullopt; }
  auto const head = s.substr(0, s.size() - digits);
  if (require_separator && !head.empty() && head.back() != '-' && head.back() != '_') {
    return std::nullopt;
  }
  return parse_number<int>(s.substr(s.size() - digits));
}

/// The queue index named by an interrupt action @p name when it services a
/// queue of @p ifname (or of the device at @p pci); @c nullopt otherwise.
std::optional<int> queue_of_irq_name(std::string_view name,
                                     std::string_view ifname,
                                     std::string_view pci)
{
  if (name.empty() || is_control_vector_name(name)) { return std::nullopt; }

  // <if>-TxRx-N, <if>-Tx-Rx-N, <if>-rx-N, <if>-tx-N, <if>-N, and the
  // driver-prefixed <drv>-<if>-TxRx-N (i40e, ice).  The bare interface name (no
  // suffix) is the device's misc vector and fails here.
  auto const prefixed = [&](std::string_view token) -> std::optional<int> {
    if (token.empty()) { return std::nullopt; }
    for (auto pos = name.find(token); pos != std::string_view::npos;
         pos      = name.find(token, pos + 1)) {
      auto const end = pos + token.size();
      if (pos != 0 && name[pos - 1] != '-') { continue; }
      if (end + 1 >= name.size() || name[end] != '-') { continue; }
      return trailing_index(name.substr(end + 1), /*require_separator=*/true);
    }
    return std::nullopt;
  };
  if (auto q = prefixed(ifname)) { return q; }
  if (pci.empty()) { return std::nullopt; }
  if (auto q = prefixed(pci)) { return q; }

  // <driver>_comp<N>@pci:<addr> (mlx5) and the <name><N>@<addr> variant.
  auto const at = name.rfind('@');
  if (at == std::string_view::npos) { return std::nullopt; }
  auto owner = name.substr(at + 1);
  if (owner.substr(0, 4) == "pci:") { owner.remove_prefix(4); }
  if (owner != pci) { return std::nullopt; }
  return trailing_index(name.substr(0, at), /*require_separator=*/false);
}

/// IRQ number -> last action name, from @c /proc/interrupts.
std::vector<std::pair<int, std::string>> read_interrupt_names(discovery_roots const& roots)
{
  std::vector<std::pair<int, std::string>> out;
  std::ifstream in(proc_path(roots, "interrupts"));
  if (!in) { return out; }
  std::string line;
  std::getline(in, line);  // CPU header
  while (std::getline(in, line)) {
    auto const tokens = split_ws(line);
    if (tokens.size() < 2) { continue; }
    auto label = tokens.front();
    if (label.empty() || label.back() != ':') { continue; }
    label.remove_suffix(1);
    auto const irq = parse_number<int>(label);
    if (!irq) { continue; }  // NMI, LOC, ...
    auto name = tokens.back();
    if (!name.empty() && name.back() == ',') { name.remove_suffix(1); }
    out.emplace_back(*irq, std::string{name});
  }
  return out;
}

/// The device's MSI/MSI-X vectors; @c nullopt when the directory is absent.
std::optional<std::unordered_set<int>> read_msi_irqs(fs::path const& device_dir)
{
  std::error_code ec;
  auto const dir = device_dir / "msi_irqs";
  if (!fs::is_directory(dir, ec)) { return std::nullopt; }
  std::unordered_set<int> irqs;
  for (auto const& entry : fs::directory_iterator(dir, ec)) {
    if (auto irq = parse_number<int>(entry.path().filename().string())) { irqs.insert(*irq); }
  }
  return irqs;
}

std::string pci_address_of(fs::path const& device_link)
{
  std::error_code ec;
  fs::path target;
  if (fs::is_symlink(device_link, ec)) {
    target = fs::read_symlink(device_link, ec);
  } else if (fs::exists(device_link, ec)) {
    target = fs::canonical(device_link, ec);
  }
  if (ec || target.empty()) { return {}; }
  auto name = target.filename().string();
  return looks_like_pci_address(name) ? name : std::string{};
}

unsigned count_queue_dirs(fs::path const& queues_dir, std::string_view prefix)
{
  std::error_code ec;
  unsigned n = 0;
  if (!fs::is_directory(queues_dir, ec)) { return 0; }
  for (auto const& entry : fs::directory_iterator(queues_dir, ec)) {
    auto const name = entry.path().filename().string();
    if (name.size() > prefix.size() && std::string_view{name}.substr(0, prefix.size()) == prefix) {
      ++n;
    }
  }
  return n;
}

bool is_all_zero_hex(std::string_view s)
{
  return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c == '0'; });
}

std::optional<std::string> default_route_v4(discovery_roots const& roots)
{
  std::ifstream in(proc_path(roots, "net/route"));
  if (!in) { return std::nullopt; }
  std::string line;
  std::getline(in, line);  // header
  std::optional<std::string> best;
  auto best_metric = std::numeric_limits<std::uint64_t>::max();
  while (std::getline(in, line)) {
    // Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT
    auto const t = split_ws(line);
    if (t.size() < 8) { continue; }
    if (t[0] == "lo" || !is_all_zero_hex(t[1]) || !is_all_zero_hex(t[7])) { continue; }
    auto const flags  = parse_number<unsigned>(t[3], 16);
    auto const metric = parse_number<std::uint64_t>(t[6]);
    if (!flags || !metric || (*flags & rtf_up) == 0 || (*flags & rtf_reject) != 0) { continue; }
    if (*metric < best_metric) {
      best_metric = *metric;
      best        = std::string{t[0]};
    }
  }
  return best;
}

std::optional<std::string> default_route_v6(discovery_roots const& roots)
{
  std::ifstream in(proc_path(roots, "net/ipv6_route"));
  if (!in) { return std::nullopt; }
  std::string line;
  std::optional<std::string> best;
  auto best_metric = std::numeric_limits<std::uint64_t>::max();
  while (std::getline(in, line)) {
    // dst dst_plen src src_plen next_hop metric refcnt use flags dev
    auto const t = split_ws(line);
    if (t.size() < 10) { continue; }
    if (t[9] == "lo" || !is_all_zero_hex(t[0]) || !is_all_zero_hex(t[1])) { continue; }
    auto const flags  = parse_number<unsigned>(t[8], 16);
    auto const metric = parse_number<std::uint64_t>(t[5], 16);
    if (!flags || !metric || (*flags & rtf_up) == 0 || (*flags & rtf_reject) != 0) { continue; }
    if (*metric < best_metric) {
      best_metric = *metric;
      best        = std::string{t[9]};
    }
  }
  return best;
}

}  // namespace

bool nic_topology::irqs_steered(std::size_t max_cpus_per_irq) const noexcept
{
  if (queue_irqs.empty() || irq_cpus.empty() || node_cpus.empty()) { return false; }
  for (auto const& q : queue_irqs) {
    if (q.cpus.empty() || q.cpus.size() > max_cpus_per_irq) { return false; }
    // The effective mask alone proves nothing: x86 delivers every IRQ to one
    // CPU the kernel picks from the requested mask.  Steering is the operator
    // (or irqbalance) narrowing the requested mask itself.
    if (q.requested_cpus.size() > max_cpus_per_irq) { return false; }
  }
  // Strict subset: every IRQ CPU belongs to the node and the node has more.
  if (irq_cpus.size() >= node_cpus.size()) { return false; }
  return std::includes(node_cpus.begin(), node_cpus.end(), irq_cpus.begin(), irq_cpus.end());
}

std::vector<int> parse_cpu_list(std::string_view text)
{
  text = trim(text);
  std::set<int> cpus;
  if (text.empty()) { return {}; }
  constexpr int max_cpu = 1 << 20;  // sanity bound on a malformed range
  while (!text.empty()) {
    auto const comma = text.find(',');
    auto item        = trim(text.substr(0, comma));
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    if (item.empty()) { return {}; }

    // a | a-b | a-b:used/group
    int used  = 0;
    int group = 0;
    if (auto const colon = item.find(':'); colon != std::string_view::npos) {
      auto const stride = item.substr(colon + 1);
      auto const slash  = stride.find('/');
      if (slash == std::string_view::npos) { return {}; }
      auto const u = parse_number<int>(stride.substr(0, slash));
      auto const g = parse_number<int>(stride.substr(slash + 1));
      if (!u || !g || *u <= 0 || *g <= 0 || *u > *g) { return {}; }
      used  = *u;
      group = *g;
      item  = item.substr(0, colon);
    }
    auto const dash = item.find('-');
    auto const lo   = parse_number<int>(item.substr(0, dash));
    auto const hi = dash == std::string_view::npos ? lo : parse_number<int>(item.substr(dash + 1));
    if (!lo || !hi || *lo < 0 || *hi < *lo || *hi > max_cpu) { return {}; }
    for (int c = *lo; c <= *hi; ++c) {
      if (group == 0 || (c - *lo) % group < used) { cpus.insert(c); }
    }
  }
  return {cpus.begin(), cpus.end()};
}

std::string format_cpu_list(std::vector<int> const& cpus_in)
{
  std::vector<int> cpus(cpus_in);
  std::sort(cpus.begin(), cpus.end());
  cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
  std::ostringstream out;
  for (std::size_t i = 0; i < cpus.size();) {
    auto j = i;
    while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) {
      ++j;
    }
    if (i != 0) { out << ','; }
    out << cpus[i];
    if (j > i) { out << '-' << cpus[j]; }
    i = j + 1;
  }
  return out.str();
}

std::vector<int> online_cpus(discovery_roots const& roots)
{
  return read_cpu_list(sys_path(roots, "devices/system/cpu/online"));
}

std::vector<int> cpus_of_numa_node(int node, discovery_roots const& roots)
{
  if (node < 0) { return {}; }
  return read_cpu_list(
    sys_path(roots, "devices/system/node/node" + std::to_string(node) + "/cpulist"));
}

std::optional<std::string> default_route_interface(discovery_roots const& roots)
{
  if (auto v4 = default_route_v4(roots)) { return v4; }
  return default_route_v6(roots);
}

std::optional<std::string> select_egress_interface(std::string_view configured,
                                                   discovery_roots const& roots)
{
  if (!configured.empty()) {
    if (configured == "lo") { return std::nullopt; }
    return std::string{configured};
  }
  return default_route_interface(roots);
}

std::optional<nic_topology> discover_nic(std::string_view interface, discovery_roots const& roots)
{
  if (interface.empty() || interface == "lo" || interface.find('/') != std::string_view::npos) {
    return std::nullopt;
  }
  std::error_code ec;
  auto const if_dir = sys_path(roots, "class/net") / std::string{interface};
  if (!fs::exists(if_dir, ec)) { return std::nullopt; }

  nic_topology topo;
  topo.interface   = std::string{interface};
  auto const dev   = if_dir / "device";
  topo.pci_address = pci_address_of(dev);
  topo.numa_node   = read_int(dev / "numa_node").value_or(-1);
  topo.mtu         = read_int(if_dir / "mtu").value_or(0);
  topo.rx_queues   = count_queue_dirs(if_dir / "queues", "rx-");
  topo.tx_queues   = count_queue_dirs(if_dir / "queues", "tx-");

  auto const msi = read_msi_irqs(dev);
  for (auto const& [irq, name] : read_interrupt_names(roots)) {
    if (msi && msi->count(irq) == 0) { continue; }
    auto const queue = queue_of_irq_name(name, interface, topo.pci_address);
    if (!queue) { continue; }
    nic_queue_irq q;
    q.irq            = irq;
    q.queue          = *queue;
    auto const dir   = proc_path(roots, "irq") / std::to_string(irq);
    q.requested_cpus = read_cpu_list(dir / "smp_affinity_list");
    q.cpus           = read_cpu_list(dir / "effective_affinity_list");
    if (q.cpus.empty()) { q.cpus = q.requested_cpus; }
    topo.queue_irqs.push_back(std::move(q));
  }
  std::sort(topo.queue_irqs.begin(), topo.queue_irqs.end(), [](auto const& a, auto const& b) {
    return std::pair{a.queue, a.irq} < std::pair{b.queue, b.irq};
  });

  std::set<int> irq_cpus;
  for (auto const& q : topo.queue_irqs) {
    irq_cpus.insert(q.cpus.begin(), q.cpus.end());
  }
  topo.irq_cpus.assign(irq_cpus.begin(), irq_cpus.end());

  topo.node_cpus = cpus_of_numa_node(topo.numa_node, roots);
  if (topo.node_cpus.empty()) { topo.node_cpus = online_cpus(roots); }
  return topo;
}

}  // namespace cucascade::memory
