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

#include <cucascade/memory/net_topology.hpp>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace cucascade::memory;
namespace fs = std::filesystem;

namespace {

/// A fake host: a temporary directory holding @c sys/ and @c proc/ subtrees.
class fake_host {
 public:
  fake_host()
  {
    static std::atomic<int> counter{0};
    _root = fs::temp_directory_path() / ("cucascade_net_topology_" + std::to_string(::getpid()) +
                                         "_" + std::to_string(counter.fetch_add(1)));
    fs::remove_all(_root);
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

  void write(std::string const& rel, std::string const& content) const
  {
    auto const path = _root / rel;
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
  }

  void node(int id, std::string const& cpulist) const
  {
    write("sys/devices/system/node/node" + std::to_string(id) + "/cpulist", cpulist + "\n");
  }
  void online(std::string const& cpulist) const
  {
    write("sys/devices/system/cpu/online", cpulist + "\n");
  }

  /// A PCI-backed interface with @p n_queues rx/tx queues.  @p msi_irqs lists
  /// the device's MSI-X vectors; @c nullopt omits the directory.
  void nic(std::string const& ifname,
           std::string const& pci,
           int numa,
           int mtu,
           int n_queues,
           std::optional<std::vector<int>> const& msi_irqs) const
  {
    auto const dev = _root / "sys/devices/pci0000:00" / pci;
    fs::create_directories(dev);
    write("sys/devices/pci0000:00/" + pci + "/numa_node", std::to_string(numa) + "\n");
    if (msi_irqs) {
      fs::create_directories(dev / "msi_irqs");
      for (int irq : *msi_irqs) {
        write("sys/devices/pci0000:00/" + pci + "/msi_irqs/" + std::to_string(irq), "msix\n");
      }
    }
    auto const if_dir = _root / "sys/class/net" / ifname;
    fs::create_directories(if_dir);
    fs::create_symlink(dev, if_dir / "device");
    write("sys/class/net/" + ifname + "/mtu", std::to_string(mtu) + "\n");
    for (int q = 0; q < n_queues; ++q) {
      fs::create_directories(if_dir / "queues" / ("rx-" + std::to_string(q)));
      fs::create_directories(if_dir / "queues" / ("tx-" + std::to_string(q)));
    }
  }

  /// One @c /proc/interrupts row plus its @c /proc/irq/<irq>/ affinity files.
  /// An empty @p effective omits @c effective_affinity_list.
  void irq(int irq, std::string const& name, std::string const& smp, std::string const& effective)
  {
    _interrupts +=
      std::to_string(irq) + ":          5          7  IR-PCI-MSI 1-edge      " + name + "\n";
    write("proc/interrupts", "           CPU0       CPU1\n" + _interrupts);
    write("proc/irq/" + std::to_string(irq) + "/smp_affinity_list", smp + "\n");
    if (!effective.empty()) {
      write("proc/irq/" + std::to_string(irq) + "/effective_affinity_list", effective + "\n");
    }
  }

 private:
  fs::path _root;
  std::string _interrupts;
};

/// 8 queues on ens5 (IRQs 100..107) whose IRQs are requested and delivered
/// according to @p effective(q); node0 = 0-47.
template <typename Affinity>
void eight_queue_host(fake_host& host, Affinity effective)
{
  host.online("0-95");
  host.node(0, "0-47");
  host.node(1, "48-95");
  std::vector<int> msi{99};
  for (int q = 0; q < 8; ++q) {
    msi.push_back(100 + q);
  }
  host.nic("ens5", "0000:00:05.0", 0, 9001, 8, msi);
  host.irq(99, "ens5", "0-95", "3");  // misc vector, bare interface name
  for (int q = 0; q < 8; ++q) {
    host.irq(100 + q, "ens5-TxRx-" + std::to_string(q), effective(q), effective(q));
  }
}

}  // namespace

TEST_CASE("parse_cpu_list handles the kernel's list syntax", "[net_topology]")
{
  CHECK(parse_cpu_list("0-3,8,10-11\n") == std::vector<int>{0, 1, 2, 3, 8, 10, 11});
  CHECK(parse_cpu_list("5") == std::vector<int>{5});
  CHECK(parse_cpu_list("8,0-1,1") == std::vector<int>{0, 1, 8});
  CHECK(parse_cpu_list("0-7:2/4") == std::vector<int>{0, 1, 4, 5});
  CHECK(parse_cpu_list("").empty());
  CHECK(parse_cpu_list("3-1").empty());
  CHECK(parse_cpu_list("a").empty());
  CHECK(parse_cpu_list("1,,2").empty());
  CHECK(format_cpu_list({5, 0, 1, 2, 8, 9}) == "0-2,5,8-9");
  CHECK(format_cpu_list({}).empty());
}

TEST_CASE("one IRQ per queue on CPUs 0-7 of a 48-core node is steered", "[net_topology]")
{
  fake_host host;
  eight_queue_host(host, [](int q) { return std::to_string(q); });
  auto const nic = discover_nic("ens5", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->interface == "ens5");
  CHECK(nic->pci_address == "0000:00:05.0");
  CHECK(nic->numa_node == 0);
  CHECK(nic->mtu == 9001);
  CHECK(nic->rx_queues == 8);
  CHECK(nic->tx_queues == 8);
  REQUIRE(nic->queue_irqs.size() == 8);  // the misc vector (IRQ 99) is excluded
  for (int q = 0; q < 8; ++q) {
    CHECK(nic->queue_irqs[q].queue == q);
    CHECK(nic->queue_irqs[q].irq == 100 + q);
    CHECK(nic->queue_irqs[q].cpus == std::vector<int>{q});
    CHECK(nic->queue_irqs[q].requested_cpus == std::vector<int>{q});
  }
  CHECK(nic->irq_cpus == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
  CHECK(nic->node_cpus.size() == 48);
  CHECK(nic->irqs_steered());
}

TEST_CASE("IRQs delivered across the whole node are spread", "[net_topology]")
{
  fake_host host;
  eight_queue_host(host, [](int) { return std::string{"0-47"}; });
  auto const nic = discover_nic("ens5", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->irq_cpus.size() == 48);
  CHECK_FALSE(nic->irqs_steered());
  CHECK_FALSE(nic->irqs_steered(64));  // covers the node: not a strict subset
}

TEST_CASE("single-CPU IRQs that together cover the node are spread", "[net_topology]")
{
  fake_host host;
  host.online("0-7");
  host.node(0, "0-7");
  host.nic("eth0", "0000:3b:00.0", 0, 1500, 8, std::nullopt);
  for (int q = 0; q < 8; ++q) {
    host.irq(200 + q, "eth0-rx-" + std::to_string(q), std::to_string(q), std::to_string(q));
  }
  auto const nic = discover_nic("eth0", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->queue_irqs.size() == 8);
  CHECK_FALSE(nic->irqs_steered());
}

TEST_CASE("a kernel-picked delivery CPU under a wide requested mask is spread", "[net_topology]")
{
  // What an untuned x86 host reports: effective_affinity_list is one CPU per
  // IRQ, chosen by the kernel from smp_affinity_list = every CPU.
  fake_host host;
  host.online("0-79");
  host.node(0, "0-79");
  host.nic("eno1", "0003:02:00.0", 0, 1500, 8, std::nullopt);
  for (int q = 0; q < 8; ++q) {
    host.irq(288 + q, "eno1-TxRx-" + std::to_string(q), "0-79", std::to_string(37 + q));
  }
  auto const nic = discover_nic("eno1", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->irq_cpus == std::vector<int>{37, 38, 39, 40, 41, 42, 43, 44});
  CHECK_FALSE(nic->irqs_steered());

  // Falling back to smp_affinity_list when there is no effective list.
  fake_host fallback;
  fallback.online("0-79");
  fallback.node(0, "0-79");
  fallback.nic("eno1", "0003:02:00.0", 0, 1500, 1, std::nullopt);
  fallback.irq(288, "eno1-TxRx-0", "12", "");
  auto const fb = discover_nic("eno1", fallback.roots());
  REQUIRE(fb.has_value());
  CHECK(fb->irq_cpus == std::vector<int>{12});
  CHECK(fb->irqs_steered());
}

TEST_CASE("the per-IRQ CPU threshold decides two-CPU IRQs", "[net_topology]")
{
  fake_host host;
  eight_queue_host(host,
                   [](int q) { return std::to_string(2 * q) + "-" + std::to_string(2 * q + 1); });
  auto const nic = discover_nic("ens5", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->irq_cpus.size() == 16);
  CHECK_FALSE(nic->irqs_steered(1));
  CHECK(nic->irqs_steered(2));
}

TEST_CASE("an unknown NUMA node compares against the online CPUs", "[net_topology]")
{
  fake_host host;
  host.online("0-15");
  host.nic("ens5", "0000:00:05.0", -1, 1500, 4, std::vector<int>{10, 11, 12, 13});
  for (int q = 0; q < 4; ++q) {
    host.irq(10 + q, "ens5-Tx-Rx-" + std::to_string(q), std::to_string(q), std::to_string(q));
  }
  auto nic = discover_nic("ens5", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->numa_node == -1);
  CHECK(nic->node_cpus.size() == 16);
  CHECK(nic->irqs_steered());

  // The same IRQ CPUs against an online set they cover are spread.
  nic->node_cpus = {0, 1, 2, 3};
  CHECK_FALSE(nic->irqs_steered());
}

TEST_CASE("without msi_irqs, queue IRQs are found by name", "[net_topology]")
{
  SECTION("ENA")
  {
    fake_host host;
    host.online("0-31");
    host.node(0, "0-31");
    host.nic("ens5", "0000:00:05.0", 0, 9001, 4, std::nullopt);
    host.irq(30, "ena-mgmnt@pci:0000:00:05.0", "0-31", "0");
    host.irq(31, "ens5-Tx-Rx-0", "4", "4");
    host.irq(32, "ens5-Tx-Rx-1", "5", "5");
    host.irq(33, "ens50-Tx-Rx-0", "0-31", "6");  // a different interface
    host.irq(34, "nvme0q1", "0-31", "7");
    auto const nic = discover_nic("ens5", host.roots());
    REQUIRE(nic.has_value());
    REQUIRE(nic->queue_irqs.size() == 2);
    CHECK(nic->queue_irqs[0].irq == 31);
    CHECK(nic->queue_irqs[1].irq == 32);
    CHECK(nic->irq_cpus == std::vector<int>{4, 5});
    CHECK(nic->irqs_steered());
  }
  SECTION("mlx5")
  {
    fake_host host;
    host.online("0-63");
    host.node(1, "32-63");
    host.nic("enp8s0f0", "0000:08:00.0", 1, 1500, 3, std::nullopt);
    host.irq(50, "mlx5_async0@pci:0000:08:00.0", "32-63", "32");
    host.irq(51, "mlx5_comp0@pci:0000:08:00.0", "32", "");  // no effective list
    host.irq(52, "mlx5_comp1@pci:0000:08:00.0", "33", "");
    host.irq(53, "mlx5_comp3@pci:0000:08:00.0", "35", "");
    host.irq(54, "mlx5_comp2@pci:0000:08:00.1", "34", "");  // the other port
    host.irq(55, "0000:08:00.0-cmd", "32-63", "40");
    auto const nic = discover_nic("enp8s0f0", host.roots());
    REQUIRE(nic.has_value());
    CHECK(nic->numa_node == 1);
    REQUIRE(nic->queue_irqs.size() == 3);
    CHECK(nic->queue_irqs[0].queue == 0);
    CHECK(nic->queue_irqs[1].queue == 1);
    CHECK(nic->queue_irqs[2].queue == 3);
    CHECK(nic->queue_irqs[2].irq == 53);
    CHECK(nic->irq_cpus == std::vector<int>{32, 33, 35});
    CHECK(nic->irqs_steered());
  }
  SECTION("driver-prefixed names")
  {
    fake_host host;
    host.online("0-15");
    host.node(0, "0-15");
    host.nic("eth1", "0000:5e:00.0", 0, 1500, 2, std::nullopt);
    host.irq(70, "i40e-eth1-TxRx-0", "0-15", "1");
    host.irq(71, "i40e-eth1-TxRx-1", "0-15", "2");
    host.irq(72, "i40e-eth10-TxRx-0", "0-15", "3");
    auto const nic = discover_nic("eth1", host.roots());
    REQUIRE(nic.has_value());
    CHECK(nic->irq_cpus == std::vector<int>{1, 2});
  }
}

TEST_CASE("msi_irqs restricts name matches to the device's own vectors", "[net_topology]")
{
  fake_host host;
  host.online("0-15");
  host.node(0, "0-15");
  host.nic("eno1", "0003:02:00.0", 0, 1500, 2, std::vector<int>{287, 288, 289});
  host.irq(287, "eno1", "0-15", "1");
  host.irq(288, "eno1-TxRx-0", "0-15", "2");
  host.irq(289, "eno1-TxRx-1", "0-15", "3");
  host.irq(400, "eno1-TxRx-2", "0-15", "4");  // not an MSI vector of this device
  auto const nic = discover_nic("eno1", host.roots());
  REQUIRE(nic.has_value());
  CHECK(nic->pci_address == "0003:02:00.0");
  CHECK(nic->irq_cpus == std::vector<int>{2, 3});
}

TEST_CASE("missing and loopback interfaces are not discovered", "[net_topology]")
{
  fake_host host;
  host.online("0-3");
  CHECK_FALSE(discover_nic("eth9", host.roots()).has_value());
  CHECK_FALSE(discover_nic("lo", host.roots()).has_value());
  CHECK_FALSE(discover_nic("", host.roots()).has_value());
  CHECK_FALSE(discover_nic("../x", host.roots()).has_value());
}

TEST_CASE("default route interface", "[net_topology]")
{
  std::string const header =
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n";
  SECTION("IPv4, lowest metric, skipping lo, down and non-default routes")
  {
    fake_host host;
    host.write("proc/net/route",
               header +
                 "lo\t00000000\t00000000\t0001\t0\t0\t0\t00000000\t0\t0\t0\n"
                 "eth0\t0000750A\t00000000\t0001\t0\t0\t0\t00E0FFFF\t0\t0\t0\n"
                 "eth2\t00000000\t0100750A\t0002\t0\t0\t1\t00000000\t0\t0\t0\n"
                 "eth1\t00000000\t0100750A\t0003\t0\t0\t200\t00000000\t0\t0\t0\n"
                 "eno1\t00000000\t0100750A\t0003\t0\t0\t100\t00000000\t0\t0\t0\n");
    CHECK(default_route_interface(host.roots()) == std::optional<std::string>{"eno1"});
    CHECK(select_egress_interface("", host.roots()) == std::optional<std::string>{"eno1"});
    CHECK(select_egress_interface("ens9", host.roots()) == std::optional<std::string>{"ens9"});
    CHECK_FALSE(select_egress_interface("lo", host.roots()).has_value());
  }
  SECTION("IPv6 when there is no IPv4 default")
  {
    fake_host host;
    host.write("proc/net/route", header);
    std::string const zero32(32, '0');
    host.write("proc/net/ipv6_route",
               zero32 + " 00 " + zero32 + " 00 " + zero32 +
                 " ffffffff 00000001 00000000 00200200       lo\n" +  // reject
                 "fe800000000000000000000000000000 40 " + zero32 + " 00 " + zero32 +
                 " 00000100 00000001 00000000 00000001     ens6\n" + zero32 + " 00 " + zero32 +
                 " 00 fe800000000000000000000000000001 00000400 00000001 00000000 00000003  "
                 "   ens7\n");
    CHECK(default_route_interface(host.roots()) == std::optional<std::string>{"ens7"});
  }
  SECTION("no default route")
  {
    fake_host host;
    host.write("proc/net/route", header);
    CHECK_FALSE(default_route_interface(host.roots()).has_value());
    CHECK_FALSE(select_egress_interface("", host.roots()).has_value());
  }
}

TEST_CASE("NUMA node and online CPU sets", "[net_topology]")
{
  fake_host host;
  host.online("0-3,8-11");
  host.node(1, "8-11");
  CHECK(online_cpus(host.roots()) == std::vector<int>{0, 1, 2, 3, 8, 9, 10, 11});
  CHECK(cpus_of_numa_node(1, host.roots()) == std::vector<int>{8, 9, 10, 11});
  CHECK(cpus_of_numa_node(0, host.roots()).empty());
  CHECK(cpus_of_numa_node(-1, host.roots()).empty());
}

TEST_CASE("discovery on this host (informational)", "[net_topology]")
{
  auto const iface = select_egress_interface("");
  std::cout << "[net_topology] host egress interface: " << iface.value_or("<none>") << "\n";
  if (!iface) { return; }
  auto const nic = discover_nic(*iface);
  REQUIRE(nic.has_value());
  std::cout << "[net_topology] " << nic->interface << " pci " << nic->pci_address << " numa "
            << nic->numa_node << " mtu " << nic->mtu << " rx/tx queues " << nic->rx_queues << "/"
            << nic->tx_queues << " node cpus " << format_cpu_list(nic->node_cpus) << " steered "
            << (nic->irqs_steered() ? "yes" : "no") << "\n";
  for (auto const& q : nic->queue_irqs) {
    std::cout << "[net_topology]   irq " << q.irq << " queue " << q.queue << " cpus "
              << format_cpu_list(q.cpus) << " (requested " << format_cpu_list(q.requested_cpus)
              << ")\n";
  }
}
