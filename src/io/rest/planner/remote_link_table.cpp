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

// EC2 instance type -> NIC bandwidth (Gbps), the controller's link seed on EC2 (C4; ported from
// gpuio's src/planner/remote_link_table.cpp). Source: the "Network specifications" tables of the
// AWS instance type pages, fetched 2026-10-12:
//   https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html  (c6in, c7gn, c7i, c7a)
//   https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html  (m6a, m6i, m7i, m7a)
//   https://docs.aws.amazon.com/ec2/latest/instancetypes/mo.html  (r7i)
//   https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html  (g6e, g7e, p5, p5e, p5en, trn1,
//                                                                  trn1n, trn2, trn2u)
//   https://docs.aws.amazon.com/ec2/latest/instancetypes/so.html  (i4i, i7ie, im4gn)
//   https://docs.aws.amazon.com/ec2/latest/instancetypes/hpc.html (hpc7g)
// "Baseline / Burst" rows ("Up to X") count as the burst figure X; multi-card totals (p5.48xlarge
// 3200, trn2.48xlarge 16 x 200, trn1.32xlarge 8 x 100) are the page totals (the controller's
// ceiling is max_connections_per_reactor x live reactors anyway).

#include "io/rest/planner/planner_remote.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <string_view>

namespace cucascade::io::rest::planner {
namespace {

struct nic {
  std::string_view type;
  double gbps;
};

// Sorted by type (binary search).
constexpr auto table = std::to_array<nic>({
  {"c6in.12xlarge", 75},    {"c6in.16xlarge", 100},   {"c6in.24xlarge", 150},
  {"c6in.2xlarge", 40},     {"c6in.32xlarge", 200},   {"c6in.4xlarge", 50},
  {"c6in.8xlarge", 50},     {"c6in.large", 25},       {"c6in.metal", 200},
  {"c6in.xlarge", 30},      {"c7a.12xlarge", 18.75},  {"c7a.16xlarge", 25},
  {"c7a.24xlarge", 37.5},   {"c7a.2xlarge", 12.5},    {"c7a.32xlarge", 50},
  {"c7a.48xlarge", 50},     {"c7a.4xlarge", 12.5},    {"c7a.8xlarge", 12.5},
  {"c7a.large", 12.5},      {"c7a.medium", 12.5},     {"c7a.metal-48xl", 50},
  {"c7a.xlarge", 12.5},     {"c7gn.12xlarge", 150},   {"c7gn.16xlarge", 200},
  {"c7gn.2xlarge", 50},     {"c7gn.4xlarge", 50},     {"c7gn.8xlarge", 100},
  {"c7gn.large", 30},       {"c7gn.medium", 25},      {"c7gn.metal", 200},
  {"c7gn.xlarge", 40},      {"c7i.12xlarge", 18.75},  {"c7i.16xlarge", 25},
  {"c7i.24xlarge", 37.5},   {"c7i.2xlarge", 12.5},    {"c7i.48xlarge", 50},
  {"c7i.4xlarge", 12.5},    {"c7i.8xlarge", 12.5},    {"c7i.large", 12.5},
  {"c7i.metal-24xl", 37.5}, {"c7i.metal-48xl", 50},   {"c7i.xlarge", 12.5},
  {"g6e.12xlarge", 100},    {"g6e.16xlarge", 35},     {"g6e.24xlarge", 200},
  {"g6e.2xlarge", 20},      {"g6e.48xlarge", 400},    {"g6e.4xlarge", 20},
  {"g6e.8xlarge", 25},      {"g6e.xlarge", 20},       {"g7e.12xlarge", 400},
  {"g7e.24xlarge", 800},    {"g7e.2xlarge", 50},      {"g7e.48xlarge", 1600},
  {"g7e.4xlarge", 50},      {"g7e.8xlarge", 100},     {"hpc7g.16xlarge", 200},
  {"hpc7g.4xlarge", 200},   {"hpc7g.8xlarge", 200},   {"i4i.12xlarge", 28.12},
  {"i4i.16xlarge", 37.5},   {"i4i.24xlarge", 56.25},  {"i4i.2xlarge", 12},
  {"i4i.32xlarge", 75},     {"i4i.4xlarge", 25},      {"i4i.8xlarge", 18.75},
  {"i4i.large", 10},        {"i4i.metal", 75},        {"i4i.xlarge", 10},
  {"i7ie.12xlarge", 50},    {"i7ie.18xlarge", 75},    {"i7ie.24xlarge", 100},
  {"i7ie.2xlarge", 25},     {"i7ie.3xlarge", 25},     {"i7ie.48xlarge", 200},
  {"i7ie.6xlarge", 25},     {"i7ie.large", 25},       {"i7ie.metal-24xl", 100},
  {"i7ie.metal-48xl", 200}, {"i7ie.xlarge", 25},      {"im4gn.16xlarge", 100},
  {"im4gn.2xlarge", 25},    {"im4gn.4xlarge", 25},    {"im4gn.8xlarge", 50},
  {"im4gn.large", 25},      {"im4gn.xlarge", 25},     {"m6a.12xlarge", 18.75},
  {"m6a.16xlarge", 25},     {"m6a.24xlarge", 37.5},   {"m6a.2xlarge", 12.5},
  {"m6a.32xlarge", 50},     {"m6a.48xlarge", 50},     {"m6a.4xlarge", 12.5},
  {"m6a.8xlarge", 12.5},    {"m6a.large", 12.5},      {"m6a.metal", 50},
  {"m6a.xlarge", 12.5},     {"m6i.12xlarge", 18.75},  {"m6i.16xlarge", 25},
  {"m6i.24xlarge", 37.5},   {"m6i.2xlarge", 12.5},    {"m6i.32xlarge", 50},
  {"m6i.4xlarge", 12.5},    {"m6i.8xlarge", 12.5},    {"m6i.large", 12.5},
  {"m6i.metal", 50},        {"m6i.xlarge", 12.5},     {"m7a.12xlarge", 18.75},
  {"m7a.16xlarge", 25},     {"m7a.24xlarge", 37.5},   {"m7a.2xlarge", 12.5},
  {"m7a.32xlarge", 50},     {"m7a.48xlarge", 50},     {"m7a.4xlarge", 12.5},
  {"m7a.8xlarge", 12.5},    {"m7a.large", 12.5},      {"m7a.medium", 12.5},
  {"m7a.metal-48xl", 50},   {"m7a.xlarge", 12.5},     {"m7i.12xlarge", 18.75},
  {"m7i.16xlarge", 25},     {"m7i.24xlarge", 37.5},   {"m7i.2xlarge", 12.5},
  {"m7i.48xlarge", 50},     {"m7i.4xlarge", 12.5},    {"m7i.8xlarge", 12.5},
  {"m7i.large", 12.5},      {"m7i.metal-24xl", 37.5}, {"m7i.metal-48xl", 50},
  {"m7i.xlarge", 12.5},     {"p5.48xlarge", 3200},    {"p5.4xlarge", 100},
  {"p5e.48xlarge", 3200},   {"p5en.48xlarge", 3200},  {"r7i.12xlarge", 18.75},
  {"r7i.16xlarge", 25},     {"r7i.24xlarge", 37.5},   {"r7i.2xlarge", 12.5},
  {"r7i.48xlarge", 50},     {"r7i.4xlarge", 12.5},    {"r7i.8xlarge", 12.5},
  {"r7i.large", 12.5},      {"r7i.metal-24xl", 37.5}, {"r7i.metal-48xl", 50},
  {"r7i.xlarge", 12.5},     {"trn1.2xlarge", 12.5},   {"trn1.32xlarge", 800},
  {"trn1n.32xlarge", 1600}, {"trn2.3xlarge", 200},    {"trn2.48xlarge", 3200},
  {"trn2u.48xlarge", 3200},
});

constexpr bool sorted() noexcept
{
  for (std::size_t i = 1; i < table.size(); ++i) {
    if (!(table[i - 1].type < table[i].type)) { return false; }
  }
  return true;
}
static_assert(sorted(), "keep the NIC table sorted by type (binary search)");

}  // namespace

std::optional<double> ec2_instance_link_gbps(std::string_view instance_type) noexcept
{
  while (!instance_type.empty() &&
         std::isspace(static_cast<unsigned char>(instance_type.front())) != 0) {
    instance_type.remove_prefix(1);
  }
  while (!instance_type.empty() &&
         std::isspace(static_cast<unsigned char>(instance_type.back())) != 0) {
    instance_type.remove_suffix(1);
  }
  std::array<char, 32> lower{};
  if (instance_type.empty() || instance_type.size() > lower.size()) { return std::nullopt; }
  for (std::size_t i = 0; i < instance_type.size(); ++i) {
    lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(instance_type[i])));
  }
  std::string_view const key(lower.data(), instance_type.size());
  auto const it = std::lower_bound(
    table.begin(), table.end(), key, [](nic const& n, std::string_view k) { return n.type < k; });
  if (it == table.end() || it->type != key) { return std::nullopt; }
  return it->gbps;
}

double link_seed_gbps(std::optional<std::string_view> instance_type) noexcept
{
  if (!instance_type || instance_type->empty()) { return default_link_gbps; }  // off EC2
  return ec2_instance_link_gbps(*instance_type).value_or(unknown_ec2_link_gbps);
}

}  // namespace cucascade::io::rest::planner
