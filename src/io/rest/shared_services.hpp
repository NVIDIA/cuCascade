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

#pragma once

#include "io/rest/net/dns_resolver.hpp"

#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/tls/tls.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <string_view>

/**
 * @file shared_services.hpp
 * @brief Transport state shared by every reactor of one REST pool (one per
 *        @c rest_reactor::reactor_context): the DNS resolver thread and the TLS provider of the
 *        pool's policy.
 */

namespace cucascade::io::rest::detail {

class shared_services {
 public:
  explicit shared_services(config const& cfg);
  ~shared_services();

  shared_services(shared_services const&)            = delete;
  shared_services& operator=(shared_services const&) = delete;

  /// The resolver (its thread starts on the first host name).
  [[nodiscard]] net::dns_resolver& dns() noexcept { return *_dns; }

  /// The TLS provider of the pool's policy, built on first use.
  /// @throws std::system_error when the policy cannot be applied (retried on the next call).
  [[nodiscard]] std::shared_ptr<tls::provider const> tls();

  /// Whether this host is an EC2 instance (DMI), read once (kTLS receive placement).
  [[nodiscard]] bool on_ec2() const noexcept { return _on_ec2; }

 private:
  tls::config _tls_policy;
  std::unique_ptr<net::dns_resolver> _dns;
  mutable std::mutex _mutex;
  std::shared_ptr<tls::provider const> _tls;
  bool _on_ec2{false};
};

}  // namespace cucascade::io::rest::detail
