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

#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/platform.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE("cufile loader resolves every read entry point or none", "[io][cufile]")
{
  auto const* const api = cucascade::io::cufile::cufile_api::get();
  // Loaded once: later calls see the same table (or the same absence).
  CHECK(api == cucascade::io::cufile::cufile_api::get());
  if (api == nullptr) {
    CHECK_FALSE(cucascade::io::cufile::cufile_api::open_driver());
    CHECK(cucascade::io::cufile::cufile_api::version() == 0);
    return;
  }
  CHECK(api->driver_open != nullptr);
  CHECK(api->driver_get_properties != nullptr);
  CHECK(api->handle_register != nullptr);
  CHECK(api->handle_deregister != nullptr);
  CHECK(api->read != nullptr);
}

TEST_CASE("cufile platform probe is cached and self-consistent", "[io][cufile]")
{
  auto const& first  = cucascade::io::cufile::probe_platform();
  auto const& second = cucascade::io::cufile::probe_platform();
  CHECK(&first == &second);
  CHECK(first.library_available == (cucascade::io::cufile::cufile_api::get() != nullptr));
  CHECK(cucascade::io::cufile::gds_available() ==
        (first.driver_opened && first.properties_available && !first.compat_only &&
         !first.forced_compat));
  if (!first.driver_opened) {
    CHECK_FALSE(first.properties_available);
    return;
  }
  CHECK(first.library_available);
  if (!first.properties_available) return;  // mode unknown, not compatibility-only
  CHECK(first.compat_only == (first.dstatusflags == 0));
  CHECK(first.max_direct_io_size_kb > 0);
  INFO("cuFile " << first.version << ", dstatusflags " << first.dstatusflags
                 << (first.compat_only ? " (compatibility mode)" : " (GDS)"));
  CHECK(first.version > 0);
}
