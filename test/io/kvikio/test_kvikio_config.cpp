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

#include <cucascade/io/kvikio/config.hpp>

#include <kvikio/defaults.hpp>

#include <catch2/catch.hpp>

#include <stdexcept>

using cucascade::io::apply_kvikio_defaults;
using cucascade::io::kvikio_config;

namespace {

/// kvikio::defaults is a process-global singleton, so every test here must put
/// back what it found or it leaks into the next one.
struct defaults_guard {
  unsigned int nthreads      = kvikio::defaults::thread_pool_nthreads();
  std::size_t task_size      = kvikio::defaults::task_size();
  std::size_t gds_threshold  = kvikio::defaults::gds_threshold();
  std::size_t bounce_size    = kvikio::defaults::bounce_buffer_size();
  bool dio_read              = kvikio::defaults::auto_direct_io_read();
  bool dio_overread          = kvikio::defaults::auto_direct_io_read_overread();
  bool pool_per_block_device = kvikio::defaults::thread_pool_per_block_device();

  ~defaults_guard()
  {
    kvikio::defaults::set_thread_pool_per_block_device(pool_per_block_device);
    kvikio::defaults::set_thread_pool_nthreads(nthreads);
    kvikio::defaults::set_task_size(task_size);
    kvikio::defaults::set_gds_threshold(gds_threshold);
    kvikio::defaults::set_bounce_buffer_size(bounce_size);
    kvikio::defaults::set_auto_direct_io_read(dio_read);
    kvikio::defaults::set_auto_direct_io_read_overread(dio_overread);
  }
};

}  // namespace

TEST_CASE("kvikio_config default-constructs with every field unset", "[kvikio][config]")
{
  kvikio_config cfg;

  CHECK_FALSE(cfg.nthreads.has_value());
  CHECK_FALSE(cfg.task_size.has_value());
  CHECK_FALSE(cfg.gds_threshold.has_value());
  CHECK_FALSE(cfg.bounce_buffer_size.has_value());
  CHECK_FALSE(cfg.auto_direct_io_read.has_value());
  CHECK_FALSE(cfg.auto_direct_io_read_overread.has_value());
  CHECK_FALSE(cfg.thread_pool_per_block_device.has_value());
  CHECK_FALSE(cfg.compat_mode.has_value());
}

TEST_CASE("apply_kvikio_defaults leaves kvikIO untouched for an empty config", "[kvikio][config]")
{
  defaults_guard guard;

  // Move every knob off its current value first, then apply an empty config and
  // confirm nothing moved back — an unset field must not overwrite.
  kvikio::defaults::set_thread_pool_nthreads(guard.nthreads + 3);
  kvikio::defaults::set_task_size(guard.task_size + 4096);
  kvikio::defaults::set_auto_direct_io_read(!guard.dio_read);

  apply_kvikio_defaults(kvikio_config{});

  CHECK(kvikio::defaults::thread_pool_nthreads() == guard.nthreads + 3);
  CHECK(kvikio::defaults::task_size() == guard.task_size + 4096);
  CHECK(kvikio::defaults::auto_direct_io_read() == !guard.dio_read);
}

TEST_CASE("apply_kvikio_defaults pushes engaged fields into kvikIO", "[kvikio][config]")
{
  defaults_guard guard;

  kvikio_config cfg;
  cfg.nthreads                     = 6;
  cfg.task_size                    = 2UL << 20;  // 2 MiB, page-aligned
  cfg.gds_threshold                = 512UL << 10;
  cfg.bounce_buffer_size           = 8UL << 20;
  cfg.auto_direct_io_read          = true;
  cfg.auto_direct_io_read_overread = true;
  cfg.thread_pool_per_block_device = false;

  apply_kvikio_defaults(cfg);

  CHECK(kvikio::defaults::thread_pool_nthreads() == 6);
  CHECK(kvikio::defaults::task_size() == (2UL << 20));
  CHECK(kvikio::defaults::gds_threshold() == (512UL << 10));
  CHECK(kvikio::defaults::bounce_buffer_size() == (8UL << 20));
  CHECK(kvikio::defaults::auto_direct_io_read());
  CHECK(kvikio::defaults::auto_direct_io_read_overread());
  CHECK_FALSE(kvikio::defaults::thread_pool_per_block_device());
}

TEST_CASE("apply_kvikio_defaults applies a partial config without disturbing the rest",
          "[kvikio][config]")
{
  defaults_guard guard;

  kvikio::defaults::set_gds_threshold(guard.gds_threshold + 1024);
  auto const untouched = kvikio::defaults::gds_threshold();

  kvikio_config cfg;
  cfg.nthreads = 2;
  apply_kvikio_defaults(cfg);

  CHECK(kvikio::defaults::thread_pool_nthreads() == 2);
  CHECK(kvikio::defaults::gds_threshold() == untouched);
}

TEST_CASE("apply_kvikio_defaults rejects zero sizes before mutating anything", "[kvikio][config]")
{
  defaults_guard guard;

  auto const before_nthreads = kvikio::defaults::thread_pool_nthreads();

  SECTION("zero nthreads")
  {
    kvikio_config cfg;
    cfg.nthreads = 0;
    CHECK_THROWS_AS(apply_kvikio_defaults(cfg), std::invalid_argument);
  }

  SECTION("zero task_size")
  {
    kvikio_config cfg;
    cfg.task_size = 0;
    CHECK_THROWS_AS(apply_kvikio_defaults(cfg), std::invalid_argument);
  }

  SECTION("zero bounce_buffer_size")
  {
    kvikio_config cfg;
    cfg.bounce_buffer_size = 0;
    CHECK_THROWS_AS(apply_kvikio_defaults(cfg), std::invalid_argument);
  }

  SECTION("a valid field alongside an invalid one is not applied")
  {
    kvikio_config cfg;
    cfg.nthreads  = before_nthreads + 5;
    cfg.task_size = 0;
    CHECK_THROWS_AS(apply_kvikio_defaults(cfg), std::invalid_argument);
    // Validation runs before any setter, so nthreads must be unchanged.
    CHECK(kvikio::defaults::thread_pool_nthreads() == before_nthreads);
  }
}

TEST_CASE("kvikio_config accepts a zero gds_threshold (always use GDS)", "[kvikio][config]")
{
  defaults_guard guard;

  kvikio_config cfg;
  cfg.gds_threshold = 0;
  CHECK_NOTHROW(apply_kvikio_defaults(cfg));
  CHECK(kvikio::defaults::gds_threshold() == 0);
}

TEST_CASE("kvikio_config carries compat_mode without touching kvikIO globals", "[kvikio][config]")
{
  defaults_guard guard;

  auto const before = kvikio::defaults::compat_mode();

  kvikio_config cfg;
  cfg.compat_mode = kvikio::CompatMode::ON;
  apply_kvikio_defaults(cfg);

  // compat_mode rides the FileHandle constructor instead, so the global default
  // must be left exactly as it was.
  CHECK(kvikio::defaults::compat_mode() == before);
}
