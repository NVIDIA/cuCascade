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

#include "io/cache/controlled_reactor.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_errors.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using namespace cucascade::test;

constexpr std::size_t read_size = 4096;

/// Reads through one datasource, each held by the test until it fails or completes it.
struct held_reads {
  held_reads()
    : context(std::make_shared<controlled_ioctx>()),
      source(std::make_unique<cucascade::io::datasource>(
        context, held_object("held.bin", "\"one\"", 2 * read_size))),
      destination(2 * read_size, stream)
  {
    context->start();
  }

  ~held_reads()
  {
    first.cancel();
    second.cancel();
    context->shutdown();
    stream.synchronize_no_throw();
  }

  held_reads(held_reads const&)            = delete;
  held_reads& operator=(held_reads const&) = delete;

  /// Take the next read off the backend into @p held.  On failure every read is settled, held or
  /// queued, so the futures can still be dropped while the failure unwinds.
  void hold(held_request& held)
  {
    held.request = context->backend().wait_take(2s);
    if (held.request == nullptr) {
      first.cancel();
      second.cancel();
      context->shutdown();
    }
    REQUIRE(held.request != nullptr);
  }

  /// Read @p slot's half of the device destination and hold the read in @p held.
  std::future<std::size_t> device_read(std::size_t slot, held_request& held)
  {
    auto* bytes = static_cast<std::uint8_t*>(destination.data()) + slot * read_size;
    auto result = source->device_read_async(slot * read_size, read_size, bytes, stream);
    hold(held);
    return result;
  }

  std::shared_ptr<controlled_ioctx> context;
  std::unique_ptr<cucascade::io::datasource> source;
  rmm::cuda_stream stream;
  rmm::device_buffer destination;
  held_request first;
  held_request second;
};

/// Cancels the held reads when destroyed.  Declared after the futures that join them, so a failed
/// assertion unwinds into a cancellation instead of waiting for a read nobody completes.
struct cancel_on_exit {
  held_reads& reads;
  ~cancel_on_exit()
  {
    reads.first.cancel();
    reads.second.cancel();
  }
};

/// Run @p body on its own thread and wait until it has started.
template <typename Body>
std::future<void> started(Body body)
{
  std::promise<void> entered;
  auto running = entered.get_future();
  auto result  = std::async(std::launch::async,
                           [body = std::move(body), entered = std::move(entered)]() mutable {
                             entered.set_value();
                             body();
                           });
  running.wait();
  return result;
}

void fail(held_request& held) { held.fail("held.bin", "\"one\"", "\"two\""); }

}  // namespace

TEST_CASE("dropping a read's future waits for the read", "[datasource][gpu_execution]")
{
  held_reads reads;
  auto read = reads.device_read(0, reads.first);

  auto dropper = started([read = std::move(read)]() mutable { read = {}; });
  cancel_on_exit const cancel{reads};
  CHECK(dropper.wait_for(100ms) == std::future_status::timeout);

  reads.first.complete(0x5a);
  CHECK(dropper.wait_for(2s) == std::future_status::ready);
}

TEST_CASE("unwinding from a failed read waits for the other reads before freeing their buffers",
          "[datasource][gpu_execution]")
{
  held_reads reads;
  std::vector<std::future<std::size_t>> futures;
  futures.push_back(reads.device_read(0, reads.first));
  futures.push_back(reads.device_read(1, reads.second));
  fail(reads.first);

  // Like cuDF's parquet reader before 26.12: stop at the first failure and drop the other reads
  auto caller = started([futures = std::move(futures)]() mutable {
    auto pending = std::move(futures);
    for (auto& read : pending) {
      read.get();
    }
  });
  cancel_on_exit const cancel{reads};
  CHECK(caller.wait_for(100ms) == std::future_status::timeout);

  reads.second.complete(0x5a);
  REQUIRE(caller.wait_for(2s) == std::future_status::ready);
  CHECK_THROWS_AS(caller.get(), cucascade::io::object_changed_error);
}

TEST_CASE("a read's result does not wait for the datasource's other reads",
          "[datasource][gpu_execution]")
{
  held_reads reads;
  auto first  = reads.device_read(0, reads.first);
  auto second = reads.device_read(1, reads.second);
  fail(reads.first);

  auto failed = std::async(std::launch::async, [&first] { first.get(); });
  cancel_on_exit const cancel{reads};
  REQUIRE(failed.wait_for(2s) == std::future_status::ready);
  CHECK_THROWS_AS(failed.get(), cucascade::io::object_changed_error);

  reads.second.complete(0x5a);
  CHECK(second.get() == read_size);
}

TEST_CASE("dropping a host read that owns its buffer waits for the read",
          "[datasource][gpu_execution]")
{
  held_reads reads;
  auto read = reads.source->host_read_async(0, read_size);
  reads.hold(reads.first);

  auto dropper = started([read = std::move(read)]() mutable { read = {}; });
  cancel_on_exit const cancel{reads};
  CHECK(dropper.wait_for(100ms) == std::future_status::timeout);

  reads.first.complete(0x5a);
  CHECK(dropper.wait_for(2s) == std::future_status::ready);
}
