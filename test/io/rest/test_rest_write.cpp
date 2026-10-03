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

// Whole-object uploads of the REST backend (single PUT / multipart upload)
// against an in-memory loopback object store.

#include "loopback_object_store.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::write_mode;
using cucascade::io::write_open_options;
using cucascade::io::write_segment;
using cucascade::io::rest::config;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::loopback_object_store;
using cucascade::test::object_store_authorizer;
using cucascade::test::object_store_faults;
using namespace std::chrono_literals;

constexpr std::size_t mib = 1UL << 20;
// The reactor clamps parts to the S3 minimum.
constexpr std::size_t part_size = 5 * mib;

std::vector<std::uint8_t> make_payload(std::size_t size, std::uint32_t seed = 1)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + (i >> 13) * 7U + seed * 17U) & 0xffU);
  }
  return bytes;
}

config write_config()
{
  config cfg{};
  cfg.request_timeout_s         = 10;
  cfg.tls_verify                = false;
  cfg.max_connections           = 8;
  cfg.max_retry_attempts        = 4;
  cfg.max_auth_retry_attempts   = 2;
  cfg.retry_backoff_base        = 1ms;
  cfg.retry_jitter              = 0ms;
  cfg.honor_retry_after         = false;
  cfg.write.part_size           = part_size;
  cfg.write.multipart_threshold = part_size;
  cfg.write.max_buffered_parts  = 4;
  return cfg;
}

std::shared_ptr<rest_ioctx> make_ioctx(
  loopback_object_store const& store,
  std::size_t n_runner_threads                                = 1,
  config cfg                                                  = write_config(),
  cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr)
{
  auto context = std::make_shared<rest_reactor::reactor_context>(
    cfg, std::make_shared<object_store_authorizer>(store.endpoint()), host_mr);
  return std::make_shared<rest_ioctx>(n_runner_threads, std::move(context));
}

bool wait_for(std::function<bool()> const& predicate, std::chrono::milliseconds limit = 10s)
{
  auto const until = std::chrono::steady_clock::now() + limit;
  while (std::chrono::steady_clock::now() < until) {
    if (predicate()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

bool stored_equals(loopback_object_store const& store,
                   std::string const& path,
                   std::vector<std::uint8_t> const& expected)
{
  auto const got = store.object(path);
  return got.has_value() && *got == expected;
}

write_segment host_segment(std::vector<std::uint8_t> const& payload,
                           std::size_t offset,
                           std::size_t size)
{
  return write_segment{cucascade::io::range{offset, size},
                       cucascade::io::host_source{payload.data() + offset}};
}

}  // namespace

TEST_CASE("rest write: a small object is uploaded with a single PUT", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store);
  ioctx->start();
  REQUIRE(ioctx->supports_write());
  REQUIRE(ioctx->supports_device_write());

  auto const payload = make_payload(3000);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/small.bin");
  CHECK(object->size() == 0);
  CHECK(ioctx->host_write_async(*object, 0, 1000, payload.data()).get() == 1000);
  // The synchronous path goes through a runner as well.
  CHECK(ioctx->host_write(*object, 1000, 2000, payload.data() + 1000) == 2000);
  CHECK(object->size() == payload.size());
  ioctx->flush_async(*object).get();
  CHECK(store.puts() == 0);  // nothing visible before commit

  ioctx->commit_async(*object).get();
  CHECK(store.puts() == 1);
  CHECK(store.initiates() == 0);
  CHECK(store.part_puts() == 0);
  CHECK(stored_equals(store, "bucket/small.bin", payload));

  // Read back through the same context.
  auto reader = ioctx->open_io_object("s3://bucket/small.bin");
  REQUIRE(reader->size() == payload.size());
  std::vector<std::uint8_t> back(payload.size());
  CHECK(ioctx->host_read_async(*reader, 0, back.size(), back.data()).get() == back.size());
  CHECK(back == payload);
  ioctx->shutdown();
}

TEST_CASE("rest write: an empty object commits as an empty PUT", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store);
  ioctx->start();
  auto object = ioctx->open_io_object_for_write("s3://bucket/empty.bin");
  ioctx->commit_async(*object).get();
  CHECK(store.puts() == 1);
  CHECK(stored_equals(store, "bucket/empty.bin", {}));
  ioctx->shutdown();
}

TEST_CASE("rest write: a multipart upload reassembles every part", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store, 2);
  ioctx->start();

  // 3 full parts + a short last part, written by requests of uneven sizes.
  auto const payload = make_payload(3 * part_size + 123457);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/multi.bin");
  std::vector<cucascade::exec::semi_future<std::size_t>> writes;
  std::size_t const cuts[] = {0, 1234567, 7 * mib + 11, payload.size()};
  for (std::size_t i = 0; i + 1 < std::size(cuts); ++i) {
    writes.push_back(
      ioctx->host_write_async(*object, cuts[i], cuts[i + 1] - cuts[i], payload.data() + cuts[i]));
  }
  for (std::size_t i = 0; i < writes.size(); ++i) {
    CHECK(std::move(writes[i]).get() == cuts[i + 1] - cuts[i]);
  }
  ioctx->commit_async(*object).get();

  CHECK(store.initiates() == 1);
  CHECK(store.part_puts() == 4);
  CHECK(store.completes() == 1);
  CHECK(store.puts() == 0);
  CHECK(store.live_uploads() == 0);
  CHECK(stored_equals(store, "bucket/multi.bin", payload));
  ioctx->shutdown();
}

TEST_CASE("rest write: out-of-order segments land in the right place", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store, 2);
  ioctx->start();

  auto const payload = make_payload(2 * part_size + 3 * mib + 7, 3);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/ooo.bin");

  // One vectored request with segments in reverse order (crossing part
  // boundaries) plus single writes issued back to front.
  std::size_t const split = payload.size() / 2;
  std::vector<write_segment> segments;
  std::size_t constexpr chunk = 700 * 1024 + 3;
  for (std::size_t end = split; end > 0;) {
    auto const begin = end > chunk ? end - chunk : 0;
    segments.push_back(host_segment(payload, begin, end - begin));
    end = begin;
  }
  auto vectored = ioctx->writev_async(*object, std::move(segments));

  std::vector<cucascade::exec::semi_future<std::size_t>> singles;
  for (std::size_t end = payload.size(); end > split;) {
    auto const begin = std::max(split, end > 2 * mib ? end - 2 * mib : 0);
    singles.push_back(ioctx->host_write_async(*object, begin, end - begin, payload.data() + begin));
    end = begin;
  }
  CHECK(std::move(vectored).get() == split);
  for (auto& write : singles) {
    CHECK(std::move(write).get() > 0);
  }
  ioctx->commit_async(*object).get();
  CHECK(store.completes() == 1);
  CHECK(stored_equals(store, "bucket/ooo.bin", payload));
  ioctx->shutdown();
}

TEST_CASE("rest write: a part upload answered 503 is retried", "[rest][write]")
{
  object_store_faults faults;
  faults.fail_first_part_puts = 2;
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(2 * part_size + 99, 5);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/retry.bin");
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  ioctx->commit_async(*object).get();
  CHECK(store.part_puts() == 3 + 2);
  CHECK(store.completes() == 1);
  CHECK(stored_equals(store, "bucket/retry.bin", payload));
  ioctx->shutdown();
}

TEST_CASE("rest write: CompleteMultipartUpload answering 200 with <Error> is retried",
          "[rest][write]")
{
  object_store_faults faults;
  faults.complete_error_body_first = 1;
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(part_size + 4096, 7);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/complete.bin");
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  ioctx->commit_async(*object).get();
  CHECK(store.completes() == 2);
  CHECK(stored_equals(store, "bucket/complete.bin", payload));
  ioctx->shutdown();
}

TEST_CASE("rest write: a retried Complete whose first response was lost succeeds", "[rest][write]")
{
  object_store_faults faults;
  faults.lose_first_complete_responses = 1;
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(part_size + 4096, 13);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/lost.bin");
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  auto const heads_before = store.gets();
  CHECK_NOTHROW(ioctx->commit_async(*object).get());
  CHECK(store.completes() == 2);            // first applied (response lost), retry -> NoSuchUpload
  CHECK(store.gets() == heads_before + 1);  // one verification HEAD
  CHECK(stored_equals(store, "bucket/lost.bin", payload));
  CHECK(store.live_uploads() == 0);
  CHECK(store.aborts() == 0);
  CHECK_THROWS_AS(ioctx->host_write(*object, 0, 1, payload.data()), std::invalid_argument);
  ioctx->shutdown();
}

TEST_CASE("rest write: a retried Complete fails when the object does not verify", "[rest][write]")
{
  object_store_faults faults;
  faults.lose_first_complete_responses = 1;
  faults.replace_after_lost_complete   = true;
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(part_size + 4096, 15);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/replaced.bin");
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  CHECK_THROWS_WITH(ioctx->commit_async(*object).get(),
                    Catch::Matchers::ContainsSubstring("NoSuchUpload"));
  CHECK(store.completes() == 2);
  ioctx->shutdown();
}

TEST_CASE("rest write: a failed part upload aborts the multipart upload", "[rest][write]")
{
  object_store_faults faults;
  faults.fail_all_part_puts = true;
  faults.part_fail_status   = 400;  // not retriable
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(2 * part_size + mib, 9);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/fail.bin");
  // The request completes parts 1 and 2, so it waits for their uploads.
  CHECK_THROWS_AS(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get(),
                  std::runtime_error);
  CHECK(wait_for([&] { return store.aborts() == 1; }));
  CHECK(store.live_uploads() == 0);
  CHECK_THROWS(ioctx->commit_async(*object).get());
  CHECK(store.completes() == 0);
  CHECK_FALSE(store.object("bucket/fail.bin").has_value());
  ioctx->shutdown();
  CHECK(store.aborts() == 1);  // the shutdown sweep does not abort twice
}

TEST_CASE("rest write: shutdown aborts an in-flight multipart upload", "[rest][write]")
{
  object_store_faults faults;
  faults.part_delay = 1500ms;
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(2 * part_size + 17, 11);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/shutdown.bin");
  auto write         = ioctx->host_write_async(*object, 0, payload.size(), payload.data());
  REQUIRE(wait_for([&] { return store.part_puts() >= 1; }));
  CHECK(store.initiates() == 1);
  CHECK(store.live_uploads() == 1);

  ioctx->shutdown();
  CHECK(store.aborts() == 1);
  CHECK(store.live_uploads() == 0);
  try {
    std::move(write).get();
    FAIL("write should have been cancelled");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::operation_canceled);
  }
  CHECK_FALSE(store.object("bucket/shutdown.bin").has_value());
}

TEST_CASE("rest write: dropping an uncommitted object aborts its multipart upload", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(part_size + 100, 23);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/orphan.bin");
  // Completes part 1: the multipart upload is created and part 1 uploaded.
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  REQUIRE(store.initiates() == 1);
  REQUIRE(store.live_uploads() == 1);

  object.reset();  // no commit, context still running
  CHECK(wait_for([&] { return store.aborts() == 1 && store.live_uploads() == 0; }));
  CHECK_FALSE(store.object("bucket/orphan.bin").has_value());
  ioctx->shutdown();
  CHECK(store.aborts() == 1);  // the shutdown sweep has nothing left to abort
}

TEST_CASE("rest write: an upload orphaned while no runner runs is aborted at shutdown",
          "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store, 1);  // never started: runners only through run_for

  auto const payload = make_payload(part_size + 100, 25);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/orphan2.bin");
  std::jthread runner([&] { static_cast<void>(ioctx->run_for(300ms)); });
  REQUIRE(wait_for([&] { return ioctx->active_runners() == 1; }));
  auto write = ioctx->host_write_async(*object, 0, payload.size(), payload.data());
  runner.join();
  REQUIRE(std::move(write).get() == payload.size());
  REQUIRE(store.live_uploads() == 1);

  object.reset();  // nobody runs: the abort waits in the reactor
  std::this_thread::sleep_for(20ms);
  CHECK(store.aborts() == 0);
  ioctx->shutdown();  // sweep aborts it synchronously
  CHECK(store.aborts() == 1);
  CHECK(store.live_uploads() == 0);
}

TEST_CASE("rest write: a session outliving its context only logs", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store);
  ioctx->start();
  auto const payload = make_payload(part_size + 100, 27);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/outlive.bin");
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  ioctx.reset();  // shutdown sweep aborts the live upload; the reactor is destroyed
  CHECK(store.aborts() == 1);
  object.reset();  // must not touch the destroyed reactor
  CHECK(store.aborts() == 1);
}

TEST_CASE("rest write: a retiring runner finishes the upload work it owns", "[rest][write]")
{
  object_store_faults faults;
  faults.part_delay = 200ms;
  loopback_object_store store(faults);
  auto ioctx = make_ioctx(store, 1);  // start() is only called for the commit

  auto const payload = make_payload(2 * part_size + 5, 21);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/retire.bin");
  std::jthread runner([&] { static_cast<void>(ioctx->run_for(50ms)); });
  REQUIRE(wait_for([&] { return ioctx->active_runners() == 1; }));
  auto write = ioctx->host_write_async(*object, 0, payload.size(), payload.data());
  runner.join();
  // Write groups are drained, not requeued: the parts the request completed
  // were uploaded before run_for returned.
  CHECK(write.is_ready());
  CHECK(std::move(write).get() == payload.size());
  CHECK(store.part_puts() == 2);

  ioctx->start();
  ioctx->commit_async(*object).get();
  CHECK(stored_equals(store, "bucket/retire.bin", payload));
  ioctx->shutdown();
}

TEST_CASE("rest write: writes and commits after a commit are rejected", "[rest][write]")
{
  loopback_object_store store;
  auto ioctx = make_ioctx(store);
  ioctx->start();

  auto const payload = make_payload(1000, 13);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/done.bin");
  CHECK(ioctx->host_write_async(*object, 0, payload.size(), payload.data()).get() ==
        payload.size());
  ioctx->commit_async(*object).get();
  CHECK_THROWS_AS(ioctx->host_write_async(*object, 1000, 10, payload.data()).get(),
                  std::invalid_argument);
  CHECK_THROWS_AS(ioctx->commit_async(*object).get(), std::invalid_argument);
  CHECK(store.puts() == 1);

  // A read-only object (opened for read) refuses writes too.
  auto reader = ioctx->open_io_object("s3://bucket/done.bin");
  CHECK_THROWS_AS(ioctx->host_write_async(*reader, 0, 10, payload.data()).get(),
                  std::invalid_argument);

  // A commit with a hole fails (and the object is never created).
  auto holey = ioctx->open_io_object_for_write("s3://bucket/holey.bin");
  CHECK(ioctx->host_write_async(*holey, 100, 10, payload.data()).get() == 10);
  CHECK_THROWS_AS(ioctx->commit_async(*holey).get(), std::invalid_argument);
  CHECK_FALSE(store.object("bucket/holey.bin").has_value());
  ioctx->shutdown();
}

TEST_CASE("rest write: opening an existing object for update is not supported", "[rest][write]")
{
  loopback_object_store store;
  store.put_object("bucket/existing.bin", make_payload(10));
  auto ioctx = make_ioctx(store);

  for (auto const mode : {write_mode::create_or_open, write_mode::open_existing}) {
    write_open_options options;
    options.mode = mode;
    try {
      static_cast<void>(ioctx->open_io_object_for_write("s3://bucket/existing.bin", options));
      FAIL("open for update should fail");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::not_supported);
    }
  }
  CHECK_THROWS(ioctx->open_io_object_for_write("file:///tmp/x.bin"));
}

TEST_CASE("rest write: reads are not starved by a large upload", "[rest][write]")
{
  object_store_faults faults;
  faults.part_delay = 400ms;
  loopback_object_store store(faults);
  auto const readable = make_payload(1 * mib, 15);
  store.put_object("bucket/read.bin", readable);
  auto ioctx = make_ioctx(store, 1);
  ioctx->start();
  auto reader = ioctx->open_io_object("s3://bucket/read.bin");

  // 8 parts at 400 ms each, at most 4 in flight: the upload takes > 800 ms.
  auto const payload = make_payload(8 * part_size, 17);
  auto object        = ioctx->open_io_object_for_write("s3://bucket/big.bin");
  auto write         = ioctx->host_write_async(*object, 0, payload.size(), payload.data());
  REQUIRE(wait_for([&] { return store.part_puts() >= 2; }));

  std::chrono::nanoseconds worst{0};
  for (int i = 0; i < 8; ++i) {
    std::vector<std::uint8_t> buffer(64 * 1024);
    auto const offset = static_cast<std::size_t>(i) * 100'000;
    auto const t0     = std::chrono::steady_clock::now();
    CHECK(ioctx->host_read_async(*reader, offset, buffer.size(), buffer.data()).get() ==
          buffer.size());
    worst = std::max(worst, std::chrono::steady_clock::now() - t0);
    CHECK(std::equal(
      buffer.begin(), buffer.end(), readable.begin() + static_cast<std::ptrdiff_t>(offset)));
  }
  // Every read was served while parts were still uploading.
  CHECK(worst < 300ms);
  CHECK_FALSE(write.is_ready());

  CHECK(std::move(write).get() == payload.size());
  ioctx->commit_async(*object).get();
  CHECK(stored_equals(store, "bucket/big.bin", payload));
  ioctx->shutdown();
}

TEST_CASE("rest write: device sources are staged through pinned memory", "[rest][write][gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { SKIP("no CUDA device"); }

  rmm::mr::pinned_host_memory_resource pinned_mr;
  cucascade::memory::fixed_size_host_memory_resource host_mr{
    0, pinned_mr, 64UL << 20, 64UL << 20, 1UL << 20, 16, 1};

  loopback_object_store store;
  auto ioctx = make_ioctx(store, 1, write_config(), &host_mr);
  ioctx->start();

  auto const payload             = make_payload(2 * part_size + 777, 19);
  std::size_t const device_bytes = part_size + 3 * mib + 5;  // rest comes from host memory
  void* device                   = nullptr;
  REQUIRE(cudaMalloc(&device, device_bytes) == cudaSuccess);
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  // Not synchronized: the upload must be ordered after this copy on the stream.
  REQUIRE(cudaMemcpyAsync(device, payload.data(), device_bytes, cudaMemcpyHostToDevice, stream) ==
          cudaSuccess);

  auto object = ioctx->open_io_object_for_write("s3://bucket/device.bin");
  std::vector<write_segment> segments;
  segments.push_back(host_segment(payload, device_bytes, payload.size() - device_bytes));
  segments.push_back(
    write_segment{cucascade::io::range{0, device_bytes},
                  cucascade::io::device_source{
                    static_cast<std::uint8_t const*>(device), ::cuda::stream_ref{stream}, -1}});
  CHECK(ioctx->writev_async(*object, std::move(segments)).get() == payload.size());
  ioctx->commit_async(*object).get();
  CHECK(store.completes() == 1);
  CHECK(stored_equals(store, "bucket/device.bin", payload));

  ioctx->shutdown();
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  CHECK(cudaFree(device) == cudaSuccess);
}
