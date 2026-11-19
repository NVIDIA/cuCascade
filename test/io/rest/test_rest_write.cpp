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

// End-to-end S3 writes through rest_ioctx (docs/io/rest-transport.md "Writes"): the public
// write API (open_io_object_for_write, host / device writes, flush) against the loopback
// server's in-memory object store, signed by the SigV4 authorizers.

#include "io/rest/reactor_test_hooks.hpp"
#include "io/rest/write/s3_call.hpp"
#include "io/rest/write/upload.hpp"
#include "rest_contract_test_utils.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::write_mode;
using cucascade::io::write_slice;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::listed_object;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::require_rest_bytes_equal;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_payload;
using cucascade::test::s3_write_policy;
using cucascade::test::write_fault;
using cucascade::test::write_kind;
using namespace std::chrono_literals;

constexpr std::size_t MiB        = std::size_t{1} << 20;
constexpr std::size_t part_bytes = 5 * MiB;  // S3's minimum part size

cucascade::io::rest::config write_config()
{
  auto cfg               = rest_contract_config();
  cfg.write_split_bytes  = part_bytes;
  cfg.max_retry_attempts = 3;
  return cfg;
}

enum class signing : std::uint8_t { presigned, header };

std::shared_ptr<cucascade::io::rest::request_authorizer> make_authorizer(
  loopback_range_server const& server, signing how)
{
  cucascade::io::rest::s3::static_credentials creds;
  creds.access_key_id     = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  if (how == signing::header) {
    return std::make_shared<cucascade::io::rest::s3::sigv4_header_authorizer>(
      creds, "us-east-1", server.endpoint());
  }
  return std::make_shared<cucascade::io::rest::s3::sigv4_presigned_authorizer>(
    creds, "us-east-1", server.endpoint());
}

std::shared_ptr<rest_ioctx> make_write_ioctx(
  loopback_range_server const& server,
  cucascade::io::rest::config cfg                          = write_config(),
  std::size_t reactors                                     = 1,
  signing how                                              = signing::presigned,
  cucascade::memory::fixed_size_host_memory_resource* host = nullptr)
{
  auto context =
    std::make_shared<rest_reactor::reactor_context>(cfg, make_authorizer(server, how), host);
  auto ioctx = std::make_shared<rest_ioctx>(reactors, std::move(context));
  ioctx->start();
  return ioctx;
}

/// A loopback object store: written keys are what GET / HEAD see; other keys are 404.
std::unique_ptr<loopback_range_server> make_store(s3_write_policy policy = {})
{
  policy.missing_key_404 = true;
  return std::make_unique<loopback_range_server>(
    rest_test_payload(4096),
    range_fault_policy{},
    std::vector<listed_object>{},
    std::unordered_map<std::string, cucascade::test::key_response_script>{},
    std::move(policy));
}

std::string uri(std::string_view key) { return "s3://bkt/" + std::string{key}; }

/// The object's bytes as a NEW read-only io_object of @p ioctx sees them (HEAD + ranged GETs).
std::vector<std::uint8_t> read_back(rest_ioctx& ioctx, std::string_view key)
{
  auto object = ioctx.open_io_object(uri(key));
  std::vector<std::uint8_t> bytes(object->size());
  if (!bytes.empty()) {
    REQUIRE(ioctx.host_read(*object, 0, bytes.size(), bytes.data()) == bytes.size());
  }
  return bytes;
}

void require_stored(loopback_range_server const& server,
                    std::string_view key,
                    std::vector<std::uint8_t> const& expected)
{
  auto const stored = server.stored_object(key);
  REQUIRE(stored.has_value());
  require_rest_bytes_equal(*stored, expected);
}

std::size_t count_requests(loopback_range_server const& server, write_kind kind)
{
  auto const records = server.write_requests();
  return static_cast<std::size_t>(std::count_if(
    records.begin(), records.end(), [kind](auto const& r) { return r.kind == kind; }));
}

/// Wait until @p done() holds or @p timeout passed. @return done().
template <class Done>
bool eventually(Done&& done, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() >= deadline) { return done(); }
    std::this_thread::sleep_for(2ms);
  }
  return true;
}

std::error_code error_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::system_error const& e) {
    return e.code();
  } catch (...) {
    return {};
  }
}

template <class Fn>
std::exception_ptr failure_of(Fn&& fn)
{
  try {
    fn();
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// 64 MiB of pinned 1 MiB blocks: the staging resource the part buffers come from.
class pinned_staging {
 public:
  pinned_staging() : _mr{0, _upstream, 64 * MiB, 64 * MiB, MiB, 64, 0} {}

  [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource* get() noexcept { return &_mr; }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream{0};
  cucascade::memory::fixed_size_host_memory_resource _mr;
};

}  // namespace

TEST_CASE("rest_ioctx writes a small object with one PUT at flush", "[rest][write]")
{
  auto how    = GENERATE(signing::presigned, signing::header);
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 1, how);
  rest_shutdown_guard cleanup{*ioctx};
  REQUIRE(ioctx->supports_write());
  REQUIRE(ioctx->supports_device_write());

  auto const data = rest_test_payload(100 * 1024);
  auto object     = ioctx->open_io_object_for_write(uri("small.bin"));
  REQUIRE(object->writable());
  CHECK(server->write_requests().empty());  // no request at open

  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  CHECK(object->size() == data.size());
  // An object store makes the bytes visible at the commit only.
  CHECK_FALSE(server->stored_object("small.bin").has_value());
  CHECK(server->write_requests().empty());
  CHECK(object->validation_tag().empty());

  ioctx->flush(*object);
  CHECK(server->put_object_count() == 1);
  CHECK(server->create_upload_count() == 0);
  CHECK(server->upload_part_count() == 0);
  require_stored(*server, "small.bin", data);
  CHECK(object->validation_tag() == server->stored_etag("small.bin"));
  auto const records = server->write_requests();
  REQUIRE(records.size() == 1);
  CHECK(records.front().presigned == (how == signing::presigned));
  CHECK(records.front().header_authorized == (how == signing::header));
  CHECK_FALSE(records.front().if_none_match);
  if (how == signing::header) { CHECK(records.front().content_sha256 == "UNSIGNED-PAYLOAD"); }

  require_rest_bytes_equal(read_back(*ioctx, "small.bin"), data);
  // The written object itself reads the committed bytes (If-Match: its new ETag).
  std::vector<std::uint8_t> through_writer(data.size());
  REQUIRE(ioctx->host_read(*object, 0, through_writer.size(), through_writer.data()) ==
          data.size());
  require_rest_bytes_equal(through_writer, data);

  // Committed: a second flush sends nothing, a further write fails.
  ioctx->flush(*object);
  CHECK(server->write_requests().size() == 1);
  auto const late = failure_of([&] { ioctx->host_write(*object, 0, 1, data.data()); });
  REQUIRE(late != nullptr);
  CHECK(error_of(late) == std::errc::bad_file_descriptor);
}

TEST_CASE("rest_ioctx flushes an object it never wrote as an empty PUT", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object_for_write(uri("empty.bin"));
  ioctx->flush(*object);
  CHECK(server->put_object_count() == 1);
  auto const stored = server->stored_object("empty.bin");
  REQUIRE(stored.has_value());
  CHECK(stored->empty());
  CHECK(object->size() == 0);
}

TEST_CASE("rest_ioctx writes a multi-part object", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  // Two whole parts (uploaded straight from the caller's buffer while writing) and a 2 MiB tail
  // (assembled; uploaded by the flush).
  auto const data = rest_test_payload(2 * part_bytes + 2 * MiB);
  auto object     = ioctx->open_io_object_for_write(uri("multi/part object.bin"));

  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  CHECK(server->create_upload_count() == 1);
  CHECK(server->upload_part_count() == 2);
  CHECK(server->active_upload_count() == 1);
  CHECK_FALSE(server->stored_object("multi/part object.bin").has_value());

  ioctx->flush(*object);
  CHECK(server->upload_part_count() == 3);
  CHECK(server->complete_upload_count() == 1);
  CHECK(server->put_object_count() == 0);
  CHECK(server->active_upload_count() == 0);
  require_stored(*server, "multi/part object.bin", data);
  auto const etag = server->stored_etag("multi/part object.bin");
  CHECK(etag.find("-3\"") != std::string::npos);
  CHECK(object->validation_tag() == etag);
  require_rest_bytes_equal(read_back(*ioctx, "multi/part object.bin"), data);
}

TEST_CASE("rest_ioctx assembles partial and out-of-order writes into parts", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 2);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 3 * MiB);
  auto object     = ioctx->open_io_object_for_write(uri("assembled.bin"));

  // Part 1 in four out-of-order pieces, all in flight together: the write that completes the
  // part's coverage uploads it.
  std::size_t const q = part_bytes / 4;
  std::vector<cucascade::exec::semi_future<std::size_t>> writes;
  for (std::size_t const piece : {3UL, 1UL, 0UL, 2UL}) {
    writes.push_back(ioctx->host_write_async(*object, piece * q, q, data.data() + piece * q));
  }
  for (auto& w : writes) {
    REQUIRE(std::move(w).get(10s) == q);
  }
  CHECK(server->create_upload_count() == 1);
  CHECK(server->upload_part_count() == 1);

  // Part 2 in two vectored pieces, back to front.
  std::vector<write_slice> const tail{
    write_slice{part_bytes + 2 * MiB, MiB, data.data() + part_bytes + 2 * MiB},
    write_slice{part_bytes, 2 * MiB, data.data() + part_bytes}};
  REQUIRE(std::move(ioctx->host_writev_async(*object, tail)).get(10s) == 3 * MiB);
  CHECK(server->upload_part_count() == 1);  // part 2 is not full: it waits for the flush

  ioctx->flush(*object);
  CHECK(server->upload_part_count() == 2);
  require_stored(*server, "assembled.bin", data);
}

TEST_CASE("rest_ioctx accepts a write slice that crosses a part boundary", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + MiB);
  auto object     = ioctx->open_io_object_for_write(uri("straddle.bin"));

  // Smaller than a part, across the part 1 / part 2 boundary, asynchronous and synchronous.
  std::size_t const first = part_bytes - 4096;
  REQUIRE(std::move(ioctx->host_write_async(*object, 0, first, data.data())).get(10s) == first);
  REQUIRE(std::move(ioctx->host_write_async(*object, first, 8192, data.data() + first)).get(10s) ==
          8192);
  std::size_t const rest = data.size() - first - 8192;
  REQUIRE(ioctx->host_write(*object, first + 8192, rest, data.data() + first + 8192) == rest);
  ioctx->flush(*object);
  require_stored(*server, "straddle.bin", data);
}

TEST_CASE("rest_ioctx commits holes as zeros", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(4096);

  SECTION("a gap inside a single-PUT object")
  {
    auto object = ioctx->open_io_object_for_write(uri("hole.bin"));
    REQUIRE(ioctx->host_write(*object, 1000, 500, data.data()) == 500);
    ioctx->flush(*object);
    std::vector<std::uint8_t> expected(1500, 0);
    std::copy(data.begin(), data.begin() + 500, expected.begin() + 1000);
    require_stored(*server, "hole.bin", expected);
    CHECK(server->put_object_count() == 1);
  }

  SECTION("gaps inside parts and a part never written")
  {
    // Part 1: 4 KiB at its start; part 2: never written; part 3: 4 KiB in its middle.
    auto object             = ioctx->open_io_object_for_write(uri("sparse.bin"));
    std::size_t const third = 2 * part_bytes + MiB;
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    REQUIRE(ioctx->host_write(*object, third, data.size(), data.data()) == data.size());
    CHECK(object->size() == third + data.size());
    ioctx->flush(*object);
    std::vector<std::uint8_t> expected(third + data.size(), 0);
    std::copy(data.begin(), data.end(), expected.begin());
    std::copy(data.begin(), data.end(), expected.begin() + static_cast<std::ptrdiff_t>(third));
    require_stored(*server, "sparse.bin", expected);
    CHECK(server->upload_part_count() == 3);
    CHECK(server->complete_upload_count() == 1);
    require_rest_bytes_equal(read_back(*ioctx, "sparse.bin"), expected);
  }
}

TEST_CASE("rest_ioctx create_new commits only when the key does not exist", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const old_data = rest_test_payload(64);
  auto const data     = rest_test_payload(part_bytes + 4096);
  server->put_stored_object("taken.bin", old_data);

  bool const multipart = GENERATE(false, true);
  auto const size      = multipart ? data.size() : std::size_t{4096};
  CAPTURE(multipart);

  auto taken = ioctx->open_io_object_for_write(uri("taken.bin"), {.mode = write_mode::create_new});
  REQUIRE(ioctx->host_write(*taken, 0, size, data.data()) == size);
  auto const refused = failure_of([&] { ioctx->flush(*taken); });
  REQUIRE(refused != nullptr);
  CHECK(error_of(refused) == std::errc::file_exists);
  require_stored(*server, "taken.bin", old_data);  // untouched
  CHECK(taken->validation_tag().empty());

  auto fresh = ioctx->open_io_object_for_write(uri("fresh.bin"), {.mode = write_mode::create_new});
  REQUIRE(ioctx->host_write(*fresh, 0, size, data.data()) == size);
  ioctx->flush(*fresh);
  require_stored(
    *server, "fresh.bin", {data.begin(), data.begin() + static_cast<std::ptrdiff_t>(size)});
  auto const commit_kind = multipart ? write_kind::complete_upload : write_kind::put_object;
  for (auto const& r : server->write_requests()) {
    if (r.kind == commit_kind) { CHECK(r.if_none_match); }
    if (r.kind == write_kind::upload_part) { CHECK_FALSE(r.if_none_match); }
  }
}

TEST_CASE("rest_ioctx refuses the open modes that keep an object's content", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  for (auto const mode : {write_mode::create_or_open, write_mode::open_existing}) {
    auto const error = failure_of(
      [&] { std::ignore = ioctx->open_io_object_for_write(uri("k.bin"), {.mode = mode}); });
    REQUIRE(error != nullptr);
    CHECK(error_of(error) == std::errc::not_supported);
  }
  // A read-only object is not a write target.
  server->put_stored_object("ro.bin", rest_test_payload(16));
  auto read_only     = ioctx->open_io_object(uri("ro.bin"));
  std::uint8_t byte  = 0;
  auto const refused = failure_of([&] { ioctx->host_write(*read_only, 0, 1, &byte); });
  REQUIRE(refused != nullptr);
  CHECK(error_of(refused) == std::errc::bad_file_descriptor);
  CHECK(server->write_requests().empty());
}

TEST_CASE("rest_ioctx writes device memory", "[rest][write][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  bool const pinned = GENERATE(false, true);
  CAPTURE(pinned);
  pinned_staging staging;
  auto server = make_store();
  auto ioctx  = make_write_ioctx(
    *server, write_config(), 2, signing::presigned, pinned ? staging.get() : nullptr);
  rest_shutdown_guard cleanup{*ioctx};

  auto const data = rest_test_payload(2 * part_bytes + 3 * MiB);
  rmm::cuda_stream stream;
  rmm::device_buffer source(data.size(), stream);
  REQUIRE(cudaMemcpyAsync(
            source.data(), data.data(), data.size(), cudaMemcpyHostToDevice, stream.value()) ==
          cudaSuccess);
  auto const* src = static_cast<std::uint8_t const*>(source.data());

  auto object = ioctx->open_io_object_for_write(uri("device.bin"));
  // A whole part, a part in two device pieces, and a tail mixing host and device segments.
  REQUIRE(std::move(ioctx->device_write_async(*object, 0, part_bytes, src, stream)).get(10s) ==
          part_bytes);
  std::vector<write_slice> const second{
    write_slice{part_bytes + MiB, part_bytes - MiB, src + part_bytes + MiB},
    write_slice{part_bytes, MiB, src + part_bytes}};
  REQUIRE(std::move(ioctx->device_writev_async(*object, second, stream)).get(10s) == part_bytes);
  std::size_t const tail = 2 * part_bytes;
  std::vector<write_slice> const host_tail{write_slice{tail, MiB, data.data() + tail}};
  std::vector<write_slice> const device_tail{write_slice{tail + MiB, 2 * MiB, src + tail + MiB}};
  REQUIRE(std::move(ioctx->mixed_writev_async(*object, host_tail, device_tail, stream)).get(10s) ==
          3 * MiB);
  ioctx->flush(*object);
  require_stored(*server, "device.bin", data);
  CHECK(server->upload_part_count() == 3);
}

TEST_CASE("rest_ioctx writes device memory of a GPU other than the worker's current one",
          "[rest][write][gpu][multi_gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
    std::ignore = cudaGetLastError();
    SKIP("requires two CUDA devices");
  }
  // The D2H copies into the part buffers run on device 1's legacy default stream while the
  // worker's current device is 0: the copy and its event must both target device 1.
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 3 * MiB);

  int previous = 0;
  REQUIRE(cudaGetDevice(&previous) == cudaSuccess);
  REQUIRE(cudaSetDevice(1) == cudaSuccess);
  void* source = nullptr;
  REQUIRE(cudaMalloc(&source, data.size()) == cudaSuccess);
  REQUIRE(cudaMemcpy(source, data.data(), data.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  auto object = ioctx->open_io_object_for_write(uri("gpu1.bin"));
  auto future = ioctx->device_write_async(*object,
                                          0,
                                          data.size(),
                                          static_cast<std::uint8_t const*>(source),
                                          ::cuda::stream_ref{cudaStreamLegacy});
  REQUIRE(cudaSetDevice(previous) == cudaSuccess);
  REQUIRE(std::move(future).get(20s) == data.size());
  ioctx->flush(*object);
  require_stored(*server, "gpu1.bin", data);
  REQUIRE(cudaSetDevice(1) == cudaSuccess);
  REQUIRE(cudaFree(source) == cudaSuccess);
  REQUIRE(cudaSetDevice(previous) == cudaSuccess);
}

TEST_CASE("rest_ioctx shutdown settles writes whose device copies are in flight",
          "[rest][write][gpu][shutdown]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  // The caller's stream is held, so the D2H copies of the two whole parts are still queued when
  // the ioctx shuts down.  Shutdown waits for them (no DMA outlives its part buffer), then settles
  // the write as cancelled instead of uploading: no request reaches the server, the future
  // resolves, and every part buffer returns to the pinned staging.
  pinned_staging staging;
  auto server     = make_store();
  auto ioctx      = make_write_ioctx(*server, write_config(), 1, signing::presigned, staging.get());
  auto const data = rest_test_payload(2 * part_bytes);
  rmm::cuda_stream stream;
  rmm::device_buffer source(data.size(), stream);
  REQUIRE(cudaMemcpyAsync(
            source.data(), data.data(), data.size(), cudaMemcpyHostToDevice, stream.value()) ==
          cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  auto* const hold = new std::chrono::milliseconds{500};
  REQUIRE(cudaLaunchHostFunc(
            stream.value(),
            [](void* arg) {
              std::unique_ptr<std::chrono::milliseconds> const time{
                static_cast<std::chrono::milliseconds*>(arg)};
              std::this_thread::sleep_for(*time);
            },
            hold) == cudaSuccess);

  auto object = ioctx->open_io_object_for_write(uri("shutdown.bin"));
  auto future = ioctx->device_write_async(
    *object, 0, data.size(), static_cast<std::uint8_t const*>(source.data()), stream);
  // Both parts' copies are tracked once their part buffers left the staging resource.
  REQUIRE(
    eventually([&] { return staging.get()->get_available_memory() <= 64 * MiB - 2 * part_bytes; }));
  ioctx->shutdown();
  // get(timeout) throws a timeout when the write never settles.
  auto const error = failure_of([&] { std::ignore = std::move(future).get(10s); });
  REQUIRE(error != nullptr);
  CHECK(error_of(error) == std::errc::operation_canceled);
  CHECK(server->write_requests().empty());
  object.reset();
  CHECK(eventually([&] { return staging.get()->get_available_memory() == 64 * MiB; }));
}

TEST_CASE("rest_ioctx aborts an upload that is destroyed without a flush", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 4096);

  SECTION("with an UploadId: AbortMultipartUpload")
  {
    {
      auto object = ioctx->open_io_object_for_write(uri("abandoned.bin"));
      REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
      CHECK(server->upload_part_count() == 1);
      CHECK(server->active_upload_count() == 1);
    }
    CHECK(eventually([&] { return server->abort_upload_count() == 1; }));
    CHECK(server->active_upload_count() == 0);
    CHECK_FALSE(server->stored_object("abandoned.bin").has_value());
  }

  SECTION("without an UploadId: no request at all")
  {
    {
      auto object = ioctx->open_io_object_for_write(uri("dropped.bin"));
      REQUIRE(ioctx->host_write(*object, 0, 4096, data.data()) == 4096);
    }
    // A barrier instead of a sleep: an abort would be posted to the first reactor's inbox at the
    // destruction above, and that reactor handles its inbox before the later commit below (a
    // flush runs there too), so once that commit answered, any abort was already sent.
    auto probe = ioctx->open_io_object_for_write(uri("barrier.bin"));
    REQUIRE(ioctx->host_write(*probe, 0, 16, data.data()) == 16);
    ioctx->flush(*probe);
    auto const records = server->write_requests();
    REQUIRE(records.size() == 1);
    CHECK(records.front().kind == write_kind::put_object);
  }
}

TEST_CASE("rest_ioctx aborts an upload whose commit failed once the object is destroyed",
          "[rest][write]")
{
  // The flush job of a failed commit must not outlive it: it holds the upload session, whose
  // destruction is what aborts an upload that was never committed.
  s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::complete_upload, .count = 100, .status = 400, .code = "InvalidRequest"});
  auto server = make_store(policy);
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 4096);
  {
    auto object = ioctx->open_io_object_for_write(uri("failed-commit.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    CHECK_THROWS(ioctx->flush(*object));
    CHECK(server->active_upload_count() == 1);
  }
  CHECK(eventually([&] { return server->abort_upload_count() == 1; }));
  CHECK(server->active_upload_count() == 0);
  CHECK_FALSE(server->stored_object("failed-commit.bin").has_value());

  // Nothing the commit used keeps a committed object alive either.
  std::weak_ptr<cucascade::io::io_object> committed;
  {
    auto object = ioctx->open_io_object_for_write(uri("committed.bin"));
    REQUIRE(ioctx->host_write(*object, 0, 4096, data.data()) == 4096);
    ioctx->flush(*object);
    committed = object;
  }
  CHECK(eventually([&] { return committed.expired(); }));
}

TEST_CASE("rest_ioctx retries transient write failures and surfaces final ones", "[rest][write]")
{
  auto const data = rest_test_payload(part_bytes + 4096);

  SECTION("a 503 on UploadPart is retried")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind = write_kind::upload_part, .nth = 0, .count = 2});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("retried.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    ioctx->flush(*object);
    CHECK(server->upload_part_count() == 4);  // 2 failed attempts + 2 parts
    require_stored(*server, "retried.bin", data);
  }

  SECTION("a dropped connection mid-body is retried")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::upload_part, .nth = 0, .what = write_fault::action::drop_mid_body});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("dropped-body.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    ioctx->flush(*object);
    require_stored(*server, "dropped-body.bin", data);
  }

  SECTION("a 200 carrying an S3 error on Complete is retried")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind = write_kind::complete_upload,
                                        .nth  = 0,
                                        .what = write_fault::action::ok_with_error,
                                        .code = "InternalError"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("complete-retry.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    ioctx->flush(*object);
    CHECK(server->complete_upload_count() == 2);
    require_stored(*server, "complete-retry.bin", data);
  }

  SECTION("a 4xx on the single PUT fails the flush; flushing again commits")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::put_object, .nth = 0, .status = 400, .code = "InvalidRequest"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("put-4xx.bin"));
    REQUIRE(ioctx->host_write(*object, 0, 4096, data.data()) == 4096);
    CHECK_THROWS(ioctx->flush(*object));
    CHECK(server->put_object_count() == 1);  // final: not retried
    CHECK_FALSE(server->stored_object("put-4xx.bin").has_value());
    ioctx->flush(*object);
    CHECK(server->put_object_count() == 2);
    require_stored(*server, "put-4xx.bin", {data.begin(), data.begin() + 4096});
  }

  SECTION("a 403 on Complete is final; flushing again commits")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::complete_upload, .nth = 0, .status = 403, .code = "AccessDenied"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("complete-403.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    CHECK_THROWS(ioctx->flush(*object));
    CHECK(server->complete_upload_count() == 1);
    CHECK(server->active_upload_count() == 1);
    ioctx->flush(*object);
    CHECK(server->complete_upload_count() == 2);
    require_stored(*server, "complete-403.bin", data);
  }

  SECTION("a 200 Complete without a result document fails the flush; flushing again commits")
  {
    // A proxy answering in S3's place: the commit was never acknowledged, so it must not read as
    // committed (with an empty tag).
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind = write_kind::complete_upload,
                                        .nth  = 0,
                                        .what = write_fault::action::ok_with_body,
                                        .body = "<html>proxy failure</html>"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("complete-proxy.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    CHECK_THROWS(ioctx->flush(*object));
    CHECK(server->complete_upload_count() == 1);
    CHECK(server->active_upload_count() == 1);
    CHECK_FALSE(server->stored_object("complete-proxy.bin").has_value());
    ioctx->flush(*object);
    CHECK(server->complete_upload_count() == 2);
    require_stored(*server, "complete-proxy.bin", data);
  }

  SECTION("a final error on a write-started part upload poisons the object")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::upload_part, .nth = 0, .status = 400, .code = "InvalidArgument"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("poisoned.bin"));
    CHECK_THROWS(ioctx->host_write(*object, 0, part_bytes, data.data()));
    CHECK_THROWS(ioctx->host_write(*object, part_bytes, 4096, data.data() + part_bytes));
    CHECK_THROWS(ioctx->flush(*object));
    CHECK(server->complete_upload_count() == 0);
    CHECK_FALSE(server->stored_object("poisoned.bin").has_value());
  }
}

TEST_CASE("rest_ioctx writes concurrently from several threads on several reactors",
          "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 3);
  rest_shutdown_guard cleanup{*ioctx};

  SECTION("distinct objects")
  {
    constexpr std::size_t n_objects = 4;
    std::vector<std::vector<std::uint8_t>> payloads;
    for (std::size_t i = 0; i < n_objects; ++i) {
      auto bytes = rest_test_payload(part_bytes + (i + 1) * 333 * 1024);
      for (auto& b : bytes) {
        b = static_cast<std::uint8_t>(b ^ (i * 37U));
      }
      payloads.push_back(std::move(bytes));
    }
    std::vector<std::thread> writers;
    std::vector<std::exception_ptr> errors(n_objects);
    for (std::size_t i = 0; i < n_objects; ++i) {
      writers.emplace_back([&, i] {
        errors[i] = failure_of([&] {
          auto object       = ioctx->open_io_object_for_write(uri("obj-" + std::to_string(i)));
          auto const& bytes = payloads[i];
          // 1 MiB slices in flight together, then the commit.
          std::vector<cucascade::exec::semi_future<std::size_t>> writes;
          for (std::size_t at = 0; at < bytes.size(); at += MiB) {
            auto const n = std::min(MiB, bytes.size() - at);
            writes.push_back(ioctx->host_write_async(*object, at, n, bytes.data() + at));
          }
          for (auto& w : writes) {
            std::ignore = std::move(w).get(20s);
          }
          ioctx->flush(*object);
        });
      });
    }
    for (auto& t : writers) {
      t.join();
    }
    for (std::size_t i = 0; i < n_objects; ++i) {
      INFO("object " << i);
      CHECK(errors[i] == nullptr);
      require_stored(*server, "obj-" + std::to_string(i), payloads[i]);
    }
  }

  SECTION("one object written by every thread")
  {
    constexpr std::size_t n_threads = 4;
    auto const data                 = rest_test_payload(3 * part_bytes + 777);
    auto object                     = ioctx->open_io_object_for_write(uri("shared.bin"));
    std::size_t const stride        = 256 * 1024;
    std::vector<std::thread> writers;
    std::vector<std::exception_ptr> errors(n_threads);
    for (std::size_t t = 0; t < n_threads; ++t) {
      writers.emplace_back([&, t] {
        errors[t] = failure_of([&] {
          std::vector<cucascade::exec::semi_future<std::size_t>> writes;
          for (std::size_t at = t * stride; at < data.size(); at += n_threads * stride) {
            auto const n = std::min(stride, data.size() - at);
            writes.push_back(ioctx->host_write_async(*object, at, n, data.data() + at));
          }
          for (auto& w : writes) {
            std::ignore = std::move(w).get(20s);
          }
        });
      });
    }
    for (auto& t : writers) {
      t.join();
    }
    for (auto const& e : errors) {
      CHECK(e == nullptr);
    }
    CHECK(object->size() == data.size());
    ioctx->flush(*object);
    require_stored(*server, "shared.bin", data);
    CHECK(server->upload_part_count() == 4);
    require_rest_bytes_equal(read_back(*ioctx, "shared.bin"), data);
  }
}

namespace {

/// The S3 calls a reactor started and finished, as the worker reported them.
struct call_event {
  std::thread::id thread;
  bool started{false};
  std::string kind;
  std::string key;
  std::size_t attempt{0};
  unsigned status{0};
  std::uint64_t waits{0};
};

/// Installs reactor events that record every S3 call (remove before the ioctx goes).
class call_recorder {
 public:
  call_recorder()
  {
    _events.on_call_started = [this](std::string_view kind,
                                     std::string_view key,
                                     std::size_t attempt,
                                     std::uint64_t waits) {
      std::lock_guard lock{_mutex};
      _log.push_back(call_event{
        std::this_thread::get_id(), true, std::string{kind}, std::string{key}, attempt, 0, waits});
    };
    _events.on_call_finished = [this](std::string_view kind,
                                      std::string_view key,
                                      std::size_t attempt,
                                      unsigned status,
                                      std::uint64_t waits) {
      std::lock_guard lock{_mutex};
      _log.push_back(call_event{std::this_thread::get_id(),
                                false,
                                std::string{kind},
                                std::string{key},
                                attempt,
                                status,
                                waits});
    };
    cucascade::io::rest::testing::set_reactor_events(&_events);
  }
  ~call_recorder() { cucascade::io::rest::testing::set_reactor_events(nullptr); }
  call_recorder(call_recorder const&)            = delete;
  call_recorder& operator=(call_recorder const&) = delete;

  [[nodiscard]] std::vector<call_event> log() const
  {
    std::lock_guard lock{_mutex};
    return _log;
  }

 private:
  cucascade::io::rest::testing::reactor_events _events;
  mutable std::mutex _mutex;
  std::vector<call_event> _log;
};

std::vector<call_event> calls_of(std::vector<call_event> const& log,
                                 std::string_view kind,
                                 bool started)
{
  std::vector<call_event> out;
  for (auto const& e : log) {
    if (e.kind == kind && e.started == started) { out.push_back(e); }
  }
  return out;
}

}  // namespace

TEST_CASE("rest_ioctx starts a write's first request without waiting for a loop tick",
          "[rest][write]")
{
  // A direct part queues CreateMultipartUpload from inside the worker's dispatch pass, and the
  // Create's completion queues the UploadPart: each must start in the pass that queued it (the
  // loop only blocks when nothing is runnable).  Asserted with the worker's own wait count.
  auto server = make_store();
  call_recorder recorder;
  auto ioctx = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes);
  auto object     = ioctx->open_io_object_for_write(uri("first-write.bin"));
  auto const t0   = std::chrono::steady_clock::now();
  REQUIRE(ioctx->host_write(*object, 0, part_bytes, data.data()) == part_bytes);
  auto const elapsed = std::chrono::steady_clock::now() - t0;
  CHECK(elapsed < 500ms);
  ioctx->flush(*object);
  require_stored(*server, "first-write.bin", data);
  ioctx->shutdown();

  auto const log           = recorder.log();
  auto const create_done   = calls_of(log, "create_upload", false);
  auto const part_started  = calls_of(log, "upload_part", true);
  auto const part_done     = calls_of(log, "upload_part", false);
  auto const commit_starts = calls_of(log, "complete_upload", true);
  REQUIRE(create_done.size() == 1);
  REQUIRE(part_started.size() == 1);
  REQUIRE(part_done.size() == 1);
  REQUIRE(commit_starts.size() == 1);
  // The UploadPart left in the pass that reaped the Create (no wait in between).
  CHECK(part_started.front().waits == create_done.front().waits);
}

namespace {

/// Delegates to @p inner, except that the first CreateMultipartUpload authorization of @p key
/// blocks until @ref release: the worker is held inside its dispatch pass meanwhile.  It then
/// signs normally.
class gated_authorizer final : public cucascade::io::rest::request_authorizer {
 public:
  gated_authorizer(std::shared_ptr<cucascade::io::rest::request_authorizer> inner, std::string key)
    : _inner(std::move(inner)), _key(std::move(key))
  {
  }

  cucascade::io::rest::authorized_request authorize(cucascade::io::rest::object_ref const& obj,
                                                    cucascade::io::rest::request_method method,
                                                    std::chrono::seconds timeout) override
  {
    return _inner->authorize(obj, method, timeout);
  }
  cucascade::io::rest::authorized_request authorize_request(
    cucascade::io::rest::request_spec const& spec, std::chrono::seconds timeout) override
  {
    if (spec.method == cucascade::io::rest::request_method::POST &&
        spec.query.starts_with("uploads")) {
      gate(spec.object);
    }
    return _inner->authorize_request(spec, timeout);
  }

  bool wait_entered(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock(_mutex);
    return _cv.wait_for(lock, timeout, [this] { return _entered; });
  }
  void release()
  {
    {
      std::lock_guard lock(_mutex);
      _released = true;
    }
    _cv.notify_all();
  }

 private:
  void gate(cucascade::io::rest::object_ref const& obj)
  {
    if (obj.key != _key || _used.exchange(true)) { return; }
    std::unique_lock lock(_mutex);
    _entered = true;
    _cv.notify_all();
    _cv.wait(lock, [this] { return _released; });
  }

  std::shared_ptr<cucascade::io::rest::request_authorizer> _inner;
  std::string _key;
  std::atomic<bool> _used{false};
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
};

}  // namespace

TEST_CASE(
  "rest_ioctx uploads parts parked on another reactor's CreateMultipartUpload on their "
  "own reactor",
  "[rest][write]")
{
  // Two whole parts on two reactors: one reactor creates the UploadId (held in its
  // authorizer), the other's part parks on the upload.  The Create's settlement starts its own
  // reactor's part in that pass and hands the other part back to the reactor it was dispatched
  // to, which starts it on its own connections when the hand-back wakes it.
  auto server  = make_store();
  auto gate    = std::make_shared<gated_authorizer>(make_authorizer(*server, signing::presigned),
                                                 "two-reactors.bin");
  auto cfg     = write_config();
  cfg.fan_out  = 2;
  auto context = std::make_shared<rest_reactor::reactor_context>(cfg, gate, nullptr);
  auto ioctx   = std::make_shared<rest_ioctx>(2, std::move(context));
  call_recorder recorder;
  ioctx->start();
  rest_shutdown_guard cleanup{*ioctx};
  auto const data    = rest_test_payload(2 * part_bytes);
  auto object        = ioctx->open_io_object_for_write(uri("two-reactors.bin"));
  auto written       = ioctx->host_write_async(*object, 0, data.size(), data.data());
  bool const entered = gate->wait_entered(5s);
  if (!entered) { gate->release(); }
  REQUIRE(entered);
  // The other reactor took its part and parked it on the upload (a reactor publishes its
  // counters once per pass: the held one cannot, so what shows is the other's request).
  CHECK(eventually([&] { return ioctx->stats().requests_started >= 1; }));
  gate->release();
  REQUIRE(std::move(written).get(10s) == data.size());
  ioctx->flush(*object);
  require_stored(*server, "two-reactors.bin", data);
  ioctx->shutdown();

  auto const log         = recorder.log();
  auto const create_done = calls_of(log, "create_upload", false);
  auto const parts       = calls_of(log, "upload_part", true);
  REQUIRE(create_done.size() == 1);
  REQUIRE(parts.size() == 2);
  auto const creator = create_done.front().thread;
  auto const local   = std::count_if(
    parts.begin(), parts.end(), [&](call_event const& e) { return e.thread == creator; });
  CHECK(local == 1);  // one part per reactor: the foreign one went home
  for (auto const& p : parts) {
    CHECK(p.attempt == 0);
    if (p.thread == creator) { CHECK(p.waits == create_done.front().waits); }
  }
}

TEST_CASE("rest_ioctx flush_async commits on any reactor and is idempotent", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 3);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(2 * part_bytes + 12345);
  auto object     = ioctx->open_io_object_for_write(uri("async-flush.bin"));
  REQUIRE(std::move(ioctx->host_write_async(*object, 0, data.size(), data.data())).get(20s) ==
          data.size());
  // Two flushes in flight together: one commits, the other joins it.
  auto first  = ioctx->flush_async(*object);
  auto second = ioctx->flush_async(*object);
  std::move(first).get(20s);
  std::move(second).get(20s);
  CHECK(server->complete_upload_count() == 1);
  require_stored(*server, "async-flush.bin", data);
  auto const requests = server->write_requests().size();
  std::move(ioctx->flush_async(*object)).get(20s);
  ioctx->flush(*object);
  CHECK(server->write_requests().size() == requests);
  CHECK(object->validation_tag() == server->stored_etag("async-flush.bin"));
}

TEST_CASE("rest_ioctx refuses a write while a flush runs", "[rest][write]")
{
  auto control = std::make_shared<cucascade::test::range_response_control>();
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind    = write_kind::put_object,
                                      .nth     = 0,
                                      .what    = write_fault::action::hold,
                                      .control = control});
  auto server = make_store(policy);
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx, control};
  auto const data = rest_test_payload(8192);
  auto object     = ioctx->open_io_object_for_write(uri("busy.bin"));
  REQUIRE(ioctx->host_write(*object, 0, 4096, data.data()) == 4096);
  auto flush = ioctx->flush_async(*object);
  REQUIRE(control->wait_for_holds(1, 5s));
  auto const refused = failure_of([&] { ioctx->host_write(*object, 4096, 4096, data.data()); });
  REQUIRE(refused != nullptr);
  CHECK(error_of(refused) == std::errc::device_or_resource_busy);
  control->release_all();
  std::move(flush).get(10s);
  require_stored(*server, "busy.bin", {data.begin(), data.begin() + 4096});
}

TEST_CASE("rest_ioctx creates the UploadId once and never re-sends a lost create", "[rest][write]")
{
  auto const data = rest_test_payload(2 * part_bytes);

  SECTION("a 503 answer is retried")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind = write_kind::create_upload, .nth = 0});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server, write_config(), 2);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("create-503.bin"));
    // Both whole parts race for the UploadId: one creates it, the other waits for it.
    auto a = ioctx->host_write_async(*object, 0, part_bytes, data.data());
    auto b = ioctx->host_write_async(*object, part_bytes, part_bytes, data.data() + part_bytes);
    REQUIRE(std::move(a).get(10s) == part_bytes);
    REQUIRE(std::move(b).get(10s) == part_bytes);
    CHECK(server->create_upload_count() == 2);  // the 503 and its retry
    ioctx->flush(*object);
    CHECK(server->upload_part_count() == 2);
    require_stored(*server, "create-503.bin", data);
  }

  SECTION("a lost response fails the writes waiting for it and is not re-sent")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind  = write_kind::create_upload,
                                        .nth   = 0,
                                        .what  = write_fault::action::drop_after_body,
                                        .apply = true});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("create-lost.bin"));
    CHECK_THROWS(ioctx->host_write(*object, 0, part_bytes, data.data()));
    CHECK(server->create_upload_count() == 1);
    CHECK(server->upload_part_count() == 0);
    // The upload the lost attempt created is an orphan nobody can abort (bucket lifecycle rules
    // collect it); re-sending would only have added a second one.
    CHECK(server->active_upload_count() == 1);
    // The failed write left the object's content unknown: it is never committed.
    CHECK_THROWS(ioctx->flush(*object));
    CHECK(server->complete_upload_count() == 0);
    CHECK(server->put_object_count() == 0);
    CHECK_FALSE(server->stored_object("create-lost.bin").has_value());
  }
}

TEST_CASE("rest_ioctx invalidates a path's metadata when its commit lands", "[rest][write]")
{
  auto how    = GENERATE(as<int>{}, 0, 1);  // 0: flush, 1: flush_async
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const v1 = rest_test_payload(4096);
  {
    auto first = ioctx->open_io_object_for_write(uri("versioned.bin"));
    REQUIRE(ioctx->host_write(*first, 0, v1.size(), v1.data()) == v1.size());
    ioctx->flush(*first);
  }

  auto writer = ioctx->open_io_object_for_write(uri("versioned.bin"));
  // A writable object is never shared through its identity: scoped to this open, size 0.
  auto const identity = writer->identity();
  CHECK(identity.kind == cucascade::io::identity_kind::opened_handle);
  CHECK(identity.size == 0);
  auto const v2 = rest_test_payload(8192);
  REQUIRE(ioctx->host_write(*writer, 0, v2.size(), v2.data()) == v2.size());

  // The write settled but is not committed: a reader opened now still sees the old version,
  // and metadata parsed through it is registered for the path...
  auto reader = ioctx->open_io_object(uri("versioned.bin"));
  CHECK(reader->size() == v1.size());
  ioctx->metadata_store().register_metadata(*reader,
                                            std::make_shared<cucascade::io::io_object_metadata>());
  REQUIRE(ioctx->metadata_store().has_path(reader->object_path()));

  // ...until the commit makes the new bytes the object's: it drops the path before returning.
  if (how == 0) {
    ioctx->flush(*writer);
  } else {
    std::move(ioctx->flush_async(*writer)).get(20s);
  }
  CHECK_FALSE(ioctx->metadata_store().has_path(reader->object_path()));
  require_rest_bytes_equal(read_back(*ioctx, "versioned.bin"), v2);
}

// =================================================================================================
// Behaviour of the io_uring reactor's write path
// =================================================================================================

TEST_CASE("rest zero bodies come from the static block in windows of at most 64 iovecs",
          "[rest][write]")
{
  namespace detail = cucascade::io::rest::detail;
  auto const block = detail::zero_block();
  REQUIRE(block.size() == detail::zero_block_bytes);
  CHECK(std::all_of(block.begin(), block.end(), [](std::byte b) { return b == std::byte{0}; }));
  // 100 MiB + 5 bytes: two windows (64 + 37 iovecs), the last iovec 5 bytes long.
  std::uint64_t const bytes = 100 * MiB + 5;
  auto next                 = detail::zero_body_generator(bytes, 64);
  std::uint64_t total       = 0;
  std::size_t windows       = 0;
  std::size_t iovecs        = 0;
  for (;;) {
    auto const w = next();
    if (w.empty()) { break; }
    ++windows;
    CHECK(w.size() <= 64);
    for (auto const& v : w) {
      CHECK(static_cast<std::byte const*>(v.iov_base) == block.data());
      CHECK(v.iov_len <= detail::zero_block_bytes);
      total += v.iov_len;
      ++iovecs;
    }
  }
  CHECK(total == bytes);
  CHECK(windows == 2);
  CHECK(iovecs == 101);
  CHECK(next().empty());  // stays at its end
  CHECK(detail::zero_body_generator(0, 64)().empty());
}

TEST_CASE("rest multipart ETags are md5(concat(binary part MD5s))-N", "[rest][write]")
{
  namespace detail = cucascade::io::rest::detail;
  // Two parts whose MD5s are md5("a") and md5("b"); the S3 rule computed independently below.
  std::vector<std::string> const parts{"\"0cc175b9c0f1b6a831c399e269772661\"",
                                       "92EB5FFEE6AE2FEC3AD71C777531578F"};
  // python3: hashlib.md5(md5(b"a").digest() + md5(b"b").digest()).hexdigest()
  CHECK(detail::multipart_etag(parts) == "96e024ba2074fe77e8e965ba43a704be-2");
  // A part ETag that is not an MD5 (SSE-KMS, other stores): the object cannot be checked.
  std::vector<std::string> const kms{"\"0cc175b9c0f1b6a831c399e269772661\"", "\"not-an-md5\""};
  CHECK(detail::multipart_etag(kms).empty());
  CHECK(detail::multipart_etag({}).empty());
}

TEST_CASE("rest_ioctx sends a never-written part larger than 64 MiB as zeros", "[rest][write]")
{
  // Part 1 (72 MiB) is never written: its UploadPart body is the static zero block, sent in
  // windows of 64 iovecs (the transport's SENDMSG limit), > 64 iovecs in all.
  auto server           = make_store();
  auto cfg              = write_config();
  cfg.write_split_bytes = 72 * MiB;
  auto ioctx            = make_write_ioctx(*server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(4096);
  auto object     = ioctx->open_io_object_for_write(uri("zeros.bin"));
  REQUIRE(ioctx->host_write(*object, 72 * MiB, data.size(), data.data()) == data.size());
  ioctx->flush(*object);
  CHECK(server->upload_part_count() == 2);
  auto const stored = server->stored_object("zeros.bin");
  REQUIRE(stored.has_value());
  REQUIRE(stored->size() == 72 * MiB + data.size());
  CHECK(std::all_of(stored->begin(),
                    stored->begin() + static_cast<std::ptrdiff_t>(72 * MiB),
                    [](std::uint8_t b) { return b == 0; }));
  CHECK(
    std::equal(data.begin(), data.end(), stored->begin() + static_cast<std::ptrdiff_t>(72 * MiB)));
  auto const stats = ioctx->stats();
  CHECK(stats.zero_part_bytes == 72 * MiB);
}

TEST_CASE("rest_ioctx re-signs a 403 on UploadPart and keeps Create / Complete final",
          "[rest][write]")
{
  auto const data             = rest_test_payload(part_bytes + 4096);
  auto cfg                    = write_config();
  cfg.max_auth_retry_attempts = 3;

  SECTION("UploadPart: data plane, bounded re-sign")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind   = write_kind::upload_part,
                                        .nth    = 0,
                                        .count  = 2,
                                        .status = 403,
                                        .code   = "AccessDenied"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server, cfg);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("resigned.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    ioctx->flush(*object);
    require_stored(*server, "resigned.bin", data);
    CHECK(server->upload_part_count() == 4);  // 2 refused + 2 parts
    CHECK(ioctx->stats().auth_retries == 2);
  }

  SECTION("a genuine AccessDenied fails after max_auth_retry_attempts")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind   = write_kind::upload_part,
                                        .nth    = 0,
                                        .count  = 100,
                                        .status = 403,
                                        .code   = "AccessDenied"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server, cfg);
    rest_shutdown_guard cleanup{*ioctx};
    auto object      = ioctx->open_io_object_for_write(uri("denied.bin"));
    auto const error = failure_of([&] { ioctx->host_write(*object, 0, part_bytes, data.data()); });
    REQUIRE(error != nullptr);
    CHECK(error_of(error) == cucascade::io::rest::make_http_error(403));
    CHECK(server->upload_part_count() == 3);
  }

  SECTION("CreateMultipartUpload: control plane, final")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::create_upload, .nth = 0, .status = 403, .code = "AccessDenied"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server, cfg);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("create-403.bin"));
    CHECK_THROWS(ioctx->host_write(*object, 0, part_bytes, data.data()));
    CHECK(server->create_upload_count() == 1);
  }
}

TEST_CASE("rest_ioctx never reports EEXIST for a conditional commit whose answer was lost",
          "[rest][write]")
{
  // The first create_new PUT takes effect but its answer is lost; the retry answers 412.  The
  // key may be ours or someone else's: the flush reports an unknown outcome (io_error).
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind  = write_kind::put_object,
                                      .nth   = 0,
                                      .what  = write_fault::action::drop_after_body,
                                      .apply = true});
  auto server = make_store(policy);
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(4096);
  auto object =
    ioctx->open_io_object_for_write(uri("lost-412.bin"), {.mode = write_mode::create_new});
  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  auto const error = failure_of([&] { ioctx->flush(*object); });
  REQUIRE(error != nullptr);
  CHECK(error_of(error) ==
        cucascade::io::rest::make_error_code(cucascade::io::rest::errc::io_error));
  CHECK(error_of(error) != std::errc::file_exists);
  CHECK(server->put_object_count() == 2);
  CHECK(object->validation_tag().empty());
}

TEST_CASE("rest_ioctx recovers a commit whose CompleteMultipartUpload answer was lost",
          "[rest][write]")
{
  auto const data = rest_test_payload(part_bytes + 4096);

  SECTION("the retry answers NoSuchUpload; the object carries our parts: committed")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind  = write_kind::complete_upload,
                                        .nth   = 0,
                                        .what  = write_fault::action::drop_after_body,
                                        .apply = true});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("lost-complete.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    ioctx->flush(*object);
    CHECK(server->complete_upload_count() == 2);
    require_stored(*server, "lost-complete.bin", data);
    CHECK(object->validation_tag() == server->stored_etag("lost-complete.bin"));
    CHECK(server->head_count() == 1);
  }

  SECTION("a lost Complete, then a failed flush: the next flush's NoSuchUpload recovers")
  {
    // The Complete's answer is lost and its retries answer 503: the flush fails.  The next
    // flush's Complete finds the upload gone; the lost answer is remembered across flushes, so
    // the HEAD recovery runs and commits.
    s3_write_policy policy;
    policy.faults.push_back(write_fault{.kind  = write_kind::complete_upload,
                                        .nth   = 0,
                                        .what  = write_fault::action::drop_after_body,
                                        .apply = true});
    // The lost attempt ran on a pooled connection: its retry is the free stale one, so three
    // 503s exhaust max_retry_attempts (3).
    policy.faults.push_back(
      write_fault{.kind = write_kind::complete_upload, .nth = 1, .count = 3, .status = 503});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object =
      ioctx->open_io_object_for_write(uri("lost-then-503.bin"), {.mode = write_mode::create_new});
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    CHECK_THROWS(ioctx->flush(*object));
    REQUIRE(server->stored_object("lost-then-503.bin").has_value());
    ioctx->flush(*object);
    CHECK(object->validation_tag() == server->stored_etag("lost-then-503.bin"));
  }

  SECTION("NoSuchUpload without a lost answer: the upload is gone, no false success")
  {
    s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::complete_upload, .nth = 0, .status = 404, .code = "NoSuchUpload"});
    auto server = make_store(policy);
    auto ioctx  = make_write_ioctx(*server);
    rest_shutdown_guard cleanup{*ioctx};
    auto object = ioctx->open_io_object_for_write(uri("gone.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    auto const error = failure_of([&] { ioctx->flush(*object); });
    REQUIRE(error != nullptr);
    CHECK(error_of(error) == cucascade::io::rest::make_http_error(404));
    CHECK(server->head_count() == 0);  // nothing to recover: no Complete answer was lost
    CHECK(object->validation_tag().empty());
    CHECK_FALSE(server->stored_object("gone.bin").has_value());
  }
}

TEST_CASE("rest_ioctx retries a PUT torn mid-body on a new connection", "[rest][write]")
{
  s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::put_object, .nth = 0, .what = write_fault::action::drop_mid_body});
  auto server = make_store(policy);
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(3 * MiB);
  auto object     = ioctx->open_io_object_for_write(uri("torn.bin"));
  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  ioctx->flush(*object);
  require_stored(*server, "torn.bin", data);
  CHECK(server->put_object_count() == 2);
  // The torn connection was closed (never reused), the retry connected anew.
  CHECK(server->accepted_connection_count() >= 2);
  CHECK(eventually([&] {
    auto const s = ioctx->stats();
    return s.closed >= 1 && s.retries >= 1 && s.puts == 1;
  }));
}

namespace {

/// @p blocks pinned 1 MiB blocks: a small staging resource for the part buffers.
class small_staging {
 public:
  explicit small_staging(std::size_t blocks)
    : _mr{0, _upstream, blocks * MiB, blocks * MiB, MiB, blocks, 0}
  {
  }
  [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource* get() noexcept { return &_mr; }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream{0};
  cucascade::memory::fixed_size_host_memory_resource _mr;
};

}  // namespace

TEST_CASE("rest_ioctx parks writes on part-buffer exhaustion while uploads are in flight",
          "[rest][write]")
{
  // Staging for at most two part buffers.  Four parts, each written in two halves in one
  // request: the halves that find no buffer while earlier parts upload park, and assemble once
  // those uploads returned their buffers.
  small_staging staging{12};
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 1, signing::presigned, staging.get());
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(4 * part_bytes);
  auto const half = part_bytes / 2;
  auto object     = ioctx->open_io_object_for_write(uri("parked.bin"));
  std::vector<write_slice> slices;
  for (std::size_t p = 0; p < 4; ++p) {
    for (std::size_t h = 0; h < 2; ++h) {
      auto const off = p * part_bytes + h * half;
      slices.push_back(write_slice{off, h == 0 ? half : part_bytes - half, data.data() + off});
    }
  }
  REQUIRE(std::move(ioctx->host_writev_async(*object, slices)).get(20s) == data.size());
  ioctx->flush(*object);
  require_stored(*server, "parked.bin", data);
  CHECK(server->upload_part_count() == 4);
  auto const stats = ioctx->stats();
  CHECK(stats.part_buffer_waits >= 1);
  CHECK(eventually([&] { return staging.get()->get_available_memory() == 12 * MiB; }));
}

TEST_CASE("rest_ioctx fails a write with out_of_memory when no part buffer can come back",
          "[rest][write]")
{
  // One part buffer: part 1 is half written (its buffer stays until the flush), nothing is in
  // flight, so the write into part 2 cannot wait for anything (the read staging rule).
  small_staging staging{6};
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 1, signing::presigned, staging.get());
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 4096);
  auto object     = ioctx->open_io_object_for_write(uri("oom.bin"));
  REQUIRE(ioctx->host_write(*object, 0, 4096, data.data()) == 4096);
  auto const error =
    failure_of([&] { ioctx->host_write(*object, part_bytes, 4096, data.data() + part_bytes); });
  REQUIRE(error != nullptr);
  CHECK_THROWS_AS(std::rethrow_exception(error), rmm::out_of_memory);
}

TEST_CASE("rest_ioctx writes device memory of every GPU", "[rest][write][gpu][multi_gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  pinned_staging staging;
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server, write_config(), 2, signing::presigned, staging.get());
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 3 * MiB);
  int previous    = 0;
  REQUIRE(cudaGetDevice(&previous) == cudaSuccess);
  for (int d = 0; d < devices; ++d) {
    CAPTURE(d);
    REQUIRE(cudaSetDevice(d) == cudaSuccess);
    void* source = nullptr;
    REQUIRE(cudaMalloc(&source, data.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(source, data.data(), data.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    auto const key = "gpu" + std::to_string(d) + ".bin";
    auto object    = ioctx->open_io_object_for_write(uri(key));
    auto future    = ioctx->device_write_async(*object,
                                            0,
                                            data.size(),
                                            static_cast<std::uint8_t const*>(source),
                                            ::cuda::stream_ref{cudaStreamLegacy});
    REQUIRE(cudaSetDevice(previous) == cudaSuccess);
    REQUIRE(std::move(future).get(20s) == data.size());
    ioctx->flush(*object);
    require_stored(*server, key, data);
    REQUIRE(cudaSetDevice(d) == cudaSuccess);
    REQUIRE(cudaFree(source) == cudaSuccess);
  }
  REQUIRE(cudaSetDevice(previous) == cudaSuccess);
  CHECK(ioctx->stats().d2h_copies >= static_cast<std::uint64_t>(2 * devices));
}

TEST_CASE("rest_ioctx writes over HTTPS on the user-space TLS record layer", "[rest][write][tls]")
{
  // This host has no tls.ko: kTLS prefer falls back to the user-space record layer, which seals
  // the bodies (16 records per SEND).  kTLS required fails cleanly instead.
  auto const ktls =
    GENERATE(cucascade::io::tls::ktls_mode::prefer, cucascade::io::tls::ktls_mode::require);
  bool const kernel_tls = ::access("/proc/net/tls_stat", R_OK) == 0;
  if (ktls == cucascade::io::tls::ktls_mode::require && !kernel_tls) {
    SKIP("kTLS is not available on this host");
  }
  s3_write_policy policy;
  policy.missing_key_404 = true;
  auto server            = std::make_unique<loopback_range_server>(
    rest_test_payload(4096),
    range_fault_policy{},
    std::vector<listed_object>{},
    std::unordered_map<std::string, cucascade::test::key_response_script>{},
    std::move(policy),
    cucascade::test::loopback_tls{.enabled = true});
  REQUIRE(server->https());
  auto cfg     = write_config();
  cfg.tls.ktls = ktls;
  auto ioctx   = make_write_ioctx(*server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(2 * part_bytes + 777);
  auto object     = ioctx->open_io_object_for_write(uri("tls.bin"));
  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  ioctx->flush(*object);
  require_stored(*server, "tls.bin", data);
  require_rest_bytes_equal(read_back(*ioctx, "tls.bin"), data);
  auto const stats = ioctx->stats();
  CHECK(stats.handshakes >= 1);
  if (kernel_tls) {
    CHECK(stats.ktls_sessions == stats.handshakes);
  } else {
    CHECK(stats.userspace_tls_sessions == stats.handshakes);
  }
}

TEST_CASE("rest_ioctx counts writes in its transport stats", "[rest][write]")
{
  auto server = make_store();
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(2 * part_bytes + 100);
  {
    auto object = ioctx->open_io_object_for_write(uri("stats-multi.bin"));
    REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
    ioctx->flush(*object);
  }
  {
    auto object = ioctx->open_io_object_for_write(uri("stats-single.bin"));
    REQUIRE(ioctx->host_write(*object, 0, 100, data.data()) == 100);
    ioctx->flush(*object);
  }
  CHECK(eventually([&] {
    auto const s = ioctx->stats();
    return s.commits == 2 && s.puts == 1 && s.upload_parts == 3 && s.creates == 1 &&
           s.completes == 1 && s.put_bytes == 100 && s.upload_part_bytes == data.size() &&
           s.commit_failures == 0 && s.write_failures == 0 && s.part_buffers_live == 0 &&
           s.bytes_sent >= data.size() + 100;
  }));
}

TEST_CASE("rest_ioctx caps the connections writes hold with max_write_connections", "[rest][write]")
{
  auto server               = make_store();
  auto cfg                  = write_config();
  cfg.max_connections       = 4;
  cfg.max_write_connections = 1;
  auto ioctx                = make_write_ioctx(*server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(4 * part_bytes);
  auto object     = ioctx->open_io_object_for_write(uri("capped.bin"));
  REQUIRE(std::move(ioctx->host_write_async(*object, 0, data.size(), data.data())).get(20s) ==
          data.size());
  ioctx->flush(*object);
  require_stored(*server, "capped.bin", data);
  CHECK(server->peak_part_uploads() == 1);
}

TEST_CASE("rest_ioctx sends bodies after 100 Continue when expect_continue is on", "[rest][write]")
{
  auto const mode = GENERATE(cucascade::test::expect_continue_mode::honor,
                             cucascade::test::expect_continue_mode::ignore);
  s3_write_policy policy;
  policy.expect        = mode;
  auto server          = make_store(policy);
  auto cfg             = write_config();
  cfg.expect_continue  = true;
  cfg.continue_timeout = 100ms;
  auto ioctx           = make_write_ioctx(*server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(part_bytes + 4096);
  auto object     = ioctx->open_io_object_for_write(uri("expect.bin"));
  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  ioctx->flush(*object);
  require_stored(*server, "expect.bin", data);
  CHECK(server->expect_continue_count() >= 3);  // two parts and the Complete
}

TEST_CASE("rest_ioctx S3 write throughput on loopback user-space TLS", "[.][bench_rest_write]")
{
  // Not part of the suites: run with "[bench_rest_write]" to print a number.  1 GiB in 16 MiB
  // parts.
  bool const https = GENERATE(true, false);
  s3_write_policy policy;
  policy.missing_key_404 = true;
  auto server            = std::make_unique<loopback_range_server>(
    rest_test_payload(4096),
    range_fault_policy{},
    std::vector<listed_object>{},
    std::unordered_map<std::string, cucascade::test::key_response_script>{},
    std::move(policy),
    cucascade::test::loopback_tls{.enabled = https});
  auto cfg              = write_config();
  cfg.write_split_bytes = 16 * MiB;
  cfg.max_connections   = 16;
  cfg.queue_depth       = 64;
  pinned_staging staging;
  auto ioctx = make_write_ioctx(*server, cfg, 2, signing::header, staging.get());
  rest_shutdown_guard cleanup{*ioctx};
  std::size_t const total = std::size_t{1} << 30;
  std::vector<std::uint8_t> data(total);
  for (std::size_t i = 0; i < total; i += 4096) {
    data[i] = static_cast<std::uint8_t>(i >> 12);
  }
  // first_part: write part 1 alone first, so the UploadId exists before the other parts are
  // dispatched; otherwise they park on the CreateMultipartUpload and are handed back to the
  // reactors they were dispatched to when it settles.  Both should run on every reactor.
  bool const first_part  = GENERATE(false, true);
  auto object            = ioctx->open_io_object_for_write(uri("bench.bin"), {.size_hint = total});
  auto const t0          = std::chrono::steady_clock::now();
  std::size_t const head = first_part ? 16 * MiB : 0;
  if (head != 0) { REQUIRE(ioctx->host_write(*object, 0, head, data.data()) == head); }
  REQUIRE(
    std::move(ioctx->host_write_async(*object, head, total - head, data.data() + head)).get(300s) ==
    total - head);
  ioctx->flush(*object);
  auto const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  auto const stats   = ioctx->stats();
  std::printf(
    "S3 multipart write, loopback %s%s: 1 GiB in %.3f s = %.2f GiB/s (%llu "
    "parts, %llu user-space TLS sessions)\n",
    https ? "user-space TLS" : "plain HTTP",
    first_part ? ", part 1 first" : "",
    seconds,
    1.0 / seconds,
    static_cast<unsigned long long>(stats.upload_parts),
    static_cast<unsigned long long>(stats.userspace_tls_sessions));
  CHECK(server->upload_part_count() == total / (16 * MiB));
}

// =================================================================================================
// Review fixes (regressions)
// =================================================================================================

TEST_CASE("rest_ioctx starts UploadParts while a staged read is parked on staging",
          "[rest][write][gpu]")
{
  // 16 MiB of staging: three assembled parts pin 15 MiB while their UploadParts wait for a
  // CreateMultipartUpload the server holds; a 12 MiB staged device read then parks.  Releasing
  // the Create must start the UploadParts in the pass that reaped it (the parked read retries
  // first and parks again: that must not end the pass), their buffers return, and the read gets
  // its staging, all without an unrelated request waking the loop.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  small_staging staging{16};
  auto hold_create = std::make_shared<cucascade::test::range_response_control>();
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind    = write_kind::create_upload,
                                      .nth     = 0,
                                      .what    = write_fault::action::hold,
                                      .control = hold_create});
  auto server = make_store(policy);
  cucascade::test::response_release_guard release_create{hold_create};
  auto const big = rest_test_payload(12 * MiB);
  server->put_stored_object("big.bin", big);
  call_recorder recorder;
  auto cfg                  = write_config();
  cfg.max_connections       = 4;
  cfg.min_segment_bytes     = 12 * MiB;
  cfg.max_segment_bytes     = 12 * MiB;
  cfg.hold_slot_during_copy = false;
  auto ioctx                = make_write_ioctx(*server, cfg, 1, signing::presigned, staging.get());
  rest_shutdown_guard cleanup{*ioctx};

  auto const data = rest_test_payload(3 * part_bytes);
  auto const half = part_bytes / 2;
  auto object     = ioctx->open_io_object_for_write(uri("starve.bin"));
  std::vector<write_slice> slices;
  for (std::size_t p = 0; p < 3; ++p) {
    for (std::size_t h = 0; h < 2; ++h) {
      auto const off = p * part_bytes + h * half;
      slices.push_back(write_slice{off, h == 0 ? half : part_bytes - half, data.data() + off});
    }
  }
  auto writes = ioctx->host_writev_async(*object, slices);
  REQUIRE(hold_create->wait_for_holds(1, 5s));
  auto reader = ioctx->open_io_object(uri("big.bin"));
  rmm::cuda_stream stream;
  rmm::device_buffer dst(big.size(), stream);
  auto read = ioctx->device_read_async(*reader,
                                       0,
                                       big.size(),
                                       static_cast<std::uint8_t*>(dst.data()),
                                       ::cuda::stream_ref{stream.value()});
  CHECK(eventually([&] { return ioctx->stats().staging_waits >= 1; }));
  hold_create->release_all();
  CHECK(std::move(writes).wait(10s));
  CHECK(std::move(read).wait(10s));
  REQUIRE(std::move(writes).get(10s) == data.size());
  REQUIRE(std::move(read).get(10s) == big.size());
  std::vector<std::uint8_t> back(big.size());
  REQUIRE(cudaMemcpy(back.data(), dst.data(), back.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
  require_rest_bytes_equal(back, big);
  ioctx->flush(*object);
  require_stored(*server, "starve.bin", data);
  ioctx->shutdown();

  // Same-pass invariant: the UploadParts left in the pass that reaped the Create.
  auto const log         = recorder.log();
  auto const create_done = calls_of(log, "create_upload", false);
  auto const parts       = calls_of(log, "upload_part", true);
  REQUIRE(create_done.size() == 1);
  REQUIRE(parts.size() == 3);
  for (auto const& p : parts) {
    CHECK(p.waits == create_done.front().waits);
  }
}

TEST_CASE("rest_ioctx never reports EEXIST after a lost conditional answer of an earlier flush",
          "[rest][write]")
{
  // PUT #0 takes effect but its answer is lost; #1 and #2 answer 503: the first flush exhausts
  // its attempts and fails, the upload stays usable.  The next flush's PUT answers 412 because
  // our first attempt created the key: an unknown outcome (io_error), never EEXIST.
  s3_write_policy policy;
  policy.faults.push_back(write_fault{.kind  = write_kind::put_object,
                                      .nth   = 0,
                                      .what  = write_fault::action::drop_after_body,
                                      .apply = true});
  policy.faults.push_back(
    write_fault{.kind = write_kind::put_object, .nth = 1, .count = 2, .status = 503});
  auto server = make_store(policy);
  auto ioctx  = make_write_ioctx(*server);
  rest_shutdown_guard cleanup{*ioctx};
  auto const data = rest_test_payload(4096);
  auto object =
    ioctx->open_io_object_for_write(uri("lost-412-later.bin"), {.mode = write_mode::create_new});
  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  auto const first = failure_of([&] { ioctx->flush(*object); });
  REQUIRE(first != nullptr);
  CHECK(server->put_object_count() == 3);
  REQUIRE(server->stored_object("lost-412-later.bin").has_value());
  auto const second = failure_of([&] { ioctx->flush(*object); });
  REQUIRE(second != nullptr);
  CHECK(server->put_object_count() == 4);
  CHECK(error_of(second) != std::errc::file_exists);
  CHECK(error_of(second) ==
        cucascade::io::rest::make_error_code(cucascade::io::rest::errc::io_error));
}

TEST_CASE("rest_ioctx retries a CreateMultipartUpload whose pooled connection was stale",
          "[rest][write]")
{
  // The pooled connection dies before any response byte (the server read the Create and closed
  // without applying it): curl's reuse retry, free, also for a Create.
  s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::create_upload, .nth = 0, .what = write_fault::action::drop_after_body});
  auto server = make_store(policy);
  server->put_stored_object("warm.bin", rest_test_payload(4096));
  auto cfg            = write_config();
  cfg.max_connections = 1;
  auto ioctx          = make_write_ioctx(*server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  {
    auto warm = ioctx->open_io_object(uri("warm.bin"));
    std::vector<std::uint8_t> buf(4096);
    REQUIRE(ioctx->host_read(*warm, 0, buf.size(), buf.data()) == buf.size());
  }
  auto const data = rest_test_payload(part_bytes);
  auto object     = ioctx->open_io_object_for_write(uri("stale-create.bin"));
  REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
  CHECK(server->create_upload_count() == 2);
  ioctx->flush(*object);
  require_stored(*server, "stale-create.bin", data);
  CHECK(ioctx->stats().stale_retries >= 1);
}

TEST_CASE("rest_ioctx writes and flushes while a staged read is parked", "[rest][write][gpu]")
{
  // Read A holds 6 of 8 MiB of staging (its body held by the server), read B parks.  Whole-part
  // writes and their flush need no staging: they must not wait behind the parked read.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  small_staging staging{8};
  auto hold = std::make_shared<cucascade::test::range_response_control>();
  range_fault_policy faults;
  faults.body.hold_after_bytes = 0;
  faults.body.control          = hold;
  auto const fixture           = rest_test_payload(6 * MiB);
  auto server                  = std::make_unique<loopback_range_server>(
    fixture,
    faults,
    std::vector<listed_object>{},
    std::unordered_map<std::string, cucascade::test::key_response_script>{},
    s3_write_policy{});
  cucascade::test::response_release_guard release{hold};
  auto cfg                  = write_config();
  cfg.max_connections       = 4;
  cfg.min_segment_bytes     = 6 * MiB;
  cfg.max_segment_bytes     = 6 * MiB;
  cfg.hold_slot_during_copy = false;
  auto ioctx                = make_write_ioctx(*server, cfg, 1, signing::presigned, staging.get());
  rest_shutdown_guard cleanup{*ioctx, hold};
  rmm::cuda_stream stream;
  rmm::device_buffer a_dst(fixture.size(), stream);
  rmm::device_buffer b_dst(fixture.size(), stream);
  auto a_obj = ioctx->open_io_object(uri("a.bin"));
  auto b_obj = ioctx->open_io_object(uri("b.bin"));
  auto a     = ioctx->device_read_async(*a_obj,
                                    0,
                                    fixture.size(),
                                    static_cast<std::uint8_t*>(a_dst.data()),
                                    ::cuda::stream_ref{stream.value()});
  REQUIRE(hold->wait_for_holds(1, 5s));
  auto b = ioctx->device_read_async(*b_obj,
                                    0,
                                    fixture.size(),
                                    static_cast<std::uint8_t*>(b_dst.data()),
                                    ::cuda::stream_ref{stream.value()});
  REQUIRE(eventually([&] { return ioctx->stats().staging_waits >= 1; }));

  auto const data = rest_test_payload(2 * part_bytes);
  auto object     = ioctx->open_io_object_for_write(uri("beside.bin"));
  auto written    = ioctx->host_write_async(*object, 0, data.size(), data.data());
  REQUIRE(std::move(written).wait(5s));
  REQUIRE(std::move(written).get(1s) == data.size());
  auto flushed = ioctx->flush_async(*object);
  REQUIRE(std::move(flushed).wait(5s));
  std::move(flushed).get(1s);
  require_stored(*server, "beside.bin", data);

  hold->release_all();
  REQUIRE(std::move(a).get(10s) == fixture.size());
  REQUIRE(std::move(b).get(10s) == fixture.size());
}

TEST_CASE("rest_ioctx releases part buffers when it is destroyed before its write objects",
          "[rest][write]")
{
  // A half-written part pins a part buffer of the context's host memory resource.  The ioctx
  // goes first: its destructor returns that buffer while the resource is alive, so the object
  // dropped later touches nothing of it (and can no longer write or commit).
  auto server = make_store();
  auto data   = rest_test_payload(4096);
  std::shared_ptr<cucascade::io::io_object> object;
  {
    small_staging staging{8};
    {
      auto ioctx = make_write_ioctx(*server, write_config(), 1, signing::presigned, staging.get());
      object     = ioctx->open_io_object_for_write(uri("orphaned.bin"));
      REQUIRE(ioctx->host_write(*object, 0, data.size(), data.data()) == data.size());
      CHECK(staging.get()->get_available_memory() == 8 * MiB - part_bytes);
    }
    CHECK(staging.get()->get_available_memory() == 8 * MiB);
  }
  object.reset();
  CHECK(server->write_requests().empty());
}
