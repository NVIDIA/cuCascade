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

// End-to-end S3 writes through rest_ioctx (docs/io/rest-uring-design.md section 5): the public
// write API (open_io_object_for_write, host / device writes, flush) against the loopback
// server's in-memory object store, signed by the SigV4 authorizers.

#include "rest_contract_test_utils.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
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
