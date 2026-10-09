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

// Cache-level object identity: the fs_cache keys its chunks by object
// generation (path + strong ETag), retires a path's older generations when a
// new one appears, keeps a retired generation alive for as long as a handle or
// an in-flight operation uses it, and fails a read whose object changed rather
// than publishing the replacement's bytes under the old identity.
//
// These tests drive the cache through the real cudf-coupled
// cucascade::io::datasource, the friend that reaches the cache's prefetch
// paths.  Two kinds of backend stand behind it: a loopback REST server whose
// per-key scripts overwrite an object between opens (end to end through the
// reactor), and a held backend whose fills the test completes or fails by hand
// (deterministic lifetime and ordering checks).  The wire-level conditional-GET
// contract is covered by test/io/rest/test_rest_cache_identity.cpp.
//
// The same fixture also hosts the fs_cache summary snapshot test, which needs a
// cache with live read counters but nothing about object identity.

#include "io/rest/loopback_range_server.hpp"
#include "io/rest/mock_authorizer.hpp"
#include "utils/test_memory_resources.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cucascade::io::datasource;
using cucascade::io::object_changed_error;
using cucascade::io::rest::rest_io_object;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::key_response_script;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::scripted_response;

constexpr std::size_t chunk_bytes  = 1U << 20;
constexpr std::size_t object_bytes = 64U << 10;  // one chunk, only partly populated
constexpr char const* object_uri   = "s3://bucket/object.bin";

std::unique_ptr<cucascade::memory::memory_reservation_manager> identity_memory()
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ULL << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource)
    .set_per_numa_region_capacity(256ULL << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0);
  return std::make_unique<cucascade::memory::memory_reservation_manager>(builder.build());
}

cucascade::io::cache::config identity_cache_config()
{
  cucascade::io::cache::config cfg;
  cfg.mode                            = cucascade::io::cache::cache_mode::cucs;
  cfg.eviction                        = cucascade::io::cache::eviction_policy::lru;
  cfg.min_prefetching_budget_fraction = 0.5;
  cfg.eviction_threshold_fraction     = 1.0;
  cfg.apply_mode();
  return cfg;
}

/// Hint [0, size) of @p ds and take staging buffers for it, as a scan does.
void advise(datasource& ds, std::size_t size)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<std::int64_t>(size));
  ds.fadvise(ranges, 0);
  REQUIRE(ds.prepare_prefetch(false) == cucascade::io::prepare_result::prepared);
}

std::vector<std::uint8_t> read_all(datasource& ds)
{
  std::vector<std::uint8_t> bytes(ds.size());
  REQUIRE(ds.host_read(0, bytes.size(), bytes.data()) == bytes.size());
  return bytes;
}

void require_generations(cucascade::io::cache::fs_cache const& cache,
                         std::string const& path,
                         std::size_t alive,
                         std::size_t retired)
{
  CHECK(cache.generation_count(path) == alive);
  CHECK(cache.retired_generation_count() == retired);
}

std::vector<std::uint8_t> base_object()
{
  std::vector<std::uint8_t> bytes(object_bytes);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + 7U) & 0xffU);
  }
  return bytes;
}

/// The bytes the loopback server serves for version @p version of the object:
/// each version is the base object with every byte XORed with the version
/// number, so a read that mixes or mislabels versions cannot go unnoticed.
std::vector<std::uint8_t> version_bytes(unsigned version)
{
  auto bytes = base_object();
  for (auto& b : bytes) {
    b = static_cast<std::uint8_t>(b ^ version);
  }
  return bytes;
}

std::string version_tag(unsigned version) { return "\"v" + std::to_string(version) + "\""; }

/// A script under which the object is overwritten between requests: HEAD number
/// k and GET number k both describe version k + 1.  An open then costs one HEAD
/// and a one-chunk fill one GET, so "the n-th open" sees version n.
key_response_script overwritten_every_open(unsigned versions)
{
  key_response_script script;
  for (unsigned version = 1; version <= versions; ++version) {
    script.heads.push_back(scripted_response{.etag = version_tag(version)});
    script.gets.push_back(scripted_response{.etag     = version_tag(version),
                                            .body_xor = static_cast<std::uint8_t>(version)});
  }
  return script;
}

/// An ioctx on a loopback REST server with the cache built over the HOST space,
/// so the reactor's staging block equals the cache's 1 MiB chunk.
struct rest_cache_fixture {
  explicit rest_cache_fixture(loopback_range_server const& server) : memory(identity_memory())
  {
    auto* host = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
    REQUIRE(host != nullptr);
    cucascade::io::rest::config cfg{};
    cfg.request_timeout_s       = 5;
    cfg.tls_verify              = false;
    cfg.max_connections         = 1;
    cfg.max_retry_attempts      = 2;
    cfg.max_auth_retry_attempts = 1;
    cfg.retry_backoff_base      = 1ms;
    cfg.retry_jitter            = 0ms;
    cfg.honor_retry_after       = false;
    auto authorizer             = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
    auto reactor_context        = std::make_shared<rest_reactor::reactor_context>(
      cfg,
      std::move(authorizer),
      const_cast<cucascade::memory::memory_space*>(host)
        ->get_memory_resource_of<cucascade::memory::Tier::HOST>());
    context = std::make_shared<rest_ioctx>(1, std::move(reactor_context));
    context->start();
    context->initialize_cache(*memory, identity_cache_config(), nullptr);
    REQUIRE(context->cache() != nullptr);
    REQUIRE(context->cache()->chunk_size() == chunk_bytes);
  }

  ~rest_cache_fixture()
  {
    if (context) {
      context->shutdown_cache();
      context->shutdown();
    }
  }

  rest_cache_fixture(rest_cache_fixture const&)            = delete;
  rest_cache_fixture& operator=(rest_cache_fixture const&) = delete;

  [[nodiscard]] std::unique_ptr<datasource> open(std::string const& uri)
  {
    return cucascade::io::open_datasource(context, uri);
  }

  /// Open @p uri, take its whole (one-chunk) range into the cache, and check
  /// that the bytes arrived with one GET and that a repeat read costs none.
  [[nodiscard]] std::unique_ptr<datasource> open_resident(loopback_range_server const& server,
                                                          std::string const& uri,
                                                          std::vector<std::uint8_t> const& expected)
  {
    auto ds = open(uri);
    REQUIRE(ds->size() == expected.size());
    REQUIRE_FALSE(ds->get_io_object().validation_tag().empty());
    advise(*ds, expected.size());
    auto const before = server.get_count();
    CHECK(read_all(*ds) == expected);
    CHECK(server.get_count() == before + 1);
    CHECK(read_all(*ds) == expected);
    CHECK(server.get_count() == before + 1);
    return ds;
  }

  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::shared_ptr<rest_ioctx> context;
};

// ---------------------------------------------------------------------------
// A backend whose fills the test holds, completes and fails by hand.
// ---------------------------------------------------------------------------

struct held_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{1};
};

class held_backend {
 public:
  using io_object_type                  = rest_io_object;
  using reactor_config_type             = held_config;
  static constexpr bool prefers_bulk_io = false;

  [[nodiscard]] held_config const& get_config() const noexcept { return config; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return chunk_bytes; }
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }

  void enqueue(std::unique_ptr<cucascade::io::grouped_io_request> request) noexcept
  {
    requests.push_back(std::move(request));
  }

  std::unique_ptr<cucascade::io::grouped_io_request> take()
  {
    if (requests.empty()) { return {}; }
    auto result = std::move(requests.front());
    requests.pop_front();
    return result;
  }

  std::size_t host_read(rest_io_object const&, std::size_t, std::size_t, std::uint8_t*)
  {
    throw std::logic_error("held backend reads must use the request queue");
  }

  void start() {}
  void interrupt() {}
  void shutdown()
  {
    while (auto request = take()) {
      request->cancel_remaining(std::make_exception_ptr(std::runtime_error("test shutdown")));
    }
  }

  static std::unique_ptr<rest_io_object> create_io_object(std::string path)
  {
    return std::make_unique<rest_io_object>(
      std::move(path), "bucket", "key", chunk_bytes, "\"one\"");
  }
  static bool supports(std::string_view) { return true; }
  static std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

  held_config config;
  std::deque<std::unique_ptr<cucascade::io::grouped_io_request>> requests;
};

class held_context final : public cucascade::io::templated_ioctx<held_backend> {
 public:
  held_context() : templated_ioctx(1, [] { return std::make_unique<held_backend>(); }) {}

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::restful;
  }

  held_backend& backend() { return *_reactors.front(); }
};

/// One request taken off the held backend; a request still held when the test
/// ends is cancelled so no completion is left dangling.
struct held_request {
  held_request() = default;
  ~held_request()
  {
    if (request) {
      request->cancel_remaining(std::make_exception_ptr(std::runtime_error("test cleanup")));
    }
  }
  held_request(held_request const&)            = delete;
  held_request& operator=(held_request const&) = delete;

  /// Fill every slice with @p value and complete it successfully.
  void complete(std::uint8_t value)
  {
    REQUIRE(request != nullptr);
    while (!request->empty()) {
      auto slice = request->take_front();
      if (slice.h_buffer.is_contiguous()) {
        std::fill_n(std::get<std::uint8_t*>(slice.h_buffer.buffer), slice.size(), value);
      } else {
        for (auto* chunk : slice.h_buffer.fragments()) {
          std::fill_n(chunk->data + slice.offset() - chunk->offset, slice.size(), value);
        }
      }
      if (slice.on_complete) { (*slice.on_complete)(slice.h_buffer.fragments(), true); }
      request->coordinator->on_complete();
    }
    request.reset();
  }

  /// Fail the whole request as a refused conditional GET would.
  void fail(std::string const& path, std::string const& expected, std::string const& observed)
  {
    REQUIRE(request != nullptr);
    request->cancel_remaining(
      std::make_exception_ptr(object_changed_error(path, expected, observed)));
    request.reset();
  }

  std::unique_ptr<cucascade::io::grouped_io_request> request;
};

std::shared_ptr<rest_io_object> held_object(std::string const& path, std::string const& tag)
{
  return std::make_shared<rest_io_object>(path, "controlled", "generation.bin", chunk_bytes, tag);
}

/// Holds back everything queued on a consumer stream until released: a host
/// function parked on a producer stream spins, and the consumer waits on an event
/// recorded behind it.  A cached-chunk copy enqueued on the consumer therefore
/// cannot finish until the gate opens.
struct stream_gate {
  stream_gate(rmm::cuda_stream& producer, rmm::cuda_stream& consumer)
    : producer(producer), consumer(consumer)
  {
  }

  ~stream_gate()
  {
    release.store(true);
    std::ignore = cudaStreamSynchronize(producer.value());
    std::ignore = cudaStreamSynchronize(consumer.value());
    if (event != nullptr) { std::ignore = cudaEventDestroy(event); }
  }

  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  void arm()
  {
    REQUIRE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaLaunchHostFunc(
              producer.value(),
              [](void* opaque) {
                auto& gate = *static_cast<stream_gate*>(opaque);
                gate.entered.store(true);
                while (!gate.release.load()) {
                  std::this_thread::yield();
                }
              },
              this) == cudaSuccess);
    REQUIRE(cudaEventRecord(event, producer.value()) == cudaSuccess);
    REQUIRE(cudaStreamWaitEvent(consumer.value(), event, 0) == cudaSuccess);
    auto const deadline = std::chrono::steady_clock::now() + 5s;
    while (!entered.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    REQUIRE(entered.load());
  }

  rmm::cuda_stream& producer;
  rmm::cuda_stream& consumer;
  cudaEvent_t event{nullptr};
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};

template <typename Action>
std::exception_ptr capture_failure(Action&& action)
{
  try {
    std::forward<Action>(action)();
  } catch (...) {
    return std::current_exception();
  }
  FAIL("reading a replaced object must fail");
  return {};
}

void require_changed(std::exception_ptr failure,
                     std::string const& path,
                     std::string const& expected,
                     std::string const& observed = {})
{
  REQUIRE(static_cast<bool>(failure));
  try {
    std::rethrow_exception(failure);
  } catch (object_changed_error const& error) {
    CHECK(error.object_path() == path);
    CHECK(error.expected_tag() == expected);
    CHECK(error.observed_tag() == observed);
  } catch (...) {
    FAIL("expected object_changed_error");
  }
}

/// The counters of one fs_cache::summary() line: the five `global[...]` values
/// followed by the five `last_cycle[...]` values (reads, hits, h2d, miss,
/// evictions), taken from the integers after each `=`.
std::vector<std::uint64_t> summary_counts(std::string const& text)
{
  std::vector<std::uint64_t> counts;
  for (auto pos = text.find('='); pos != std::string::npos; pos = text.find('=', pos + 1)) {
    counts.push_back(std::stoull(text.substr(pos + 1)));
  }
  return counts;
}

}  // namespace

// ===========================================================================
// REST: opens, reuse and overwrite, end to end through the reactor
// ===========================================================================

TEST_CASE("cache identity reuses a strong generation across opens", "[cache][cache_identity][rest]")
{
  range_fault_policy fault;
  fault.successful_head_etag = "\"generation-one\"";
  fault.successful_get_etag  = fault.successful_head_etag;
  loopback_range_server server(base_object(), fault);
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto first = fixture.open(object_uri);
  REQUIRE(first->get_io_object().validation_tag() == fault.successful_head_etag);
  advise(*first, object_bytes);
  CHECK(read_all(*first) == base_object());
  REQUIRE(server.get_count() == 1);
  CHECK(read_all(*first) == base_object());
  CHECK(server.get_count() == 1);

  // A second open observes the same tag, so it shares the generation and finds
  // the bytes already resident: one more HEAD, no more GET.
  auto second = fixture.open(object_uri);
  CHECK(second->get_io_object().raw_file_cache_id() == first->get_io_object().raw_file_cache_id());
  CHECK(read_all(*second) == base_object());
  CHECK(server.get_count() == 1);
  CHECK(server.head_count() == 2);
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity refetches on each open without a strong validator",
          "[cache][cache_identity][rest]")
{
  for (std::string const tag :
       {std::string{}, std::string{"W/\"x\""}, std::string{"*"}, std::string{"open#1"}}) {
    DYNAMIC_SECTION("tag=" << (tag.empty() ? "<none>" : tag))
    {
      range_fault_policy fault;
      fault.successful_head_etag = tag;
      fault.successful_get_etag  = tag;
      loopback_range_server server(base_object(), fault);
      rest_cache_fixture fixture(server);
      auto& cache = *fixture.context->cache();

      auto first           = fixture.open(object_uri);
      auto const first_key = first->get_io_object().raw_file_cache_id();
      advise(*first, object_bytes);
      CHECK(read_all(*first) == base_object());
      REQUIRE(server.get_count() == 1);
      // Within one open the bytes are cached as usual.
      CHECK(read_all(*first) == base_object());
      CHECK(server.get_count() == 1);

      // A second open has no version it could share: it has its own generation,
      // fetches again, and the first open's generation is retired.
      auto second = fixture.open(object_uri);
      CHECK(second->get_io_object().raw_file_cache_id() != first_key);
      advise(*second, object_bytes);
      CHECK(read_all(*second) == base_object());
      CHECK(server.get_count() == 2);
      CHECK(server.head_count() == 2);
      require_generations(cache, object_uri, 2, 1);

      // Without a strong tag neither fill was conditional.
      auto const requests = server.get_requests();
      REQUIRE(requests.size() == 2);
      for (auto const& request : requests) {
        CHECK(request.if_match.empty());
        REQUIRE(request.ranges.size() == 1);
      }

      // The retired generation still serves its own open, from its own bytes.
      CHECK(read_all(*first) == base_object());
      CHECK(server.get_count() == 2);
      first.reset();
      require_generations(cache, object_uri, 1, 0);
      CHECK(cache.claimed_bytes() == chunk_bytes);
    }
  }
}

TEST_CASE("cache identity isolates an ETag-only overwrite", "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"] = overwritten_every_open(2);
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto first     = fixture.open_resident(server, object_uri, version_bytes(1));
  auto const key = first->get_io_object().raw_file_cache_id();
  CHECK(key == rest_io_object::generation_key(object_uri, version_tag(1)));
  require_generations(cache, object_uri, 1, 0);

  // The key is overwritten in place: same path, same size, new ETag and bytes.
  auto second = fixture.open_resident(server, object_uri, version_bytes(2));
  CHECK(second->get_io_object().raw_file_cache_id() ==
        rest_io_object::generation_key(object_uri, version_tag(2)));
  CHECK(second->get_io_object().raw_file_cache_id() != key);
  CHECK(server.get_count() == 2);

  // Both versions are alive, one of them retired, each still reading its own
  // bytes; the retired one costs nothing to read and is not refetched.
  require_generations(cache, object_uri, 2, 1);
  CHECK(cache.claimed_bytes() == 2 * chunk_bytes);
  CHECK(read_all(*first) == version_bytes(1));
  CHECK(read_all(*second) == version_bytes(2));
  CHECK(server.get_count() == 2);

  // Dropping the last handle on the retired generation reclaims its chunks.
  first.reset();
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
  CHECK(read_all(*second) == version_bytes(2));
}

TEST_CASE("cache identity bounds retired generations across five overwrites",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"] = overwritten_every_open(6);
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto current = fixture.open_resident(server, object_uri, version_bytes(1));
  for (unsigned version = 2; version <= 6; ++version) {
    auto const previous_tag = std::string(current->get_io_object().validation_tag());
    auto next               = fixture.open_resident(server, object_uri, version_bytes(version));
    REQUIRE(next->get_io_object().validation_tag() != previous_tag);
    // The old generation outlives the overwrite only while its handle does.
    require_generations(cache, object_uri, 2, 1);
    current.reset();
    require_generations(cache, object_uri, 1, 0);
    CHECK(cache.claimed_bytes() == chunk_bytes);
    current = std::move(next);
  }
  CHECK(server.get_count() == 6);
  CHECK(server.head_count() == 6);
  current.reset();
  // The last generation stays mapped (it is the current version) and resident.
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity retains a disposed datasource generation until destruction",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"] = overwritten_every_open(2);
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto first = fixture.open_resident(server, object_uri, version_bytes(1));
  first->update(cucascade::io::cache::scan_stage::disposed);
  auto second = fixture.open_resident(server, object_uri, version_bytes(2));
  require_generations(cache, object_uri, 2, 1);
  first.reset();
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity releases read pins when a read of a retired generation fails",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"] = overwritten_every_open(2);
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto first  = fixture.open_resident(server, object_uri, version_bytes(1));
  auto second = fixture.open_resident(server, object_uri, version_bytes(2));
  require_generations(cache, object_uri, 2, 1);
  REQUIRE(cache.claimed_bytes() == 2 * chunk_bytes);

  // A vectored read whose second destination is null fails after the first
  // range's resident chunk has been pinned; the pin must be released.
  std::vector<std::uint8_t> head(64);
  std::array<cucascade::io::slice, 2> slices{};
  slices[0]     = cucascade::io::slice{0, head.size(), head.data()};
  slices[1].rng = cucascade::io::range{head.size(), head.size()};
  auto failed   = first->host_read_ranges_async(slices);
  CHECK_THROWS_AS(failed.get(), std::invalid_argument);
  CHECK(server.get_count() == 2);

  CHECK(read_all(*first) == version_bytes(1));
  CHECK(server.get_count() == 2);
  first.reset();
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity pins retired bytes until a gated device copy completes",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"] = overwritten_every_open(2);
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto first = fixture.open_resident(server, object_uri, version_bytes(1));
  rmm::cuda_stream producer;
  rmm::cuda_stream consumer;
  rmm::device_buffer destination(object_bytes, consumer);
  stream_gate gate(producer, consumer);
  gate.arm();

  // A device read of the resident chunks: the copy to the GPU is queued behind the
  // gate, so the read stays in flight while it holds the chunks.
  auto copying = first->device_read_async(
    0, object_bytes, static_cast<std::uint8_t*>(destination.data()), consumer);
  CHECK(copying.wait_for(0ms) == std::future_status::timeout);
  CHECK(server.get_count() == 1);

  // The object is overwritten and the first datasource goes away mid-copy.
  auto second = fixture.open_resident(server, object_uri, version_bytes(2));
  require_generations(cache, object_uri, 2, 1);
  first.reset();
  // The pending copy's retirement still owns the retired generation and its bytes.
  require_generations(cache, object_uri, 2, 1);
  CHECK(cache.claimed_bytes() == 2 * chunk_bytes);

  gate.release.store(true);
  REQUIRE(copying.wait_for(5s) == std::future_status::ready);
  CHECK(copying.get() == object_bytes);
  // Once the stream has passed the copy the retirement lets go and the generation
  // is reclaimed.
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (cache.retired_generation_count() != 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);

  // The copy delivered the retired generation's bytes, not the replacement's.
  std::vector<std::uint8_t> bytes(object_bytes);
  REQUIRE(
    cudaMemcpyAsync(
      bytes.data(), destination.data(), bytes.size(), cudaMemcpyDeviceToHost, consumer.value()) ==
    cudaSuccess);
  consumer.synchronize();
  CHECK(bytes == version_bytes(1));
  CHECK(server.get_count() == 2);
}

TEST_CASE("cache identity rejects mid-open replacement on allocated fills",
          "[cache][cache_identity][rest]")
{
  // The key is replaced after the open: the conditional GET is refused.
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"].heads = {scripted_response{.etag = version_tag(1)}};
  scripts["object.bin"].gets  = {scripted_response{.status = 412}};
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto source         = fixture.open(object_uri);
  auto const expected = std::string(source->get_io_object().validation_tag());
  REQUIRE(rest_io_object::is_strong_tag(expected));
  advise(*source, source->size());
  auto const footprint = cache.claimed_bytes();
  REQUIRE(footprint == chunk_bytes);

  std::vector<std::uint8_t> bytes(source->size());
  for (std::size_t attempt = 1; attempt <= 2; ++attempt) {
    auto failure = capture_failure([&] { source->host_read(0, bytes.size(), bytes.data()); });
    require_changed(failure, object_uri, expected);
    // Terminal: exactly one GET per read, and nothing published or leaked.
    CHECK(server.get_count() == attempt);
    CHECK(cache.claimed_bytes() == footprint);
  }
}

TEST_CASE("cache identity rejects mid-open replacement on bypass reads",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"].heads = {scripted_response{.etag = version_tag(1)}};
  // The replacement answers the conditional GET with a 206 of its own version.
  scripts["object.bin"].gets = {scripted_response{.etag = version_tag(2)}};
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto source         = fixture.open(object_uri);
  auto const expected = std::string(source->get_io_object().validation_tag());
  REQUIRE(rest_io_object::is_strong_tag(expected));
  // No fadvise: nothing is allocated, so the read bypasses the cache's chunks.
  REQUIRE(cache.claimed_bytes() == 0);

  std::vector<std::uint8_t> bytes(source->size());
  for (std::size_t attempt = 1; attempt <= 2; ++attempt) {
    auto failure = capture_failure([&] { source->host_read(0, bytes.size(), bytes.data()); });
    require_changed(failure, object_uri, expected, version_tag(2));
    CHECK(server.get_count() == attempt);
    CHECK(cache.claimed_bytes() == 0);
  }
}

TEST_CASE("cache identity fails a mixed hit and replaced miss without publication",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"].heads = {scripted_response{.etag = version_tag(1)}};
  // The first GET fills the resident prefix; the object is replaced after it.
  scripts["object.bin"].gets = {scripted_response{.etag = version_tag(1)},
                                scripted_response{.status = 412}};
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  auto source         = fixture.open(object_uri);
  auto const expected = std::string(source->get_io_object().validation_tag());
  REQUIRE(rest_io_object::is_strong_tag(expected));
  constexpr std::size_t prefix = 4096;
  advise(*source, prefix);
  std::vector<std::uint8_t> resident(prefix);
  REQUIRE(source->host_read(0, prefix, resident.data()) == prefix);
  CHECK(std::equal(resident.begin(), resident.end(), base_object().begin()));
  REQUIRE(server.get_count() == 1);
  REQUIRE(source->host_read(0, prefix, resident.data()) == prefix);
  REQUIRE(server.get_count() == 1);
  auto const footprint = cache.claimed_bytes();
  REQUIRE(footprint == chunk_bytes);

  // [0, prefix) is resident, [prefix, 2 * prefix) is a miss that now fails.
  std::vector<std::uint8_t> missing(prefix);
  std::array<cucascade::io::slice, 2> ranges{cucascade::io::slice{0, prefix, resident.data()},
                                             cucascade::io::slice{prefix, prefix, missing.data()}};
  for (std::size_t attempt = 1; attempt <= 2; ++attempt) {
    auto failure = capture_failure([&] { source->host_read_ranges_async(ranges).get(); });
    require_changed(failure, object_uri, expected);
    CHECK(server.get_count() == 1 + attempt);
    CHECK(cache.claimed_bytes() == footprint);
  }
  // The failed reads left the resident prefix intact and still readable.
  auto const after = server.get_count();
  REQUIRE(source->host_read(0, prefix, resident.data()) == prefix);
  CHECK(std::equal(resident.begin(), resident.end(), base_object().begin()));
  CHECK(server.get_count() == after);
}

TEST_CASE("cache identity publishes prefetch failure before notifying the consumer",
          "[cache][cache_identity][rest]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"].heads = {scripted_response{.etag = version_tag(1)},
                                 scripted_response{.etag = version_tag(2)}};
  scripts["object.bin"].gets  = {scripted_response{.status = 412}};
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  std::shared_ptr<datasource> source = fixture.open(object_uri);
  auto const expected                = std::string(source->get_io_object().validation_tag());
  REQUIRE(rest_io_object::is_strong_tag(expected));
  advise(*source, source->size());
  REQUIRE_FALSE(static_cast<bool>(source->prefetch_failure()));
  auto const footprint = cache.claimed_bytes();
  REQUIRE(footprint == chunk_bytes);
  // Another open sees the replacement, retiring the first open's generation.
  auto current = fixture.open(object_uri);
  REQUIRE(current->get_io_object().validation_tag() != expected);

  struct outcome {
    bool ok;
    std::exception_ptr failure;
  };
  struct observer {
    std::atomic<std::size_t> calls{0};
    std::promise<outcome> completed;
  };
  auto watcher   = std::make_shared<observer>();
  auto completed = watcher->completed.get_future();
  REQUIRE(source->prefetch_async([watcher, source](bool ok) noexcept {
    // Read from the completion callback itself: the failure must already be there.
    std::exception_ptr failure = source->prefetch_failure();
    if (watcher->calls.fetch_add(1) == 0) {
      watcher->completed.set_value(outcome{ok, std::move(failure)});
    }
  }) == cucascade::io::prefetch_refusal::issued);
  REQUIRE(completed.wait_for(5s) == std::future_status::ready);
  auto result = completed.get();
  CHECK_FALSE(result.ok);
  require_changed(result.failure, object_uri, expected);
  CHECK(source->prefetch_failure() == result.failure);
  CHECK(watcher->calls.load() == 1);
  CHECK(server.get_count() == 1);
  CHECK(cache.claimed_bytes() == footprint);

  // A demand read then fails the same way instead of returning replacement bytes.
  std::vector<std::uint8_t> bytes(source->size());
  auto failure = capture_failure([&] { source->host_read(0, bytes.size(), bytes.data()); });
  require_changed(failure, object_uri, expected);
  CHECK(server.get_count() == 2);
  CHECK(cache.claimed_bytes() == footprint);
  CHECK(watcher->calls.load() == 1);
}

TEST_CASE("cache identity releases a retired footer stash with its last datasource",
          "[cache][cache_identity][rest]")
{
  // The first open probes the footer instead of sending a HEAD: its suffix GET
  // (version 1) is the first GET, the fill after it the second.  The overwritten
  // object is then opened with a plain HEAD (version 2) and filled by the third.
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["object.bin"].heads = {scripted_response{.etag = version_tag(2)}};
  scripts["object.bin"].gets  = {scripted_response{.etag = version_tag(1), .body_xor = 1},
                                 scripted_response{.etag = version_tag(1), .body_xor = 1},
                                 scripted_response{.etag = version_tag(2), .body_xor = 2}};
  loopback_range_server server(base_object(), {}, {}, std::move(scripts));
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();

  // The probe's answer gives the first open a stash of trailing bytes and a
  // strong tag.
  auto first = cucascade::io::open_datasource(
    fixture.context, object_uri, cucascade::io::open_hint::parquet_footer_probe);
  REQUIRE(first != nullptr);
  REQUIRE(first->size() == object_bytes);
  auto const& first_object = dynamic_cast<rest_io_object const&>(first->get_io_object());
  REQUIRE(rest_io_object::is_strong_tag(first_object.validation_tag()));
  REQUIRE(first_object.stash() != nullptr);
  std::weak_ptr<cucascade::io::io_object const> object_alive = first_object.shared_from_this();
  std::weak_ptr<std::span<std::uint8_t const> const> stash_alive{first_object.stash()};
  auto const first_tag = std::string(first_object.validation_tag());

  // Cache the object and leave the evictor holding the request that names it.
  advise(*first, object_bytes);
  CHECK(read_all(*first) == version_bytes(1));
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (cache.eviction_batch_size_for_testing() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  REQUIRE(cache.eviction_batch_size_for_testing() > 0);

  // Overwrite the key: version 2's open retires the first generation.
  auto second = fixture.open_resident(server, object_uri, version_bytes(2));
  REQUIRE(second->get_io_object().validation_tag() != first_tag);
  require_generations(cache, object_uri, 2, 1);
  // The retired generation is alive, and so is the object it was created from.
  REQUIRE_FALSE(object_alive.expired());
  REQUIRE_FALSE(stash_alive.expired());

  // Dropping the last datasource frees the stash and the object even though the
  // evictor still holds a copy of the request: that copy keeps neither.
  first.reset();
  CHECK(object_alive.expired());
  CHECK(stash_alive.expired());
  require_generations(cache, object_uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

// ===========================================================================
// Held backend: lifetime and ordering, with every fill under test control
// ===========================================================================

TEST_CASE("cache identity retains a retired generation until a held fill releases",
          "[cache][cache_identity]")
{
  auto memory  = identity_memory();
  auto context = std::make_shared<held_context>();
  context->initialize_cache(*memory, identity_cache_config(), nullptr);
  REQUIRE(context->cache() != nullptr);
  auto& cache            = *context->cache();
  std::string const path = "s3://controlled/generation.bin";
  auto first             = std::make_unique<datasource>(context, held_object(path, "\"one\""));
  auto second            = std::make_unique<datasource>(context, held_object(path, "\"two\""));
  std::atomic<int> first_completions{0};
  std::atomic<int> second_completions{0};
  std::atomic<bool> second_ok{false};
  held_request old_fill;
  held_request new_fill;

  advise(*first, chunk_bytes);
  REQUIRE(first->prefetch_async([&](bool) noexcept { first_completions.fetch_add(1); }) ==
          cucascade::io::prefetch_refusal::issued);
  old_fill.request = context->backend().take();
  REQUIRE(old_fill.request != nullptr);
  require_generations(cache, path, 1, 0);

  // A second generation of the path supersedes the first while its fill is held.
  advise(*second, chunk_bytes);
  REQUIRE(second->prefetch_async([&](bool ok) noexcept {
    second_ok.store(ok);
    second_completions.fetch_add(1);
  }) == cucascade::io::prefetch_refusal::issued);
  new_fill.request = context->backend().take();
  REQUIRE(new_fill.request != nullptr);
  new_fill.complete(0x22);
  CHECK(second_ok.load());
  CHECK(second_completions.load() == 1);
  require_generations(cache, path, 2, 1);

  // The datasource is gone but its fill is still in flight: the in-flight
  // terminal keeps the retired generation alive.
  first.reset();
  require_generations(cache, path, 2, 1);
  CHECK(first_completions.load() == 0);

  // The fill lands on the retired generation, which is then reclaimed.
  old_fill.complete(0x11);
  CHECK(first_completions.load() == 1);
  require_generations(cache, path, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);

  // The current generation kept its own bytes throughout.
  auto bytes = read_all(*second);
  CHECK(std::ranges::all_of(bytes, [](auto value) { return value == 0x22; }));
  second.reset();
  context->shutdown_cache();
}

TEST_CASE("cache identity delivers a held fill's failure through the datasource",
          "[cache][cache_identity]")
{
  auto memory  = identity_memory();
  auto context = std::make_shared<held_context>();
  context->initialize_cache(*memory, identity_cache_config(), nullptr);
  REQUIRE(context->cache() != nullptr);
  auto& cache            = *context->cache();
  std::string const path = "s3://controlled/generation.bin";
  std::shared_ptr<datasource> source =
    std::make_shared<datasource>(context, held_object(path, "\"one\""));

  // Before any prefetch there is no failure to report.
  CHECK_FALSE(static_cast<bool>(source->prefetch_failure()));
  advise(*source, chunk_bytes);
  CHECK_FALSE(static_cast<bool>(source->prefetch_failure()));

  std::exception_ptr seen;
  bool ok_seen = true;
  std::atomic<int> calls{0};
  REQUIRE(source->prefetch_async([&, source](bool ok) noexcept {
    ok_seen = ok;
    seen    = source->prefetch_failure();
    calls.fetch_add(1);
  }) == cucascade::io::prefetch_refusal::issued);
  held_request fill;
  fill.request = context->backend().take();
  REQUIRE(fill.request != nullptr);
  fill.fail(path, "\"one\"", "\"two\"");

  CHECK(calls.load() == 1);
  CHECK_FALSE(ok_seen);
  // The callback saw the failure already published, with its payload intact.
  require_changed(seen, path, "\"one\"", "\"two\"");
  CHECK(source->prefetch_failure() == seen);
  CHECK(cache.claimed_bytes() == chunk_bytes);
  source.reset();
  context->shutdown_cache();
}

// ===========================================================================
// Summary snapshots
// ===========================================================================

TEST_CASE("cache summary snapshots stay consistent under concurrent queries",
          "[cache][cache_summary][rest]")
{
  range_fault_policy fault;
  fault.successful_head_etag = "\"generation-one\"";
  fault.successful_get_etag  = fault.successful_head_etag;
  loopback_range_server server(base_object(), fault);
  rest_cache_fixture fixture(server);
  auto& cache = *fixture.context->cache();
  auto source = fixture.open_resident(server, object_uri, base_object());

  auto const before = summary_counts(cache.summary());
  REQUIRE(before.size() == 10);

  // Embedders call prepare_for_query() once per query and summary() whenever
  // they report, from concurrent connections, while reads keep moving the
  // counters.  Both touch the per-cycle snapshot.  Each summary must read the
  // counters and that snapshot as one unit: a snapshot taken after the counters
  // were read would put a cycle delta above its running total.
  constexpr int iterations = 100000;
  std::atomic<bool> readers_done{false};
  std::atomic<bool> bad_snapshot{false};
  std::atomic<bool> malformed{false};

  std::thread reader([&] {
    std::vector<std::uint8_t> bytes(source->size());
    while (!readers_done.load()) {
      std::ignore = source->host_read(0, bytes.size(), bytes.data());
    }
  });

  auto const query_cycle = [&] {
    for (int i = 0; i < iterations; ++i) {
      cache.prepare_for_query();
    }
  };
  auto const report = [&] {
    for (int i = 0; i < iterations; ++i) {
      auto const counts = summary_counts(cache.summary());
      if (counts.size() != 10) {
        malformed.store(true);
        continue;
      }
      for (std::size_t field = 0; field < 5; ++field) {
        if (counts[5 + field] > counts[field]) { bad_snapshot.store(true); }
      }
    }
  };

  std::vector<std::thread> workers;
  workers.emplace_back(query_cycle);
  workers.emplace_back(query_cycle);
  workers.emplace_back(report);
  workers.emplace_back(report);
  for (auto& worker : workers) {
    worker.join();
  }
  readers_done.store(true);
  reader.join();

  CHECK_FALSE(malformed.load());
  CHECK_FALSE(bad_snapshot.load());
  // The reader really was moving the counters while the snapshots were taken.
  auto const after = summary_counts(cache.summary());
  REQUIRE(after.size() == 10);
  CHECK(after[1] > before[1]);
  CHECK(server.get_count() == 1);
}

// ===========================================================================
// Datasource cache query
// ===========================================================================

TEST_CASE("a datasource reports whether its ioctx reads through an fs_cache",
          "[cache][cache_identity]")
{
  auto memory  = identity_memory();
  auto context = std::make_shared<held_context>();
  datasource const source(context, held_object("s3://controlled/uses-cache.bin", "\"one\""));

  // The query is part of the public const surface, and follows the ioctx's cache
  // as it is initialized and shut down.
  CHECK_FALSE(source.uses_fs_cache());
  context->initialize_cache(*memory, identity_cache_config(), nullptr);
  REQUIRE(context->cache() != nullptr);
  CHECK(source.uses_fs_cache());
  context->shutdown_cache();
  CHECK_FALSE(source.uses_fs_cache());
}
