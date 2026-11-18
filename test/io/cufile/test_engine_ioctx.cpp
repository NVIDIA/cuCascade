/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "engine_test_utils.hpp"

#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/runtime.hpp>

#include <pthread.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;
namespace cufile_detail = cucascade::io::cufile::detail;

namespace {
async_engine_config api_config()
{
  auto cfg            = configuration();
  cfg.mount_overrides = {
    {mount_match::mount_prefix, "/", mount_override_state::operator_qualified, 4, mib}};
  return cfg;
}
struct pattern_file {
  std::string path;
  explicit pattern_file(std::vector<std::byte> const& image)
  {
    static std::atomic<unsigned> next{};
    path = (std::filesystem::temp_directory_path() /
            ("cucascade_engine_api_" + std::to_string(getpid()) + "_" +
             std::to_string(next.fetch_add(1))))
             .string();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(image.data()),
              static_cast<std::streamsize>(image.size()));
    REQUIRE(out);
  }
  ~pattern_file()
  {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};
struct api_fixture {
  fixture resources{4 * mib, api_config()};
  pattern_file disk{resources.image};
  std::shared_ptr<cufile_reactor::reactor_context> shared;
  std::unique_ptr<cufile_ioctx> context;
  std::shared_ptr<io_object> object;

  explicit api_fixture(bool legacy                = false,
                       bool posix                 = false,
                       mount_override_state state = mount_override_state::operator_qualified,
                       bool probe_only            = false)
  {
    config cfg;
    cfg.async_engine                               = resources.config;
    cfg.async_engine.mount_overrides.front().state = state;
    if (probe_only) {
      cfg.async_engine.mount_overrides.clear();
      cfg.async_engine.probe.require_direct_evidence = false;
      cfg.async_engine.probe.bytes                   = mib;
      cfg.async_engine.probe.submit_max_us           = 100000;
    }
    if (legacy) cfg.async_engine.mode = engine_mode::legacy;
    cfg.use_odirect = !posix;
    shared          = std::make_shared<cufile_reactor::reactor_context>(cfg);
    context         = std::make_unique<cufile_ioctx>(1, shared);
    object          = context->open_io_object(disk.path);
    if (!posix && !static_cast<cufile_io_object const&>(*object).cufile_registered())
      SKIP("public engine routing requires a registered file (compat libcufile is sufficient)");
    resources.start();
    auto binding       = std::make_shared<cufile_reactor::reactor_context::engine_binding>();
    binding->engine    = resources.engine.get();
    binding->arena     = resources.arena.get();
    binding->transport = resources.transport;
    binding->simulated = true;
    // Isolate first-configuration-wins policies between simulator tests.
    static std::atomic<std::uint64_t> generation{1000000};
    binding->generation  = generation.fetch_add(1);
    binding->device_uuid = "ioctx-simulator-0";
    shared->install_engine(0, std::move(binding));
    // Most tests exercise warm routing. Cold-file discovery is covered separately below.
    if (!legacy && !posix) {
      auto const& file = static_cast<cufile_io_object const&>(*object);
      auto installed   = shared->engine_for_device(0);
      static_cast<void>(mount_policy_registry::instance().get_or_create(
        file.policy_key(installed->device_uuid, installed->generation),
        file.mount_identity_info(),
        cfg.async_engine));
    }
    context->start();
  }
  auto read(cudaStream_t stream,
            std::size_t offset     = 0,
            std::size_t size       = mib,
            std::size_t dst_offset = 0)
  {
    slice request{offset, size, static_cast<std::uint8_t*>(resources.destination) + dst_offset};
    return context->device_readv_async_io(
      *object, std::span{&request, 1}, ::cuda::stream_ref{stream});
  }
  ~api_fixture()
  {
    if (context) context->shutdown();
  }
};
struct seam_reset {
  ~seam_reset()
  {
    cufile_detail::set_staging_copy_delay_for_testing(0us);
    cufile_detail::set_copy_event_record_failure_for_testing(false);
    cufile_detail::set_completion_observer_for_testing(nullptr, nullptr);
    cufile_detail::set_cufile_transfer_hook_for_testing(nullptr, nullptr);
  }
};
}  // namespace

TEST_CASE("ioctx engine vectored reads join exact bytes onto the caller stream",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  std::vector<slice> slices{
    slice{37, mib - 37, static_cast<std::uint8_t*>(r.destination)},
    slice{2 * mib + 11, mib - 11, static_cast<std::uint8_t*>(r.destination) + mib}};
  auto future = f.context->device_readv_async_io(*f.object, slices, ::cuda::stream_ref{r.stream});
  REQUIRE(std::move(future).get(5s) == 2 * mib - 48);
  // No preceding synchronization: consumer work queued after the future must see the copies.
  auto contents = r.contents(2 * mib);
  CHECK(std::equal(contents.begin(),
                   contents.begin() + static_cast<std::ptrdiff_t>(mib - 37),
                   r.image.begin() + 37));
  CHECK(std::equal(contents.begin() + static_cast<std::ptrdiff_t>(mib),
                   contents.begin() + static_cast<std::ptrdiff_t>(2 * mib - 11),
                   r.image.begin() + static_cast<std::ptrdiff_t>(2 * mib + 11)));
  CHECK(r.transport->snapshot().submitted >= 2);
}

TEST_CASE("ioctx tiny columns publish together across a multi-reactor context",
          "[cufile][engine][coalesce][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  cufile_ioctx context(4, f.shared);
  context.start();
  auto object = context.open_io_object(f.disk.path);
  // Coalescing consults worker-populated metadata only; a cold file safely fans out.
  static_cast<void>(static_cast<cufile_io_object const&>(*object).mount_identity_info());
  std::vector<slice> slices;
  auto expected = std::vector<std::byte>(r.bytes, std::byte{0xcc});
  for (std::size_t i = 100; i != 0; --i) {
    auto offset = (i - 1) * 4096 + 37;
    auto dest   = (100 - i) * 8192;
    slices.emplace_back(offset, 1024, static_cast<std::uint8_t*>(r.destination) + dest);
    std::copy_n(r.image.begin() + static_cast<std::ptrdiff_t>(offset),
                1024,
                expected.begin() + static_cast<std::ptrdiff_t>(dest));
  }
  CHECK(context.device_readv_async_io(*object, slices, ::cuda::stream_ref{r.stream}).get(5s) ==
        102400);
  CHECK(r.contents() == expected);
  REQUIRE(r.engine->drain(5s));
  CHECK(r.transport->snapshot().read_calls == 2);
  CHECK(r.transport->snapshot().wait_calls == 4);  // one fence + one join per group
  context.shutdown();
}

TEST_CASE("ioctx engine success is enqueued rather than host complete", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  gate producer{r.stream};
  auto future = f.read(r.stream);
  REQUIRE(std::move(future).get(5s) == mib);
  CHECK(cudaStreamQuery(r.stream) == cudaErrorNotReady);
  CHECK(r.transport->snapshot().completed == 1);  // AM-2: I/O bypassed the busy input fence.
  producer.release();
  auto contents = r.contents(mib);
  CHECK(std::equal(contents.begin(), contents.end(), r.image.begin()));
}

TEST_CASE("ioctx engine probes without blocking legacy admission and honors test evidence",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f(false, false, mount_override_state::operator_qualified, true);
  auto& r = f.resources;
  transport_simulator::fault fault;
  fault.index            = 0;
  fault.stuck_completion = true;
  r.transport->add_fault(fault);
  auto binding     = f.shared->engine_for_device(0);
  auto const& file = static_cast<cufile_io_object const&>(*f.object);
  auto policy      = mount_policy_registry::instance().get_or_create(
    file.policy_key(binding->device_uuid, binding->generation),
    file.mount_identity_info(),
    f.shared->cfg().async_engine);
  REQUIRE(cudaStreamSynchronize(r.stream) == cudaSuccess);
  auto first = f.read(r.stream);
  REQUIRE(until([&] { return r.transport->snapshot().submitted == 1; }));
  CHECK(policy->state() == mount_state::probing);
  // Submission returned while the probe is gated. Real compat cuFileRead may
  // itself synchronize CUDA work, so do not demand completion behind an infinite GPU gate.
  REQUIRE(r.transport->unblock(0));
  REQUIRE(std::move(first).get(5s) == mib);
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
  REQUIRE(until([&] { return policy->read_eligible(); }));
  CHECK(policy->state() == mount_state::operator_qualified);  // Never reported as verified.
  REQUIRE(f.read(r.stream).get(5s) == mib);
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
  CHECK(r.transport->snapshot().submitted == 2);
}

TEST_CASE("ioctx engine captures the submit fence before worker dispatch",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  gate producer{r.stream};
  REQUIRE(cudaMemsetAsync(r.destination, 0xab, mib, r.stream) == cudaSuccess);
  auto future = f.read(r.stream);
  REQUIRE(until([&] { return r.transport->snapshot().completed == 1; }));
  producer.release();
  REQUIRE(std::move(future).get(5s) == mib);
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
}

TEST_CASE("ioctx engine PTDS reads remain host complete", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  seam_reset reset;
  cufile_detail::set_staging_copy_delay_for_testing(50ms);
  // A finite producer is important: CUDA may serialize stand-in stream creation
  // behind PTDS work after real cuFile stream registration (the existing resolver).
  REQUIRE(cudaLaunchHostFunc(
            cudaStreamPerThread, [](void*) { std::this_thread::sleep_for(30ms); }, nullptr) ==
          cudaSuccess);
  REQUIRE(cudaMemsetAsync(r.destination, 0xab, mib, cudaStreamPerThread) == cudaSuccess);
  auto future = f.read(cudaStreamPerThread);
  REQUIRE(until([&] { return r.transport->snapshot().completed == 1; }));
  CHECK_FALSE(future.is_ready());
  REQUIRE(std::move(future).get(5s) == mib);
  // A foreign nonblocking stream may consume immediately for original PTDS callers.
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
}

TEST_CASE("ioctx engine null streams and disabled routes retain legacy completion",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const route = GENERATE(0, 1, 2, 3, 4);
  CAPTURE(route);
  api_fixture f(route == 1,
                route == 2,
                route == 3   ? mount_override_state::known_compat
                : route == 4 ? mount_override_state::legacy
                             : mount_override_state::operator_qualified);
  auto& r     = f.resources;
  auto stream = route == 0 ? cudaStream_t{nullptr} : r.stream;
  REQUIRE(f.read(stream).get(5s) == mib);
  CHECK(r.transport->snapshot().submitted == 0);
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
}

TEST_CASE("ioctx engine stopped bindings retain legacy routing", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  r.engine->request_stop();
  REQUIRE(r.engine->drain(5s));
  CHECK_FALSE(r.engine->ready());
  REQUIRE(f.read(r.stream).get(5s) == mib);
  CHECK(r.transport->snapshot().submitted == 0);
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
}

TEST_CASE("ioctx engine mixed requests keep host slices and pairs on legacy",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  std::vector<std::uint8_t> host(mib), pair(mib);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, mib}, host_buffer{host.data()});
  slices.emplace_back(
    range{mib, mib},
    device_buffer{static_cast<std::uint8_t*>(r.destination), ::cuda::stream_ref{r.stream}, 0});
  slices.emplace_back(
    range{2 * mib, mib},
    host_buffer{pair.data()},
    device_buffer{
      static_cast<std::uint8_t*>(r.destination) + mib, ::cuda::stream_ref{r.stream}, 0});
  REQUIRE(f.context->mixed_readv_async_io(*f.object, std::move(slices)).get(5s) == 3 * mib);
  REQUIRE(cudaStreamSynchronize(r.stream) == cudaSuccess);
  CHECK(std::memcmp(host.data(), r.image.data(), mib) == 0);
  CHECK(std::memcmp(pair.data(), r.image.data() + 2 * mib, mib) == 0);
  auto contents = r.contents(2 * mib);
  CHECK(std::memcmp(contents.data(), r.image.data() + mib, 2 * mib) == 0);
  CHECK(r.transport->snapshot().submitted == 1);
}

TEST_CASE("ioctx engine busy user stream does not block other requests", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  gate producer{r.stream};
  auto blocked = f.read(r.stream);
  auto ready   = f.read(other, mib, mib, mib);
  REQUIRE(std::move(ready).get(5s) == mib);
  REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
  CHECK(cudaStreamQuery(r.stream) == cudaErrorNotReady);
  producer.release();
  REQUIRE(std::move(blocked).get(5s) == mib);
  auto contents = r.contents(2 * mib);
  CHECK(std::equal(contents.begin(), contents.end(), r.image.begin()));
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
}

TEST_CASE("ioctx engine facade shutdown waits for retirements not early futures",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  gate producer{r.stream};
  REQUIRE(f.read(r.stream).get(5s) == mib);
  auto stopped = std::async(std::launch::async, [&] { f.context->shutdown(); });
  CHECK(stopped.wait_for(30ms) == std::future_status::timeout);
  producer.release();
  REQUIRE(stopped.wait_for(5s) == std::future_status::ready);
  stopped.get();
  // The shared engine remains alive and accepts another facade's read.
  cufile_ioctx second{1, f.shared};
  second.start();
  auto object = second.open_io_object(f.disk.path);
  slice request{0, mib, static_cast<std::uint8_t*>(r.destination)};
  REQUIRE(
    second.device_readv_async_io(*object, std::span{&request, 1}, ::cuda::stream_ref{r.stream})
      .get(5s) == mib);
  second.shutdown();
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
}

TEST_CASE("ioctx engine cancellation is scoped to one facade", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  cufile_ioctx second{1, f.shared};
  second.start();
  f.context->shutdown();
  slice request{0, mib, static_cast<std::uint8_t*>(r.destination)};
  REQUIRE(
    second.device_readv_async_io(*f.object, std::span{&request, 1}, ::cuda::stream_ref{r.stream})
      .get(5s) == mib);
  second.shutdown();
  CHECK(r.transport->snapshot().submitted == 1);
}

TEST_CASE("ioctx engine preserves transfer completion delay and failure seams",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto const fail = GENERATE(false, true);
  seam_reset reset;
  std::atomic<unsigned> transfers{}, validated{};
  cufile_detail::set_cufile_transfer_hook_for_testing(
    [](void* value) { static_cast<std::atomic<unsigned>*>(value)->fetch_add(1); }, &transfers);
  cufile_detail::set_completion_observer_for_testing(
    [](cufile_detail::completion_observation const& result, void* value) noexcept {
      if (result.outcome == cufile_detail::completion_outcome::validated)
        static_cast<std::atomic<unsigned>*>(value)->fetch_add(1);
    },
    &validated);
  cufile_detail::set_staging_copy_delay_for_testing(50ms);
  cufile_detail::set_copy_event_record_failure_for_testing(fail);
  auto future = f.read(f.resources.stream);
  if (fail)
    CHECK_THROWS(std::move(future).get(5s));
  else
    REQUIRE(std::move(future).get(5s) == mib);
  REQUIRE(cudaStreamSynchronize(f.resources.stream) == cudaSuccess);
  CHECK(transfers == 1);
  CHECK(validated == 1);
  if (fail) CHECK(f.resources.engine->gauges().recoveries >= 1);
}

TEST_CASE("ioctx engine facade cancels its pending slices and drains submitted work",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  transport_simulator::fault fault;
  fault.stuck_completion = true;
  r.transport->add_fault(fault);
  std::vector<slice> slices;
  constexpr std::size_t chunk = 64UL << 10;
  for (std::size_t offset = 0; offset < r.bytes; offset += chunk)
    slices.emplace_back(offset, chunk, static_cast<std::uint8_t*>(r.destination) + offset);
  auto future = f.context->device_readv_async_io(*f.object, slices, ::cuda::stream_ref{r.stream});
  REQUIRE(until([&] { return r.transport->snapshot().submitted >= 4; }));
  auto stopped = std::async(std::launch::async, [&] { f.context->shutdown(); });
  CHECK(stopped.wait_for(30ms) == std::future_status::timeout);
  r.transport->clear_faults();
  r.transport->unblock_all();
  REQUIRE(stopped.wait_for(5s) == std::future_status::ready);
  stopped.get();
  CHECK_THROWS(std::move(future).get(5s));
  CHECK(r.transport->snapshot().submitted < slices.size());
  cufile_ioctx second{1, f.shared};
  second.start();
  slice request{0, mib, static_cast<std::uint8_t*>(r.destination)};
  REQUIRE(
    second.device_readv_async_io(*f.object, std::span{&request, 1}, ::cuda::stream_ref{r.stream})
      .get(5s) == mib);
  second.shutdown();
}

TEST_CASE("ioctx engine cancels queued work behind another facade's stuck submit",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  transport_simulator::fault fault;
  fault.index        = 0;
  fault.stuck_submit = true;
  r.transport->add_fault(fault);
  auto active = f.read(r.stream);
  REQUIRE(until([&] { return r.transport->snapshot().blocked_submissions == 1; }));
  cufile_ioctx second{1, f.shared};
  second.start();
  slice request{mib, mib, static_cast<std::uint8_t*>(r.destination) + mib};
  auto pending =
    second.device_readv_async_io(*f.object, std::span{&request, 1}, ::cuda::stream_ref{r.stream});
  REQUIRE(until([&] { return r.engine->gauges().storage_qd == 2; }));
  auto stopped      = std::async(std::launch::async, [&] { second.shutdown(); });
  auto const prompt = stopped.wait_for(100ms) == std::future_status::ready;
  CHECK(prompt);
  CHECK_FALSE(active.is_ready());
  r.transport->unblock_all();
  REQUIRE(stopped.wait_for(5s) == std::future_status::ready);
  stopped.get();
  CHECK_THROWS(std::move(pending).get(5s));
  REQUIRE(std::move(active).get(5s) == mib);
  CHECK(r.transport->snapshot().read_calls == 1);
}

TEST_CASE("ioctx engine errors wait for already-enqueued sibling destination copies",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  transport_simulator::fault fault;
  fault.index             = 1;
  fault.stuck_completion  = true;
  fault.completion_result = -ENOSPC;
  r.transport->add_fault(fault);
  gate producer{r.stream};
  std::vector<slice> slices;
  slices.emplace_back(0, mib, static_cast<std::uint8_t*>(r.destination));
  slices.emplace_back(mib, mib, static_cast<std::uint8_t*>(r.destination) + mib);
  // The second slice is input-ready; the first consumes the mount's gated-work allowance.
  auto future = f.context->mixed_readv_async_io(*f.object, [&] {
    std::vector<prepared_io_slice> prepared;
    prepared.emplace_back(
      range{0, mib},
      device_buffer{static_cast<std::uint8_t*>(r.destination), ::cuda::stream_ref{r.stream}, 0});
    prepared.emplace_back(
      range{mib, mib},
      device_buffer{
        static_cast<std::uint8_t*>(r.destination) + mib, ::cuda::stream_ref{cudaStreamLegacy}, 0});
    return prepared;
  }());
  REQUIRE(until([&] { return r.transport->snapshot().copy_calls == 1; }));
  REQUIRE(r.transport->unblock(1));
  REQUIRE(until([&] { return r.transport->snapshot().completed == 2; }));
  std::this_thread::sleep_for(5ms);
  CHECK_FALSE(future.is_ready());
  producer.release();
  CHECK_THROWS(std::move(future).get(5s));
  CHECK(cudaStreamQuery(r.stream) == cudaSuccess);
}

TEST_CASE("cold caller fences do not discover file metadata or create policies",
          "[cufile][engine][ioctx][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto cold        = f.context->open_io_object(f.disk.path);
  auto const& file = static_cast<cufile_io_object const&>(*cold);
  REQUIRE_FALSE(file.cached_mount_identity());
  cufile_reactor recorder{f.shared};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, mib},
                      device_buffer{static_cast<std::uint8_t*>(f.resources.destination),
                                    ::cuda::stream_ref{f.resources.stream},
                                    0});
  CHECK_NOTHROW(recorder.record_submit_fences(file, slices));
  CHECK(slices.front().d_buffer.fence);
  CHECK_FALSE(file.cached_mount_identity());
  CHECK_FALSE(recorder.coalescing_candidate(file, slices));
  CHECK_FALSE(file.cached_mount_identity());
}

TEST_CASE("ioctx skips caller fence records for known compat legacy and non-routable files",
          "[cufile][engine][ioctx][hardening]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto state = GENERATE(mount_override_state::known_compat,
                        mount_override_state::legacy,
                        mount_override_state::operator_qualified);
  api_fixture f(false, false, state);
  cufile_reactor recorder{f.shared};
  auto const before = cufile_detail::submit_fence_records_for_testing();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, mib},
                      device_buffer{static_cast<std::uint8_t*>(f.resources.destination),
                                    ::cuda::stream_ref{f.resources.stream},
                                    0});
  recorder.record_submit_fences(static_cast<cufile_io_object const&>(*f.object), slices);
  CHECK(cufile_detail::submit_fence_records_for_testing() - before ==
        (state == mount_override_state::operator_qualified ? 1 : 0));
  cufile_io_object unroutable("unregistered", file_descriptor{}, file_descriptor{}, nullptr, mib);
  slices[0].d_buffer.fence.reset();
  auto const previous = cufile_detail::submit_fence_records_for_testing();
  recorder.record_submit_fences(unroutable, slices);
  CHECK(cufile_detail::submit_fence_records_for_testing() == previous);
  CHECK_FALSE(slices[0].d_buffer.fence);
}

TEST_CASE("ioctx engine shares one caller fence per device and resolved stream",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  cufile_reactor recorder{f.shared};
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  std::vector<prepared_io_slice> slices;
  for (auto stream : {r.stream, r.stream, other})
    slices.emplace_back(
      range{0, mib},
      device_buffer{static_cast<std::uint8_t*>(r.destination), ::cuda::stream_ref{stream}, 0});
  REQUIRE(recorder.submit_fence_enabled());
  recorder.record_submit_fences(static_cast<cufile_io_object const&>(*f.object), slices);
  REQUIRE(slices[0].d_buffer.fence);
  CHECK(slices[0].d_buffer.fence == slices[1].d_buffer.fence);
  CHECK(slices[0].d_buffer.fence != slices[2].d_buffer.fence);
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
}

TEST_CASE("ioctx real compat engine uses persistent slot registrations", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto cfg    = api_config();
  auto& rt    = runtime::instance();
  std::ignore = rt.ensure_initialized();
  if (!rt.engine_possible(cfg.mode)) SKIP("compat async libcufile unavailable");
  auto image = pattern(4 * mib);
  pattern_file file{image};
  fixture storage{4 * mib};
  config reactor_cfg;
  reactor_cfg.async_engine = cfg;
  auto shared              = std::make_shared<cufile_reactor::reactor_context>(reactor_cfg);
  cufile_ioctx context{1, shared};
  context.start();
  CHECK_FALSE(shared->engine_for_device(0));  // No eager all-device warm-up at start.
  auto object = context.open_io_object(file.path);
  if (!static_cast<cufile_io_object const&>(*object).cufile_registered())
    SKIP("file cannot register with libcufile");
  shared->ensure_engine(0);
  REQUIRE(until([&] { return shared->engine_for_device(0) != nullptr; }, 15s));
  auto before = shared->engine_for_device(0)->engine->gauges().async_registered_bytes;
  std::vector<slice> slices;
  for (std::size_t i = 0; i < 4; ++i)
    slices.push_back(
      slice{i * mib, mib, static_cast<std::uint8_t*>(storage.destination) + i * mib});
  REQUIRE(
    context.device_readv_async_io(*object, slices, ::cuda::stream_ref{storage.stream}).get(15s) ==
    image.size());
  CHECK(storage.contents() == image);
  CHECK(until([&] {
    return shared->engine_for_device(0)->engine->gauges().async_registered_bytes >=
           before + image.size();
  }));
  auto registered = cufile_detail::persistent_registered_bytes_for_testing();
  CHECK(registered >= device_arena::slab_bytes);
  CHECK(cufile_detail::arena_slab_counts_for_testing() >= 1);
  context.shutdown();
  CHECK(cufile_detail::persistent_registered_bytes_for_testing() == registered);
  // Only persistent arena and staging registrations exist after the merge.
  CHECK(cufile_detail::arena_slab_counts_for_testing() >= 1);
}

TEST_CASE("ioctx engine delayed copy is joined before an immediate consumer",
          "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  seam_reset reset;
  REQUIRE(cudaStreamSynchronize(r.stream) == cudaSuccess);
  cufile_detail::set_staging_copy_delay_for_testing(200ms);
  auto targets = GENERATE(1U, 8U);
  std::vector<slice> slices;
  for (unsigned i = 0; i < targets; ++i)
    slices.emplace_back(i * 4096, 1024, static_cast<std::uint8_t*>(r.destination) + i * 8192);
  auto future = f.context->device_readv_async_io(*f.object, slices, ::cuda::stream_ref{r.stream});
  // This is intentionally before the copy can execute, not a host-complete read.
  REQUIRE(std::move(future).get(5s) == targets * 1024);
  // Synchronize ONLY the caller stream: without the join this reads the 0xcc poison.
  auto expected = std::vector<std::byte>(mib, std::byte{0xcc});
  for (unsigned i = 0; i < targets; ++i)
    std::copy_n(r.image.begin() + i * 4096, 1024, expected.begin() + i * 8192);
  CHECK(r.contents(mib) == expected);
  REQUIRE(r.engine->drain(5s));
  CHECK(r.engine->gauges().coalesced_targets == (targets > 1 ? targets : 0));
}

TEST_CASE("ioctx engine metadata pressure retains the legacy read path", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  std::size_t reserved{};
  while (r.engine->try_reserve_slice())
    ++reserved;
  struct release_reservations {
    device_engine& engine;
    std::size_t count;
    ~release_reservations()
    {
      while (count--)
        engine.release_slice_reservation();
    }
  } release{*r.engine, reserved};
  REQUIRE(reserved > 0);
  REQUIRE(f.read(r.stream).get(5s) == mib);
  CHECK(r.contents(mib) == std::vector<std::byte>(r.image.begin(), r.image.begin() + mib));
  CHECK(r.transport->snapshot().submitted == 0);
}

TEST_CASE("ioctx published engine lookup does not miss under contention", "[cufile][engine][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto expected = f.shared->engine_for_device(0);
  REQUIRE(expected);
  std::atomic<unsigned> misses{};
  std::vector<std::thread> readers;
  for (unsigned t = 0; t < 8; ++t)
    readers.emplace_back([&] {
      for (unsigned i = 0; i < 2000; ++i)
        if (f.shared->engine_for_device(0) != expected) ++misses;
    });
  for (auto& reader : readers)
    reader.join();
  CHECK(misses.load() == 0);
}

TEST_CASE("ioctx legacy dispatch publishes engine handoffs before blocking",
          "[cufile][engine][coalesce][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  api_fixture f;
  auto& r = f.resources;
  struct hold {
    std::atomic<bool> entered{}, released{};
    ~hold() { released.store(true); }
  } held;
  seam_reset reset;
  cufile_detail::set_cufile_transfer_hook_for_testing(
    [](void* value) {
      char name[16]{};
      static_cast<void>(pthread_getname_np(pthread_self(), name, sizeof(name)));
      if (std::string_view(name).find("mixed-pass") != 0) return;
      auto& state = *static_cast<hold*>(value);
      state.entered.store(true);
      while (!state.released.load())
        std::this_thread::sleep_for(50us);
    },
    &held);
  cufile_reactor worker(f.shared, "mixed-pass");
  worker.start();
  // Ensure assertion unwinding releases the held worker BEFORE its destructor joins.
  struct release {
    hold& state;
    ~release() { state.released.store(true); }
  } guard{held};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(
    range{0, 1024},
    device_buffer{static_cast<std::uint8_t*>(r.destination), ::cuda::stream_ref{r.stream}, 0});
  slices.emplace_back(
    range{mib, mib},
    device_buffer{
      static_cast<std::uint8_t*>(r.destination) + mib, ::cuda::stream_ref{cudaStreamLegacy}, 0});
  worker.record_submit_fences(static_cast<cufile_io_object const&>(*f.object), slices);
  slices[1].d_buffer.original_stream_was_null = true;  // Force synchronous legacy.
  REQUIRE(cudaStreamSynchronize(r.stream) == cudaSuccess);
  auto group  = std::make_shared<grouped_coordinator>(mib + 1024, 2);
  auto future = group->get_future();
  worker.enqueue(grouped_io_request::create(f.object, std::move(slices), group));
  REQUIRE(until([&] { return held.entered.load(); }));
  // The engine must run WHILE the legacy worker remains held in the same pass.
  REQUIRE(until([&] { return r.transport->snapshot().completed == 1; }));
  CHECK_FALSE(future.is_ready());
  held.released.store(true);
  REQUIRE(std::move(future).get(5s) == mib + 1024);
  worker.shutdown();
}

TEST_CASE("ioctx coalescing routing preserves legacy mixed and large fan-out",
          "[cufile][engine][coalesce][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto eligible = GENERATE(false, true);
  api_fixture f(false,
                false,
                eligible ? mount_override_state::operator_qualified : mount_override_state::legacy);
  struct inspect_context : cufile_ioctx {
    using cufile_ioctx::cufile_ioctx;
    using cufile_ioctx::select_reactors;
  } context(4, f.shared);
  auto& r = f.resources;
  cufile_reactor recorder(f.shared);
  std::vector<prepared_io_slice> slices;
  for (unsigned i = 0; i < 4; ++i)
    slices.emplace_back(
      range{i * mib, mib},
      device_buffer{
        static_cast<std::uint8_t*>(r.destination) + i * mib, ::cuda::stream_ref{r.stream}, 0});
  recorder.record_submit_fences(static_cast<cufile_io_object const&>(*f.object), slices);
  auto const& object = static_cast<cufile_io_object const&>(*f.object);
  using io_op_type   = cufile_ioctx::io_op_type;
  CHECK(context.select_reactors(object, slices, io_op_type::host_vector_async, 0).size() == 4);
  for (auto& slice : slices)
    slice.rng.size = 1024;
  CHECK(context.select_reactors(object, slices, io_op_type::host_vector_async, 0).size() ==
        (eligible ? 1 : 4));
  slices.back().d_buffer.original_stream_was_null = true;
  CHECK(context.select_reactors(object, slices, io_op_type::host_vector_async, 0).size() == 4);
  std::uint8_t dummy{};  // Routing-only: host_buffer still requires a non-null address.
  slices.back() = prepared_io_slice(range{0, 1024}, host_buffer{&dummy});
  CHECK(context.select_reactors(object, slices, io_op_type::host_vector_async, 0).size() == 4);
}
