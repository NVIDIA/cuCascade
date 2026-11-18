/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "transport_simulator.hpp"

#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_api.hpp>
#endif

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::io::cufile;
using namespace std::chrono_literals;
constexpr auto sentinel = std::numeric_limits<ssize_t>::min();

struct pointer_args {
  std::size_t size{16};
  off_t file_offset{0};
  off_t buffer_offset{0};
  ssize_t bytes_done{sentinel};
};

bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count != 0;
}

std::vector<std::byte> pattern(std::size_t size)
{
  std::vector<std::byte> image(size);
  for (std::size_t i = 0; i != size; ++i)
    image[i] = static_cast<std::byte>(i % 251);
  return image;
}

template <typename Predicate>
bool wait_until(Predicate predicate)
{
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(50us);
  }
  return true;
}

struct cuda_resources {
  cudaStream_t stream{nullptr};
  cudaEvent_t event{nullptr};
  void* device{nullptr};
  std::size_t capacity;

  explicit cuda_resources(std::size_t bytes = 256) : capacity(bytes)
  {
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaMalloc(&device, capacity) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(device, 0x55, capacity, stream) == cudaSuccess);
  }
  ~cuda_resources()
  {
    static_cast<void>(cudaStreamSynchronize(stream));
    static_cast<void>(cudaFree(device));
    static_cast<void>(cudaEventDestroy(event));
    static_cast<void>(cudaStreamDestroy(stream));
  }

  std::vector<std::byte> contents()
  {
    std::vector<std::byte> result(capacity);
    REQUIRE(cudaMemcpy(result.data(), device, result.size(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return result;
  }
};

struct stream_gate {
  std::atomic<bool> entered{false};
  std::atomic<bool> released{false};
  std::atomic<bool> finished{false};
  pointer_args* update{nullptr};
  pointer_args replacement;
  bool enqueued{false};

  static void CUDART_CB run(void* context)
  {
    auto& gate = *static_cast<stream_gate*>(context);
    gate.entered.store(true);
    while (!gate.released.load())
      std::this_thread::sleep_for(50us);
    if (gate.update != nullptr) *gate.update = gate.replacement;
    gate.finished.store(true);
  }
  void enqueue(cudaStream_t stream)
  {
    REQUIRE(cudaLaunchHostFunc(stream, run, this) == cudaSuccess);
    enqueued = true;
  }
  ~stream_gate()
  {
    released.store(true);
    if (enqueued) static_cast<void>(wait_until([&] { return finished.load(); }));
  }
};

struct joined_submitter {
  transport_simulator& simulator;
  std::thread thread;
  ~joined_submitter()
  {
    simulator.unblock_all();
    if (thread.joinable()) thread.join();
  }
};

cufile_transport::status read(transport_simulator& sim, cuda_resources& gpu, pointer_args& args)
{
  return sim.read_async(nullptr,
                        gpu.device,
                        &args.size,
                        &args.file_offset,
                        &args.buffer_offset,
                        &args.bytes_done,
                        gpu.stream);
}

cufile_transport::status write(transport_simulator& sim, cuda_resources& gpu, pointer_args& args)
{
  return sim.write_async(nullptr,
                         gpu.device,
                         &args.size,
                         &args.file_offset,
                         &args.buffer_offset,
                         &args.bytes_done,
                         gpu.stream);
}

void observe(transport_simulator& sim, cuda_resources& gpu)
{
  REQUIRE(sim.record_event(gpu.event, gpu.stream) == cudaSuccess);
  REQUIRE(cudaEventSynchronize(gpu.event) == cudaSuccess);
}

transport_simulator::options sampled_arguments()
{
  transport_simulator::options result;
  result.sample_arguments_at_submit = true;
  return result;
}

static_assert(std::is_base_of_v<cufile_transport, transport_simulator>);
static_assert(std::is_same_v<cufile_transport, async_transport>);

}  // namespace

TEST_CASE("transport simulator registration hooks are independently injectable",
          "[cufile][transport_simulator]")
{
  transport_simulator sim;
  std::array<std::byte, 32> buffer{};
  REQUIRE(sim.register_buffer(buffer.data(), buffer.size()).ok());
  REQUIRE(sim.deregister_buffer(buffer.data()).ok());
  REQUIRE(sim.register_stream(nullptr, 0).ok());
  REQUIRE(sim.deregister_stream(nullptr).ok());
  sim.set_registration_status({-ENOMEM}, {-EBUSY}, {-ENOSYS}, {-EIO, 17});
  CHECK(sim.register_buffer(nullptr, 0).error == -ENOMEM);
  CHECK(sim.deregister_buffer(nullptr).error == -EBUSY);
  CHECK(sim.register_stream(nullptr, 0).error == -ENOSYS);
  CHECK(sim.deregister_stream(nullptr) == cufile_transport::status{-EIO, 17});
  auto const counts = sim.snapshot();
  CHECK(counts.buffer_registrations == 2);
  CHECK(counts.buffer_deregistrations == 2);
  CHECK(counts.stream_registrations == 2);
  CHECK(counts.stream_deregistrations == 2);
}

TEST_CASE("transport simulator validates fault scripts and persistent arguments",
          "[cufile][transport_simulator]")
{
  transport_simulator sim({}, sampled_arguments());
  transport_simulator::fault fault;
  fault.file_offset = -1;
  CHECK_THROWS_AS(sim.add_fault(fault), std::invalid_argument);
  fault.file_offset.reset();
  fault.submit_latency = -1us;
  CHECK_THROWS_AS(sim.add_fault(fault), std::invalid_argument);
  fault.submit_latency     = 0us;
  fault.completion_latency = -1us;
  CHECK_THROWS_AS(sim.add_fault(fault), std::invalid_argument);
  pointer_args args;
  std::byte byte;
  CHECK(
    sim
      .read_async(
        nullptr, &byte, nullptr, &args.file_offset, &args.buffer_offset, &args.bytes_done, nullptr)
      .error == -EINVAL);
  args.buffer_offset = -1;
  CHECK(sim
          .read_async(nullptr,
                      &byte,
                      &args.size,
                      &args.file_offset,
                      &args.buffer_offset,
                      &args.bytes_done,
                      nullptr)
          .error == -EINVAL);
  CHECK(args.bytes_done == sentinel);
  CHECK(sim.snapshot().read_calls == 0);
  CHECK_FALSE(sim.unblock(0));
}

TEST_CASE("transport simulator submit faults preserve status and caller thread identity",
          "[cufile][transport_simulator]")
{
  transport_simulator sim;
  transport_simulator::fault fault;
  fault.index          = 0;
  fault.submit_status  = {-EAGAIN, 23};
  fault.submit_latency = 5ms;
  sim.add_fault(fault);
  pointer_args args;
  std::byte byte;
  auto const start  = std::chrono::steady_clock::now();
  auto const result = sim.read_async(
    nullptr, &byte, &args.size, &args.file_offset, &args.buffer_offset, &args.bytes_done, nullptr);
  CHECK(std::chrono::steady_clock::now() - start >= 5ms);
  CHECK(result == fault.submit_status);
  CHECK(args.bytes_done == sentinel);
  CHECK(sim.snapshot().submit_errors == 1);
  CHECK(sim.snapshot().submitted == 0);
  REQUIRE(sim.submissions().size() == 1);
  CHECK(sim.submissions()[0].thread == std::this_thread::get_id());
  sim.clear_faults();
}

TEST_CASE("transport simulator fault selectors intersect and first matching rule wins",
          "[cufile][transport_simulator]")
{
  transport_simulator sim({}, sampled_arguments());
  transport_simulator::fault exact;
  exact.index         = 0;
  exact.file_offset   = 7;
  exact.write         = false;
  exact.submit_status = {-EIO};
  sim.add_fault(exact);
  transport_simulator::fault write_only;
  write_only.file_offset   = 7;
  write_only.write         = true;
  write_only.submit_status = {-ENOSPC};
  sim.add_fault(write_only);
  transport_simulator::fault fallback;
  fallback.submit_status = {-EINVAL};
  sim.add_fault(fallback);
  pointer_args args{16, 7, 0, sentinel};
  std::byte byte;
  auto submit = [&](bool write) {
    return write ? sim.write_async(nullptr,
                                   &byte,
                                   &args.size,
                                   &args.file_offset,
                                   &args.buffer_offset,
                                   &args.bytes_done,
                                   nullptr)
                 : sim.read_async(nullptr,
                                  &byte,
                                  &args.size,
                                  &args.file_offset,
                                  &args.buffer_offset,
                                  &args.bytes_done,
                                  nullptr);
  };
  CHECK(submit(false).error == -EIO);
  CHECK(submit(false).error == -EINVAL);
  CHECK(submit(true).error == -ENOSPC);
  args.file_offset = 8;
  CHECK(submit(false).error == -EINVAL);
  CHECK(args.bytes_done == sentinel);
  CHECK(sim.snapshot().submit_errors == 4);
  CHECK(sim.snapshot().read_calls == 3);
  CHECK(sim.snapshot().write_calls == 1);
}

TEST_CASE("transport simulator stuck submit is explicitly unblocked without touching arguments",
          "[cufile][transport_simulator]")
{
  transport_simulator sim({}, sampled_arguments());
  transport_simulator::fault fault;
  fault.file_offset   = 7;
  fault.stuck_submit  = true;
  fault.submit_status = {-EAGAIN};
  sim.add_fault(fault);
  pointer_args args;
  args.file_offset = 7;
  std::byte byte;
  std::atomic<bool> returned{false};
  cufile_transport::status result;
  joined_submitter submitter{sim, std::thread([&] {
                               result = sim.read_async(nullptr,
                                                       &byte,
                                                       &args.size,
                                                       &args.file_offset,
                                                       &args.buffer_offset,
                                                       &args.bytes_done,
                                                       nullptr);
                               returned.store(true);
                             })};
  REQUIRE(wait_until([&] { return sim.snapshot().blocked_submissions == 1; }));
  CHECK_FALSE(returned.load());
  CHECK(args.bytes_done == sentinel);
  REQUIRE(sim.unblock(0));
  submitter.thread.join();
  CHECK(returned.load());
  CHECK(result.error == -EAGAIN);
  CHECK(sim.snapshot().blocked_submissions == 0);
  CHECK(sim.submissions()[0].thread != std::this_thread::get_id());
}

TEST_CASE("transport simulator counts overlapping stuck submit calls and resets current gauges",
          "[cufile][transport_simulator]")
{
  transport_simulator sim;
  transport_simulator::fault fault;
  fault.stuck_submit  = true;
  fault.submit_status = {-EAGAIN};
  sim.add_fault(fault);
  pointer_args a;
  pointer_args b;
  std::byte byte;
  cufile_transport::status ra, rb;
  joined_submitter first{
    sim, std::thread([&] {
      ra = sim.read_async(
        nullptr, &byte, &a.size, &a.file_offset, &a.buffer_offset, &a.bytes_done, nullptr);
    })};
  joined_submitter second{
    sim, std::thread([&] {
      rb = sim.read_async(
        nullptr, &byte, &b.size, &b.file_offset, &b.buffer_offset, &b.bytes_done, nullptr);
    })};
  REQUIRE(wait_until([&] { return sim.snapshot().blocked_submissions == 2; }));
  auto const blocked = sim.snapshot();
  CHECK(blocked.current_submit_calls == 2);
  CHECK(blocked.max_submit_concurrency >= 2);
  CHECK(blocked.current_inflight == 0);
  sim.unblock_all();
  first.thread.join();
  second.thread.join();
  CHECK(ra.error == -EAGAIN);
  CHECK(rb.error == -EAGAIN);
  auto const finished = sim.snapshot();
  CHECK(finished.current_submit_calls == 0);
  CHECK(finished.current_inflight == 0);
  CHECK(finished.max_submit_concurrency >= 2);
}

TEST_CASE("transport simulator counts two accepted operations until their completion gates open",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources first;
  cuda_resources second;
  transport_simulator sim(pattern(64));
  transport_simulator::fault fault;
  fault.stuck_completion = true;
  sim.add_fault(fault);
  pointer_args a{16, 0, 0, sentinel};
  pointer_args b{16, 16, 0, sentinel};
  REQUIRE(read(sim, first, a).ok());
  REQUIRE(read(sim, second, b).ok());
  auto const blocked = sim.snapshot();
  CHECK(blocked.current_inflight == 2);
  CHECK(blocked.max_inflight >= 2);
  CHECK(blocked.current_submit_calls == 0);
  sim.unblock_all();
  observe(sim, first);
  observe(sim, second);
  auto const finished = sim.snapshot();
  CHECK(finished.current_inflight == 0);
  CHECK(finished.max_inflight >= 2);
  CHECK(finished.current_submit_calls == 0);
  CHECK(a.bytes_done == 16);
  CHECK(b.bytes_done == 16);
}

TEST_CASE("transport simulator reads into a registered-base offset and publishes event-safe status",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources gpu;
  auto const image = pattern(256);
  transport_simulator::options options;
  options.strict_registration = true;
  transport_simulator sim(image, options);
  REQUIRE(sim.register_buffer(gpu.device, gpu.capacity).ok());
  pointer_args args{32, 11, 64, sentinel};
  REQUIRE(read(sim, gpu, args).ok());
  observe(sim, gpu);
  CHECK(args.bytes_done == 32);
  auto const actual = gpu.contents();
  CHECK(std::equal(image.begin() + 11, image.begin() + 43, actual.begin() + 64));
  CHECK(std::all_of(
    actual.begin(), actual.begin() + 64, [](std::byte b) { return b == std::byte{0x55}; }));
  CHECK(std::all_of(
    actual.begin() + 96, actual.end(), [](std::byte b) { return b == std::byte{0x55}; }));
  auto const counts = sim.snapshot();
  CHECK(counts.read_calls == 1);
  CHECK(counts.submitted == 1);
  CHECK(counts.completed == 1);
  CHECK(counts.bytes_read == 32);
  CHECK(counts.event_records == 1);
}

TEST_CASE("transport simulator writes update the image before status and preserve stream order",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources gpu;
  auto const source = pattern(256);
  transport_simulator sim(std::vector<std::byte>(256));
  REQUIRE(
    cudaMemcpyAsync(gpu.device, source.data(), source.size(), cudaMemcpyHostToDevice, gpu.stream) ==
    cudaSuccess);
  pointer_args output{32, 33, 7, sentinel};
  pointer_args input{32, 33, 128, sentinel};
  REQUIRE(write(sim, gpu, output).ok());
  REQUIRE(read(sim, gpu, input).ok());
  observe(sim, gpu);
  CHECK(output.bytes_done == 32);
  CHECK(input.bytes_done == 32);
  auto const image = sim.image();
  CHECK(std::equal(source.begin() + 7, source.begin() + 39, image.begin() + 33));
  auto const actual = gpu.contents();
  CHECK(std::equal(source.begin() + 7, source.begin() + 39, actual.begin() + 128));
  CHECK(sim.snapshot().bytes_written == 32);
  CHECK(sim.snapshot().bytes_read == 32);
  // Host-observed write completion releases the source independently of future device mutation.
  REQUIRE(cudaMemsetAsync(gpu.device, 0xcc, 256, gpu.stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(gpu.stream) == cudaSuccess);
  CHECK(sim.image() == image);
}

TEST_CASE("transport simulator short reads EOF errors and missing completion never fabricate bytes",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  for (auto const result : std::array<ssize_t, 4>{7, 0, -EIO, 100}) {
    CAPTURE(result);
    cuda_resources gpu;
    auto const image = pattern(64);
    transport_simulator sim(image);
    transport_simulator::fault fault;
    fault.index             = 0;
    fault.completion_result = result;
    sim.add_fault(fault);
    pointer_args args;
    REQUIRE(read(sim, gpu, args).ok());
    observe(sim, gpu);
    auto const expected = std::min<ssize_t>(result, 16);
    CHECK(args.bytes_done == expected);
    auto const count  = expected < 0 ? 0 : static_cast<std::size_t>(expected);
    auto const actual = gpu.contents();
    CHECK(std::equal(
      image.begin(), image.begin() + static_cast<std::ptrdiff_t>(count), actual.begin()));
    CHECK(std::all_of(actual.begin() + static_cast<std::ptrdiff_t>(count),
                      actual.end(),
                      [](std::byte b) { return b == std::byte{0x55}; }));
  }
  cuda_resources gpu;
  transport_simulator sim(pattern(64));
  pointer_args eof{16, 59, 0, sentinel};
  REQUIRE(read(sim, gpu, eof).ok());
  observe(sim, gpu);
  CHECK(eof.bytes_done == 5);
  pointer_args beyond{16, 99, 0, sentinel};
  REQUIRE(read(sim, gpu, beyond).ok());
  observe(sim, gpu);
  CHECK(beyond.bytes_done == 0);
  transport_simulator::fault fault;
  fault.index               = 2;
  fault.preserve_bytes_done = true;
  sim.add_fault(fault);
  pointer_args missing;
  REQUIRE(read(sim, gpu, missing).ok());
  observe(sim, gpu);
  CHECK(missing.bytes_done == sentinel);
}

TEST_CASE("transport simulator scripts short writes errors and extension with zero-filled gaps",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources gpu;
  transport_simulator sim(pattern(8), sampled_arguments());
  transport_simulator::fault short_write;
  short_write.write             = true;
  short_write.file_offset       = 16;
  short_write.completion_result = 5;
  sim.add_fault(short_write);
  pointer_args output{16, 16, 0, sentinel};
  REQUIRE(write(sim, gpu, output).ok());
  observe(sim, gpu);
  CHECK(output.bytes_done == 5);
  auto const image = sim.image();
  REQUIRE(image.size() == 21);
  CHECK(std::all_of(
    image.begin() + 8, image.begin() + 16, [](std::byte b) { return b == std::byte{0}; }));
  CHECK(
    std::all_of(image.begin() + 16, image.end(), [](std::byte b) { return b == std::byte{0x55}; }));
  sim.clear_faults();
  transport_simulator::fault error;
  error.write             = true;
  error.completion_result = -ENOSPC;
  sim.add_fault(error);
  pointer_args failed{16, 0, 0, sentinel};
  REQUIRE(write(sim, gpu, failed).ok());
  observe(sim, gpu);
  CHECK(failed.bytes_done == -ENOSPC);
  CHECK(sim.image() == image);
  CHECK(sim.snapshot().bytes_written == 5);
}

TEST_CASE("transport simulator completion gate protects data status and following event",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources gpu;
  transport_simulator sim(pattern(64));
  transport_simulator::fault fault;
  fault.index              = 0;
  fault.stuck_completion   = true;
  fault.completion_latency = 5ms;
  sim.add_fault(fault);
  pointer_args args;
  REQUIRE(read(sim, gpu, args).ok());
  REQUIRE(sim.record_event(gpu.event, gpu.stream) == cudaSuccess);
  REQUIRE(wait_until([&] { return sim.snapshot().blocked_completions == 1; }));
  CHECK(cudaEventQuery(gpu.event) == cudaErrorNotReady);
  CHECK(args.bytes_done == sentinel);
  CHECK(sim.snapshot().completed == 0);
  auto const start = std::chrono::steady_clock::now();
  REQUIRE(sim.unblock(0));
  REQUIRE(cudaEventSynchronize(gpu.event) == cudaSuccess);
  CHECK(std::chrono::steady_clock::now() - start >= 5ms);
  CHECK(args.bytes_done == 16);
  CHECK(sim.snapshot().blocked_completions == 0);
}

TEST_CASE("transport simulator record failure leaves partial work for explicit recovery",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources gpu;
  transport_simulator sim(pattern(64));
  transport_simulator::fault fault;
  fault.index            = 0;
  fault.stuck_completion = true;
  fault.record_error     = cudaErrorUnknown;
  sim.add_fault(fault);
  pointer_args args;
  REQUIRE(read(sim, gpu, args).ok());
  CHECK(sim.record_event(gpu.event, gpu.stream) == cudaErrorUnknown);
  REQUIRE(wait_until([&] { return sim.snapshot().blocked_completions == 1; }));
  CHECK(args.bytes_done == sentinel);
  REQUIRE(sim.unblock(0));
  REQUIRE(cudaStreamSynchronize(gpu.stream) == cudaSuccess);
  CHECK(args.bytes_done == 16);
  sim.fail_record(1, cudaErrorInvalidResourceHandle);
  CHECK(sim.record_event(gpu.event, gpu.stream) == cudaErrorInvalidResourceHandle);
  observe(sim, gpu);
  CHECK(sim.snapshot().event_records == 3);
  CHECK(sim.snapshot().record_errors == 2);
}

TEST_CASE("transport simulator teardown unblocks and drains outstanding callback contexts",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources gpu;
  pointer_args args;
  auto const image = pattern(64);
  {
    transport_simulator sim(image);
    transport_simulator::fault fault;
    fault.stuck_completion = true;
    sim.add_fault(fault);
    REQUIRE(read(sim, gpu, args).ok());
    REQUIRE(wait_until([&] { return sim.snapshot().blocked_completions == 1; }));
    CHECK(args.bytes_done == sentinel);
  }
  CHECK(args.bytes_done == 16);
  auto const actual = gpu.contents();
  CHECK(std::equal(image.begin(), image.begin() + 16, actual.begin()));
}

TEST_CASE("transport simulator per-offset scripts isolate concurrent streams and record submitters",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device");
    return;
  }
  cuda_resources first;
  cuda_resources second;
  auto const image = pattern(256);
  transport_simulator sim(image, sampled_arguments());
  transport_simulator::fault fault;
  fault.file_offset       = 64;
  fault.completion_result = 7;
  sim.add_fault(fault);
  pointer_args a{16, 0, 0, sentinel};
  pointer_args b{16, 64, 0, sentinel};
  cufile_transport::status ra, rb;
  int device = 0;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  std::thread one([&] {
    static_cast<void>(cudaSetDevice(device));
    ra = read(sim, first, a);
  });
  std::thread two([&] {
    static_cast<void>(cudaSetDevice(device));
    rb = read(sim, second, b);
  });
  one.join();
  two.join();
  REQUIRE(ra.ok());
  REQUIRE(rb.ok());
  observe(sim, first);
  observe(sim, second);
  CHECK(a.bytes_done == 16);
  CHECK(b.bytes_done == 7);
  auto const counts = sim.snapshot();
  CHECK(counts.read_calls == 2);
  CHECK(counts.submitted == 2);
  CHECK(counts.completed == 2);
  CHECK(counts.bytes_read == 23);
  auto const calls = sim.submissions();
  REQUIRE(calls.size() == 2);
  CHECK(calls[0].thread != calls[1].thread);
  CHECK(calls[0].thread != std::this_thread::get_id());
  CHECK(calls[1].thread != std::this_thread::get_id());
}

TEST_CASE("transport simulator validates fixed capacities and opt-in argument snapshots",
          "[cufile][transport_simulator]")
{
  transport_simulator::options options;
  options.image_capacity = 8;
  CHECK_THROWS_AS(transport_simulator(pattern(16), options), std::invalid_argument);
  options.image_capacity = 0;
  options.max_inflight   = 0;
  CHECK_THROWS_AS(transport_simulator({}, options), std::invalid_argument);
  transport_simulator sim;
  transport_simulator::fault fault;
  fault.file_offset = 0;
  CHECK_THROWS_AS(sim.add_fault(fault), std::invalid_argument);
}

TEST_CASE("transport simulator consumes stream-produced arguments rather than stale submit values",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  bool const writing              = GENERATE(false, true);
  bool const initialized_snapshot = GENERATE(false, true);
  cuda_resources gpu;
  auto const source  = pattern(256);
  auto const initial = std::vector<std::byte>(256, std::byte{0x22});
  transport_simulator::options options;
  options.strict_registration        = true;
  options.sample_arguments_at_submit = initialized_snapshot;
  transport_simulator sim(writing ? initial : source, options);
  REQUIRE(sim.register_buffer(gpu.device, gpu.capacity).ok());
  if (writing)
    REQUIRE(cudaMemcpyAsync(
              gpu.device, source.data(), source.size(), cudaMemcpyHostToDevice, gpu.stream) ==
            cudaSuccess);
  // Invalid until a preceding stream callback produces all three input arguments.
  pointer_args args = initialized_snapshot
                        ? pointer_args{}
                        : pointer_args{std::numeric_limits<std::size_t>::max(), -1, -1, sentinel};
  stream_gate producer;
  producer.update      = &args;
  producer.replacement = {19, 37, 64, sentinel};
  producer.enqueue(gpu.stream);
  REQUIRE(wait_until([&] { return producer.entered.load(); }));
  REQUIRE((writing ? write(sim, gpu, args) : read(sim, gpu, args)).ok());
  producer.released.store(true);
  observe(sim, gpu);
  CHECK(args.bytes_done == 19);
  if (writing) {
    auto const actual = sim.image();
    CHECK(std::equal(source.begin() + 64, source.begin() + 83, actual.begin() + 37));
  } else {
    auto const actual = gpu.contents();
    CHECK(std::equal(source.begin() + 37, source.begin() + 56, actual.begin() + 64));
  }
  auto const calls = sim.submissions();
  REQUIRE(calls.size() == 1);
  CHECK(calls[0].size == 19);
  CHECK(calls[0].file_offset == 37);
  CHECK(calls[0].buffer_offset == 64);
  CHECK(sim.snapshot().argument_changes == (initialized_snapshot ? 1 : 0));
}

TEST_CASE("transport simulator reads an extending write in stream order and enforces the image cap",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  cuda_resources gpu;
  transport_simulator::options options;
  options.image_capacity = 64;
  transport_simulator sim(pattern(8), options);
  pointer_args output{16, 32, 0, sentinel};
  pointer_args input{16, 32, 128, sentinel};
  REQUIRE(write(sim, gpu, output).ok());
  REQUIRE(read(sim, gpu, input).ok());
  observe(sim, gpu);
  CHECK(output.bytes_done == 16);
  CHECK(input.bytes_done == 16);
  CHECK(sim.image().size() == 48);
  auto const actual = gpu.contents();
  CHECK(std::all_of(actual.begin() + 128, actual.begin() + 144, [](std::byte value) {
    return value == std::byte{0x55};
  }));
  pointer_args overflow{16, 60, 0, sentinel};
  REQUIRE(write(sim, gpu, overflow).ok());
  observe(sim, gpu);
  CHECK(overflow.bytes_done == -ENOSPC);
  CHECK(sim.image().size() == 48);
}

TEST_CASE(
  "transport simulator strict registration rejects interior bases and execution-time extents",
  "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  cuda_resources gpu;
  transport_simulator::options options;
  options.strict_registration = true;
  transport_simulator sim(pattern(256), options);
  pointer_args args;
  CHECK(read(sim, gpu, args).error == -EINVAL);
  CHECK(sim.snapshot().bounce == 1);
  REQUIRE(sim.register_buffer(gpu.device, gpu.capacity).ok());
  CHECK(sim.register_buffer(gpu.device, gpu.capacity).error == -EEXIST);
  auto* interior = static_cast<std::byte*>(gpu.device) + 64;
  CHECK(sim
          .read_async(nullptr,
                      interior,
                      &args.size,
                      &args.file_offset,
                      &args.buffer_offset,
                      &args.bytes_done,
                      gpu.stream)
          .error == -EINVAL);
  CHECK(sim.snapshot().bounce == 2);
  // The registered base is valid at submit; the extent is checked only in stream order.
  args.buffer_offset = 250;
  REQUIRE(read(sim, gpu, args).ok());
  observe(sim, gpu);
  CHECK(args.bytes_done == -EINVAL);
  args = {16, 0, std::numeric_limits<off_t>::max(), sentinel};
  REQUIRE(read(sim, gpu, args).ok());
  observe(sim, gpu);
  CHECK(args.bytes_done == -EINVAL);
  sim.set_registration_status({}, {-EBUSY});
  CHECK(sim.deregister_buffer(gpu.device).error == -EBUSY);
  args = {16, 0, 240, sentinel};
  REQUIRE(read(sim, gpu, args).ok());
  observe(sim, gpu);
  CHECK(args.bytes_done == 16);  // Failed deregistration preserved the full registration.
  sim.set_registration_status({});
  REQUIRE(sim.deregister_buffer(gpu.device).ok());
  CHECK(read(sim, gpu, args).error == -EINVAL);
  CHECK(args.bytes_done == 16);
}

#ifdef CUCASCADE_HAS_CUFILE
TEST_CASE("transport simulator arena hooks validate each registration layout end to end",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  auto const granularity = GENERATE(registration_granularity::slot,
                                    registration_granularity::slab,
                                    registration_granularity::arena);
  cuda_resources gpu;
  transport_simulator::options options;
  options.strict_registration = true;
  auto const file             = pattern(128);
  transport_simulator sim(file, options);
  auto hooks             = sim.arena_hooks();
  hooks.free_bar1        = [](int) -> std::optional<std::size_t> { return 1UL << 30; };
  hooks.pinning_headroom = [](int) -> std::optional<std::size_t> { return 1UL << 30; };
  int device             = 0;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  device_arena arena(device, std::move(hooks));
  async_engine_config config;
  config.mode        = engine_mode::async_simulated_for_testing;
  config.arena_bytes = device_arena::slab_bytes;
  device_arena_options arena_options;
  arena_options.registration              = granularity;
  arena_options.allow_shared_registration = true;  // Simulator qualification test, never libcufile.
  REQUIRE(arena.warm_up(config, arena_options).registered_bytes == device_arena::slab_bytes);
  static_cast<void>(arena.set_quota(1, 1, 0, slot_class::mib1));
  auto first  = arena.try_lease(1, 32, true);
  auto second = arena.try_lease(1, 32, true);
  REQUIRE(first);
  REQUIRE(second);
  pointer_args args{32, 11, second->buffer_offset_to_pass(), sentinel};
  REQUIRE(sim
            .read_async(nullptr,
                        second->base_ptr_to_pass(),
                        &args.size,
                        &args.file_offset,
                        &args.buffer_offset,
                        &args.bytes_done,
                        gpu.stream)
            .ok());
  observe(sim, gpu);
  CHECK(args.bytes_done == 32);
  std::array<std::byte, 32> actual{};
  REQUIRE(cudaMemcpy(actual.data(), second->data(), actual.size(), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  CHECK(std::equal(file.begin() + 11, file.begin() + 43, actual.begin()));
  if (granularity != registration_granularity::slot) {
    CHECK(args.buffer_offset != 0);
    off_t zero = 0;
    CHECK(sim
            .read_async(nullptr,
                        second->data(),
                        &args.size,
                        &args.file_offset,
                        &zero,
                        &args.bytes_done,
                        gpu.stream)
            .error == -EINVAL);
  }
  auto* old_base = second->base_ptr_to_pass();
  first.reset();
  second.reset();
  arena.release_for_maintenance();
  CHECK(sim
          .read_async(nullptr,
                      old_base,
                      &args.size,
                      &args.file_offset,
                      &args.buffer_offset,
                      &args.bytes_done,
                      gpu.stream)
          .error == -EINVAL);
  CHECK(sim.snapshot().buffer_registrations == sim.snapshot().buffer_deregistrations);
}

#endif

TEST_CASE(
  "transport simulator independent nonzero I/O completes before another completion gate opens",
  "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  cuda_resources first;
  cuda_resources second;
  transport_simulator sim(pattern(256));
  transport_simulator::fault fault;
  fault.index            = 0;
  fault.stuck_completion = true;
  sim.add_fault(fault);
  pointer_args a{32, 0, 0, sentinel};
  pointer_args b{32, 64, 0, sentinel};
  REQUIRE(read(sim, first, a).ok());
  REQUIRE(wait_until([&] { return sim.snapshot().blocked_completions == 1; }));
  int device = 0;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  std::atomic<bool> returned{false};
  cufile_transport::status result;
  joined_submitter submitter{sim, std::thread([&] {
                               static_cast<void>(cudaSetDevice(device));
                               result = read(sim, second, b);
                               returned.store(true);
                             })};
  REQUIRE(wait_until([&] { return returned.load(); }));
  submitter.thread.join();
  REQUIRE(result.ok());
  REQUIRE(sim.record_event(second.event, second.stream) == cudaSuccess);
  REQUIRE(wait_until([&] { return cudaEventQuery(second.event) == cudaSuccess; }));
  CHECK(b.bytes_done == 32);
  CHECK(a.bytes_done == sentinel);
  CHECK(sim.snapshot().blocked_completions == 1);
  REQUIRE(sim.unblock(0));
  observe(sim, first);
  CHECK(a.bytes_done == 32);
}

TEST_CASE("transport simulator submits 32 one-MiB operations below the healthy latency floor",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  constexpr std::size_t qd  = 32;
  constexpr std::size_t mib = 1UL << 20;
  std::vector<std::unique_ptr<cuda_resources>> lanes;
  lanes.reserve(qd);
  for (std::size_t i = 0; i < qd; ++i)
    lanes.push_back(std::make_unique<cuda_resources>(mib));
  transport_simulator sim(pattern(qd * mib));
  std::array<pointer_args, qd> args;
  // Warm CUDA's host-function/stream enqueue machinery too, not just the simulator copy kernel.
  for (std::size_t i = 0; i < qd; ++i) {
    args[i] = {mib, static_cast<off_t>(i * mib), 0, sentinel};
    REQUIRE(read(sim, *lanes[i], args[i]).ok());
    observe(sim, *lanes[i]);
  }
  transport_simulator::fault gate;
  gate.index            = qd;
  gate.stuck_completion = true;
  sim.add_fault(gate);
  transport_simulator::fault delayed;
  delayed.completion_latency = 200ms;
  sim.add_fault(delayed);
  std::array<std::chrono::nanoseconds, qd> latencies;
  for (std::size_t i = 0; i < qd; ++i) {
    args[i].bytes_done = sentinel;
    auto const start   = std::chrono::steady_clock::now();
    auto const result  = read(sim, *lanes[i], args[i]);
    latencies[i]       = std::chrono::steady_clock::now() - start;
    REQUIRE(result.ok());
  }
  CHECK(sim.snapshot().current_inflight == qd);
  CHECK(sim.snapshot().max_inflight >= qd);
  std::sort(latencies.begin(), latencies.end());
  INFO("submit p99 ns = " << latencies.back().count());
  CHECK(latencies.back() < 50us);
  sim.unblock_all();
  for (std::size_t i = 0; i < qd; ++i) {
    observe(sim, *lanes[i]);
    CHECK(args[i].bytes_done == static_cast<ssize_t>(mib));
  }
}

TEST_CASE("transport simulator bounded records recycle only after completion",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  cuda_resources first;
  cuda_resources second;
  transport_simulator::options options;
  options.max_inflight = 1;
  transport_simulator sim(pattern(64), options);
  transport_simulator::fault fault;
  fault.index            = 0;
  fault.stuck_completion = true;
  sim.add_fault(fault);
  pointer_args a;
  pointer_args b;
  REQUIRE(read(sim, first, a).ok());
  CHECK(read(sim, second, b).error == -EAGAIN);
  CHECK(b.bytes_done == sentinel);
  REQUIRE(sim.unblock(0));
  observe(sim, first);
  REQUIRE(read(sim, second, b).ok());
  observe(sim, second);
  CHECK(b.bytes_done == 16);
  CHECK(sim.snapshot().completed == 2);
}

TEST_CASE("transport simulator recycles a no-enqueue CUDA error but retains partial work",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  cuda_resources gpu;
  transport_simulator::options options;
  options.max_inflight = 1;
  transport_simulator sim(pattern(64), options);
  pointer_args args;
  SECTION("preflight error has no queued descriptor users")
  {
    static_cast<void>(cudaGetLastError());
    REQUIRE(cudaMemsetAsync(nullptr, 0, 16, gpu.stream) == cudaErrorInvalidValue);
    REQUIRE(cudaPeekAtLastError() != cudaSuccess);
    CHECK_FALSE(read(sim, gpu, args).ok());
    static_cast<void>(cudaGetLastError());
    REQUIRE(read(sim, gpu, args).ok());
    observe(sim, gpu);
    CHECK(args.bytes_done == 16);
  }
  SECTION("partial gate is quarantined until an explicit recovery drain")
  {
    transport_simulator::fault fault;
    fault.index            = 0;
    fault.stuck_completion = true;
    fault.prepare_error    = cudaErrorUnknown;
    sim.add_fault(fault);
    CHECK_FALSE(read(sim, gpu, args).ok());
    CHECK(read(sim, gpu, args).error == -EAGAIN);
    CHECK(args.bytes_done == sentinel);
    REQUIRE(sim.unblock(0));
    REQUIRE(sim.recover_stream(gpu.stream) == cudaSuccess);
    REQUIRE(read(sim, gpu, args).ok());
    observe(sim, gpu);
    CHECK(args.bytes_done == 16);
    CHECK(sim.snapshot().completed == 1);
  }
}

TEST_CASE("transport simulator copy and wait faults use independent per-call scripts",
          "[cufile][transport_simulator]")
{
  transport_simulator sim;
  sim.fail_copy(0, cudaErrorInvalidValue);
  sim.fail_wait(0, cudaErrorInvalidResourceHandle);
  CHECK(sim.memcpy_async(nullptr, nullptr, 16, cudaMemcpyDeviceToDevice, nullptr) ==
        cudaErrorInvalidValue);
  CHECK(sim.stream_wait_event(nullptr, nullptr) == cudaErrorInvalidResourceHandle);
  CHECK(sim.snapshot().copy_errors == 1);
  CHECK(sim.snapshot().wait_errors == 1);
  CHECK(sim.snapshot().copy_calls == 1);
  CHECK(sim.snapshot().wait_calls == 1);
}

TEST_CASE("transport CUDA seams forward D2D copies and stream joins after injected faults",
          "[cufile][transport_simulator][gpu]")
{
  if (!has_cuda_device()) return;
  cuda_resources source;
  cuda_resources destination;
  transport_simulator sim;
  REQUIRE(cudaMemsetAsync(destination.device, 0x22, 256, destination.stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(destination.stream) == cudaSuccess);
  sim.fail_copy(0, cudaErrorInvalidValue);
  CHECK(sim.memcpy_async(
          destination.device, source.device, 32, cudaMemcpyDeviceToDevice, source.stream) ==
        cudaErrorInvalidValue);
  REQUIRE(sim.record_event(source.event, source.stream) == cudaSuccess);
  REQUIRE(cudaEventSynchronize(source.event) == cudaSuccess);
  auto actual = destination.contents();
  CHECK(actual[0] == std::byte{0x22});
  REQUIRE(sim.memcpy_async(
            destination.device, source.device, 32, cudaMemcpyDeviceToDevice, source.stream) ==
          cudaSuccess);
  REQUIRE(sim.record_event(source.event, source.stream) == cudaSuccess);
  sim.fail_wait(0, cudaErrorInvalidResourceHandle);
  CHECK(sim.stream_wait_event(destination.stream, source.event) == cudaErrorInvalidResourceHandle);
  REQUIRE(sim.stream_wait_event(destination.stream, source.event) == cudaSuccess);
  REQUIRE(sim.record_event(destination.event, destination.stream) == cudaSuccess);
  REQUIRE(cudaEventSynchronize(destination.event) == cudaSuccess);
  actual = destination.contents();
  CHECK(std::all_of(
    actual.begin(), actual.begin() + 32, [](std::byte value) { return value == std::byte{0x55}; }));
  CHECK(actual[32] == std::byte{0x22});
  CHECK(sim.snapshot().copy_calls == 2);
  CHECK(sim.snapshot().wait_calls == 2);
}

#ifdef CUCASCADE_HAS_CUFILE
namespace {
struct forwarded_call {
  void* handle{nullptr};
  void* base{nullptr};
  std::size_t* size{nullptr};
  off_t* file_offset{nullptr};
  off_t* buffer_offset{nullptr};
  ssize_t* bytes_done{nullptr};
  cudaStream_t stream{nullptr};
  int buffer_flags{0};
  unsigned stream_flags{0};
  std::size_t length{0};
  unsigned reads{0}, writes{0}, buffers{0}, streams{0};
};
thread_local forwarded_call forwarded;

CUfileError_t fake_read(CUfileHandle_t handle,
                        void* base,
                        std::size_t* size,
                        off_t* file_offset,
                        off_t* buffer_offset,
                        ssize_t* bytes_done,
                        CUstream stream)
{
  forwarded.handle        = handle;
  forwarded.base          = base;
  forwarded.size          = size;
  forwarded.file_offset   = file_offset;
  forwarded.buffer_offset = buffer_offset;
  forwarded.bytes_done    = bytes_done;
  forwarded.stream        = stream;
  ++forwarded.reads;
  return {CU_FILE_CUDA_DRIVER_ERROR, CUDA_ERROR_INVALID_VALUE};
}
CUfileError_t fake_write(CUfileHandle_t handle,
                         void* base,
                         std::size_t* size,
                         off_t* file_offset,
                         off_t* buffer_offset,
                         ssize_t* bytes_done,
                         CUstream stream)
{
  auto const result = fake_read(handle, base, size, file_offset, buffer_offset, bytes_done, stream);
  --forwarded.reads;
  ++forwarded.writes;
  return result;
}
CUfileError_t fake_register_buffer(void const*, std::size_t length, int flags)
{
  forwarded.length       = length;
  forwarded.buffer_flags = flags;
  ++forwarded.buffers;
  // cu_err is unspecified on success; a stale nonzero value must not turn success into failure.
  return {CU_FILE_SUCCESS, CUDA_ERROR_INVALID_VALUE};
}
CUfileError_t fake_deregister_buffer(void const*)
{
  --forwarded.buffers;
  return {CU_FILE_SUCCESS, CUDA_SUCCESS};
}
CUfileError_t fake_register_stream(CUstream, unsigned flags)
{
  forwarded.stream_flags = flags;
  ++forwarded.streams;
  return {CU_FILE_SUCCESS, CUDA_SUCCESS};
}
CUfileError_t fake_deregister_stream(CUstream)
{
  --forwarded.streams;
  return {CU_FILE_SUCCESS, CUDA_SUCCESS};
}
}  // namespace

TEST_CASE("real transport forwards persistent pointers unchanged and gates optional groups",
          "[cufile][transport_simulator][cufile_transport]")
{
  forwarded = {};
  cufile_api api;
  api.read_async        = fake_read;
  api.write_async       = fake_write;
  api.buf_register      = fake_register_buffer;
  api.buf_deregister    = fake_deregister_buffer;
  api.stream_register   = fake_register_stream;
  api.stream_deregister = fake_deregister_stream;
  cufile_async_transport transport(&api);
  pointer_args args;
  int handle = 0, buffer = 0;
  CHECK(transport
          .read_async(&handle,
                      &buffer,
                      &args.size,
                      &args.file_offset,
                      &args.buffer_offset,
                      &args.bytes_done,
                      nullptr)
          .error == -ENOSYS);
  api.has_async_read = true;
  auto const result  = transport.read_async(&handle,
                                           &buffer,
                                           &args.size,
                                           &args.file_offset,
                                           &args.buffer_offset,
                                           &args.bytes_done,
                                           nullptr);
  CHECK(result.error == CU_FILE_CUDA_DRIVER_ERROR);
  CHECK(result.cuda_error == CUDA_ERROR_INVALID_VALUE);
  CHECK(forwarded.handle == &handle);
  CHECK(forwarded.base == &buffer);
  CHECK(forwarded.size == &args.size);
  CHECK(forwarded.file_offset == &args.file_offset);
  CHECK(forwarded.buffer_offset == &args.buffer_offset);
  CHECK(forwarded.bytes_done == &args.bytes_done);
  CHECK(args.bytes_done == sentinel);
  CHECK(forwarded.reads == 1);
  CHECK(transport
          .write_async(&handle,
                       &buffer,
                       &args.size,
                       &args.file_offset,
                       &args.buffer_offset,
                       &args.bytes_done,
                       nullptr)
          .error == -ENOSYS);
  api.has_async_write = true;
  CHECK(transport.write_async(&handle,
                              &buffer,
                              &args.size,
                              &args.file_offset,
                              &args.buffer_offset,
                              &args.bytes_done,
                              nullptr) == result);
  CHECK(forwarded.writes == 1);
  CHECK(transport.register_buffer(&buffer, 71, 9).ok());
  CHECK(forwarded.length == 71);
  CHECK(forwarded.buffer_flags == 9);
  CHECK(transport.deregister_buffer(&buffer).ok());
  CHECK(forwarded.buffers == 0);
  CHECK(transport.register_stream(nullptr, 0).error == -ENOSYS);
  api.has_stream_register = true;
  CHECK(transport.register_stream(nullptr, 13).ok());
  CHECK(forwarded.stream_flags == 13);
  CHECK(transport.deregister_stream(nullptr).ok());
  CHECK(forwarded.streams == 0);
  api.stream_deregister = nullptr;
  CHECK(transport.register_stream(nullptr, 0).error == -ENOSYS);
  cufile_async_transport unavailable(nullptr);
  CHECK(unavailable.register_buffer(nullptr, 0).error == -ENOSYS);
  CHECK(unavailable.deregister_buffer(nullptr).error == -ENOSYS);
  CHECK(unavailable.deregister_stream(nullptr).error == -ENOSYS);
  if (has_cuda_device()) {
    cuda_resources gpu;
    REQUIRE(unavailable.memcpy_async(static_cast<std::byte*>(gpu.device) + 64,
                                     gpu.device,
                                     16,
                                     cudaMemcpyDeviceToDevice,
                                     gpu.stream) == cudaSuccess);
    REQUIRE(unavailable.record_event(gpu.event, gpu.stream) == cudaSuccess);
    REQUIRE(unavailable.stream_wait_event(gpu.stream, gpu.event) == cudaSuccess);
    CHECK(cudaEventSynchronize(gpu.event) == cudaSuccess);
  }
}
#endif
