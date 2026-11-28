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

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using cucascade::io::cufile::cufile_api;
using clock_type          = std::chrono::steady_clock;
constexpr std::size_t mib = 1ull << 20;
constexpr double gib      = static_cast<double>(1ull << 30);

void cuda_check(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void cufile_check(CUfileError_t status, char const* operation)
{
  if (status.err != CU_FILE_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " + cufileop_status_error(status.err) +
                             " (" + std::to_string(static_cast<int>(status.err)) + ", CUDA " +
                             std::to_string(static_cast<int>(status.cu_err)) + ")");
  }
}

double elapsed_us(clock_type::time_point begin, clock_type::time_point end)
{
  return std::chrono::duration<double, std::micro>(end - begin).count();
}

enum class registration_mode { none, arena, slot };

std::string_view registration_name(registration_mode mode)
{
  switch (mode) {
    case registration_mode::none: return "0";
    case registration_mode::arena: return "arena";
    case registration_mode::slot: return "slot";
  }
  throw std::invalid_argument("Unknown registration mode");
}

struct options {
  std::string file;
  std::size_t size_bytes{1024 * mib};
  std::vector<std::size_t> queue_depths{1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<std::size_t> op_bytes{mib, 2 * mib, 4 * mib, 8 * mib};
  double seconds{3.0};
  int device{0};
  registration_mode registration{registration_mode::slot};
  unsigned poll_us{0};
  unsigned warmup_ops{1};
  bool stream_register{true};
  bool async{true};
  bool microbench{false};
  bool verify{true};
  bool prepare_file{false};
};

void usage()
{
  std::cout << "Usage: cucascade_cufile_async_qd [options]\n"
               "  --file PATH        Existing file, opened read-only with O_DIRECT\n"
               "  --size-mib N       Working-set size (default 1024; file must be at least N MiB)\n"
               "  --qd N[,N...]      Queue depths, 1..128 (default 1,2,4,8,16,32,64,128)\n"
               "  --op-mib N[,N...]  Operation sizes in MiB (default 1,2,4,8)\n"
               "  --seconds N        Admission time per QD/size pair (default 3)\n"
               "  --device N         CUDA device ordinal (default 0)\n"
               "  --register 0|arena|slot  Buffer registration (default slot; 1 aliases arena)\n"
               "  --stream-register 0|1   Register streams when supported (default 1)\n"
               "  --poll-us N        Sleep between polling sweeps, 0..1000000 us (default 0)\n"
               "  --warmup-ops N     Untimed I/Os per lane per case, 0..1024 (default 1)\n"
               "  --mode async|sync  cuFileReadAsync or serial cuFileRead (default async)\n"
               "  --verify 0|1       Full content check after every completion (default 1)\n"
               "  --prepare-file     Create a NEW deterministic pattern file; refuses overwrite\n"
               "  --microbench       Run CUDA reaper-cost tests; --file is optional\n"
               "  --help             Show this help\n";
}

std::size_t parse_integer(std::string_view value, std::string_view name)
{
  std::size_t result{};
  auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || value.empty()) {
    throw std::invalid_argument(std::string(name) + " requires a nonnegative integer");
  }
  return result;
}

std::size_t to_bytes(std::size_t value)
{
  if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) / mib) {
    throw std::invalid_argument("MiB values must be positive and fit in off_t");
  }
  return value * mib;
}

std::vector<std::size_t> parse_list(std::string_view value, bool sizes)
{
  std::vector<std::size_t> result;
  while (true) {
    auto const comma = value.find(',');
    auto const item  = parse_integer(value.substr(0, comma), sizes ? "--op-mib" : "--qd");
    if (item == 0 || (!sizes && item > 128)) {
      throw std::invalid_argument("--qd must be in 1..128; --op-mib must be positive");
    }
    result.push_back(sizes ? to_bytes(item) : item);
    if (comma == std::string_view::npos) { break; }
    value.remove_prefix(comma + 1);
  }
  return result;
}

options parse_options(int argc, char** argv)
{
  options result;
  for (int i = 1; i < argc; ++i) {
    std::string_view const key{argv[i]};
    if (key == "--help") {
      usage();
      std::exit(EXIT_SUCCESS);
    }
    if (key == "--microbench") {
      result.microbench = true;
      continue;
    }
    if (key == "--prepare-file") {
      result.prepare_file = true;
      continue;
    }
    if (key != "--file" && key != "--size-mib" && key != "--qd" && key != "--op-mib" &&
        key != "--seconds" && key != "--device" && key != "--register" &&
        key != "--stream-register" && key != "--poll-us" && key != "--warmup-ops" &&
        key != "--mode" && key != "--verify") {
      throw std::invalid_argument("Unknown option: " + std::string(key));
    }
    if (++i == argc) { throw std::invalid_argument("Missing value for " + std::string(key)); }
    std::string_view const value{argv[i]};
    if (key == "--file") {
      result.file = value;
      if (value.empty()) { throw std::invalid_argument("--file must not be empty"); }
    } else if (key == "--size-mib") {
      result.size_bytes = to_bytes(parse_integer(value, key));
    } else if (key == "--qd") {
      result.queue_depths = parse_list(value, false);
    } else if (key == "--op-mib") {
      result.op_bytes = parse_list(value, true);
    } else if (key == "--seconds") {
      std::string const text{value};
      std::size_t consumed{};
      result.seconds = std::stod(text, &consumed);
      if (consumed != text.size() || !std::isfinite(result.seconds) || result.seconds <= 0 ||
          result.seconds > 86400) {
        throw std::invalid_argument("--seconds must be finite and in (0,86400]");
      }
    } else if (key == "--device") {
      auto const device = parse_integer(value, key);
      if (device > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--device exceeds INT_MAX");
      }
      result.device = static_cast<int>(device);
    } else if (key == "--register") {
      if (value == "0") {
        result.registration = registration_mode::none;
      } else if (value == "arena" || value == "1") {
        result.registration = registration_mode::arena;
      } else if (value == "slot") {
        result.registration = registration_mode::slot;
      } else {
        throw std::invalid_argument("--register requires 0|arena|slot (1 aliases arena)");
      }
    } else if (key == "--verify") {
      if (value != "0" && value != "1") { throw std::invalid_argument("--verify requires 0|1"); }
      result.verify = value == "1";
    } else if (key == "--stream-register") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--stream-register requires 0|1");
      }
      result.stream_register = value == "1";
    } else if (key == "--poll-us") {
      auto const delay = parse_integer(value, key);
      if (delay > 1000000) { throw std::invalid_argument("--poll-us must be in 0..1000000"); }
      result.poll_us = static_cast<unsigned>(delay);
    } else if (key == "--warmup-ops") {
      auto const count = parse_integer(value, key);
      if (count > 1024) { throw std::invalid_argument("--warmup-ops must be in 0..1024"); }
      result.warmup_ops = static_cast<unsigned>(count);
    } else if (key == "--mode") {
      if (value != "async" && value != "sync") {
        throw std::invalid_argument("--mode requires async|sync");
      }
      result.async = value == "async";
    }
  }
  if (result.file.empty() && !result.microbench) {
    throw std::invalid_argument("--file is required unless only --microbench is requested");
  }
  if (*std::max_element(result.op_bytes.begin(), result.op_bytes.end()) > result.size_bytes) {
    throw std::invalid_argument("--op-mib cannot exceed --size-mib");
  }
  return result;
}

class device_buffer {
 public:
  explicit device_buffer(std::size_t size) { cuda_check(cudaMalloc(&_ptr, size), "cudaMalloc"); }
  ~device_buffer() { static_cast<void>(cudaFree(_ptr)); }
  device_buffer(device_buffer const&)            = delete;
  device_buffer& operator=(device_buffer const&) = delete;
  [[nodiscard]] void* get() const { return _ptr; }

 private:
  void* _ptr{};
};

class buffer_registration {
 public:
  buffer_registration(cufile_api const& api, void* ptr, std::size_t size) : _api(api), _ptr(ptr)
  {
    cufile_check(_api.buf_register(ptr, size, 0), "cuFileBufRegister");
  }
  ~buffer_registration()
  {
    if (_ptr != nullptr) { static_cast<void>(_api.buf_deregister(_ptr)); }
  }
  buffer_registration(buffer_registration const&)            = delete;
  buffer_registration& operator=(buffer_registration const&) = delete;

 private:
  cufile_api const& _api;
  void* _ptr;
};

class file_descriptor {
 public:
  explicit file_descriptor(std::string const& path, int flags = O_RDONLY | O_DIRECT)
    : _fd(::open(path.c_str(), flags, 0600))
  {
    if (_fd < 0) { throw std::runtime_error("open: " + std::string(std::strerror(errno))); }
  }
  ~file_descriptor() { static_cast<void>(::close(_fd)); }
  file_descriptor(file_descriptor const&)            = delete;
  file_descriptor& operator=(file_descriptor const&) = delete;
  [[nodiscard]] int get() const { return _fd; }

 private:
  int _fd;
};

// Offset-dependent data makes wrong-offset and cross-lane writes observable. Never replace an
// existing file: an operator must explicitly remove a disposable diagnostic file before reuse.
void prepare_pattern_file(options const& config)
{
  file_descriptor output(config.file, O_WRONLY | O_CREAT | O_EXCL);
  std::vector<std::uint64_t> words(mib / sizeof(std::uint64_t));
  for (std::size_t offset = 0; offset < config.size_bytes; offset += mib) {
    for (std::size_t i = 0; i < words.size(); ++i) {
      auto value =
        static_cast<std::uint64_t>(offset / sizeof(std::uint64_t) + i) + 0x9e3779b97f4a7c15ULL;
      value    = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value    = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      words[i] = value ^ (value >> 31);
    }
    auto const* bytes = reinterpret_cast<char const*>(words.data());
    std::size_t written{};
    while (written < mib) {
      auto const count = ::write(output.get(), bytes + written, mib - written);
      if (count < 0 && errno == EINTR) { continue; }
      if (count <= 0) { throw std::runtime_error("writing pattern file failed"); }
      written += static_cast<std::size_t>(count);
    }
  }
  if (::fsync(output.get()) != 0) { throw std::runtime_error("fsync pattern file failed"); }
}

// Correctness mode intentionally includes D2H/reference-read overhead in throughput. Use
// --verify 0 only for performance measurements AFTER a content-validated smoke run.
class content_checker {
 public:
  content_checker(options const& config, std::size_t capacity)
  {
    if (!config.verify) { return; }
    _file = std::make_unique<file_descriptor>(config.file, O_RDONLY);
    _expected.resize(capacity);
    cuda_check(cudaMallocHost(&_actual, capacity), "cudaMallocHost(verification)");
  }
  ~content_checker()
  {
    if (_actual != nullptr) { static_cast<void>(cudaFreeHost(_actual)); }
  }
  content_checker(content_checker const&)            = delete;
  content_checker& operator=(content_checker const&) = delete;

  bool check(void* device, std::size_t size, off_t offset, cudaStream_t stream)
  {
    if (_file == nullptr) { return true; }
    std::size_t read{};
    while (read < size) {
      auto const count = ::pread(
        _file->get(), _expected.data() + read, size - read, offset + static_cast<off_t>(read));
      if (count < 0 && errno == EINTR) { continue; }
      if (count <= 0) { throw std::runtime_error("verification reference pread failed"); }
      read += static_cast<std::size_t>(count);
    }
    cuda_check(cudaMemcpyAsync(_actual, device, size, cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(verification)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(verification)");
    return std::memcmp(_actual, _expected.data(), size) == 0;
  }

 private:
  std::unique_ptr<file_descriptor> _file;
  std::vector<std::byte> _expected;
  void* _actual{};
};

class file_handle {
 public:
  file_handle(cufile_api const& api, int fd) : _api(api)
  {
    CUfileDescr_t descriptor{};
    descriptor.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = fd;
    cufile_check(api.handle_register(&_handle, &descriptor), "cuFileHandleRegister");
  }
  ~file_handle() { _api.handle_deregister(_handle); }
  file_handle(file_handle const&)            = delete;
  file_handle& operator=(file_handle const&) = delete;
  [[nodiscard]] CUfileHandle_t get() const { return _handle; }

 private:
  cufile_api const& _api;
  CUfileHandle_t _handle{};
};

class stream_handle {
 public:
  stream_handle()
  {
    cuda_check(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  ~stream_handle()
  {
    static_cast<void>(cudaStreamSynchronize(_stream));
    static_cast<void>(cudaStreamDestroy(_stream));
  }
  stream_handle(stream_handle const&)            = delete;
  stream_handle& operator=(stream_handle const&) = delete;
  [[nodiscard]] cudaStream_t get() const { return _stream; }

 private:
  cudaStream_t _stream{};
};

class event_handle {
 public:
  event_handle()
  {
    cuda_check(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming),
               "cudaEventCreateWithFlags");
  }
  ~event_handle() { static_cast<void>(cudaEventDestroy(_event)); }
  event_handle(event_handle const&)            = delete;
  event_handle& operator=(event_handle const&) = delete;
  [[nodiscard]] cudaEvent_t get() const { return _event; }

 private:
  cudaEvent_t _event{};
};

struct arguments {
  std::size_t size{};
  off_t file_offset{};
  off_t buffer_offset{};
  ssize_t bytes_done{std::numeric_limits<ssize_t>::min()};
};

class pinned_arguments {
 public:
  pinned_arguments()
  {
    void* ptr{};
    cuda_check(cudaMallocHost(&ptr, sizeof(arguments)), "cudaMallocHost(arguments)");
    _ptr = ::new (ptr) arguments{};
  }
  ~pinned_arguments() { static_cast<void>(cudaFreeHost(_ptr)); }
  pinned_arguments(pinned_arguments const&)            = delete;
  pinned_arguments& operator=(pinned_arguments const&) = delete;
  [[nodiscard]] arguments& get() { return *_ptr; }

 private:
  arguments* _ptr{};
};

// Pinned argument addresses, arena slots, streams and events never move or change while in flight.
struct lane {
  pinned_arguments args;
  stream_handle stream;
  event_handle event;
  cufile_api const* api{};
  bool registered_stream{false};
  bool active{false};
  bool first_poll{false};
  clock_type::time_point started;

  explicit lane(cufile_api const* table = nullptr, bool register_stream = true) : api(table)
  {
    cuda_check(cudaEventRecord(event.get(), stream.get()), "cudaEventRecord(warmup)");
    cuda_check(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize(warmup)");
    if (register_stream && api != nullptr && api->has_stream_register) {
      constexpr unsigned flags = CU_FILE_STREAM_FIXED_BUF_OFFSET |
                                 CU_FILE_STREAM_FIXED_FILE_OFFSET | CU_FILE_STREAM_FIXED_FILE_SIZE |
                                 CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;
      cufile_check(api->stream_register(stream.get(), flags), "cuFileStreamRegister");
      registered_stream = true;
    }
  }
  ~lane()
  {
    // Also drains a submitted operation if recording its event or later bookkeeping threw.
    static_cast<void>(cudaStreamSynchronize(stream.get()));
    if (registered_stream) { static_cast<void>(api->stream_deregister(stream.get())); }
  }
  lane(lane const&)            = delete;
  lane& operator=(lane const&) = delete;
};

struct distribution {
  std::vector<double> samples;
  void add(double us) { samples.push_back(us); }
  void sort() { std::sort(samples.begin(), samples.end()); }
  [[nodiscard]] double percentile(double fraction) const
  {
    if (samples.empty()) { return 0; }
    auto const index =
      static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(samples.size())));
    return samples[std::min(samples.size() - 1, index == 0 ? 0 : index - 1)];
  }
  [[nodiscard]] double maximum() const { return samples.empty() ? 0 : samples.back(); }
};

struct measurements {
  distribution submit;
  distribution completion;
  std::uint64_t bytes{};
  std::uint64_t completed{};
  std::uint64_t mismatches{};
  std::uint64_t data_mismatches{};
  std::uint64_t queries{};
  std::uint64_t ready_first_poll{};
  std::uint64_t slow_submits{};
};

void* prepare_read(lane& slot,
                   void* arena,
                   registration_mode registration,
                   std::size_t index,
                   std::size_t slot_stride,
                   std::size_t op_size,
                   std::size_t file_offset)
{
  auto& args       = slot.args.get();
  args.size        = op_size;
  args.file_offset = static_cast<off_t>(file_offset);
  args.buffer_offset =
    registration == registration_mode::arena ? static_cast<off_t>(index * slot_stride) : 0;
  args.bytes_done = std::numeric_limits<ssize_t>::min();
  // Slot registrations and unregistered calls both use the slot base with offset zero.
  return registration == registration_mode::arena
           ? arena
           : static_cast<void*>(static_cast<std::byte*>(arena) + index * slot_stride);
}

void warmup_io(cufile_api const& api,
               CUfileHandle_t file,
               void* arena,
               std::vector<std::unique_ptr<lane>>& lanes,
               options const& config,
               std::size_t qd,
               std::size_t op_size,
               std::size_t slot_stride,
               content_checker& checker)
{
  // Exercise cuFile's lazy per-stream initialization outside every measured case.
  for (std::size_t operation = 0; operation < qd * config.warmup_ops; ++operation) {
    std::size_t const i = operation % qd;
    auto& slot          = *lanes[i];
    auto& args          = slot.args.get();
    void* const base = prepare_read(slot, arena, config.registration, i, slot_stride, op_size, 0);
    if (config.async) {
      cufile_check(api.read_async(file,
                                  base,
                                  &args.size,
                                  &args.file_offset,
                                  &args.buffer_offset,
                                  &args.bytes_done,
                                  slot.stream.get()),
                   "cuFileReadAsync(warmup)");
    } else {
      args.bytes_done = api.read(file, base, args.size, args.file_offset, args.buffer_offset);
    }
    cuda_check(cudaEventRecord(slot.event.get(), slot.stream.get()), "cudaEventRecord(I/O warmup)");
    cuda_check(cudaEventSynchronize(slot.event.get()), "cudaEventSynchronize(I/O warmup)");
    if (args.bytes_done != static_cast<ssize_t>(op_size)) {
      throw std::runtime_error("I/O warmup bytes_done mismatch: lane=" + std::to_string(i) +
                               " expected=" + std::to_string(op_size) +
                               " actual=" + std::to_string(args.bytes_done));
    }
    if (!checker.check(static_cast<std::byte*>(arena) + i * slot_stride,
                       op_size,
                       args.file_offset,
                       slot.stream.get())) {
      throw std::runtime_error("I/O warmup data mismatch: lane=" + std::to_string(i) +
                               " file_offset=" + std::to_string(args.file_offset));
    }
  }
}

bool run_case(cufile_api const& api,
              CUfileHandle_t file,
              void* arena,
              std::vector<std::unique_ptr<lane>>& lanes,
              options const& config,
              std::size_t qd,
              std::size_t op_size,
              std::size_t slot_stride,
              content_checker& checker)
{
  warmup_io(api, file, arena, lanes, config, qd, op_size, slot_stride, checker);
  measurements measured;
  std::size_t next_offset{};
  std::size_t inflight{};
  auto const begin    = clock_type::now();
  auto const deadline = begin + std::chrono::duration<double>(config.seconds);
  auto submit         = [&](std::size_t index) {
    auto& slot = *lanes[index];
    auto& args = slot.args.get();
    void* const base =
      prepare_read(slot, arena, config.registration, index, slot_stride, op_size, next_offset);
    next_offset += op_size;
    if (next_offset > config.size_bytes - op_size) { next_offset = 0; }
    slot.started = clock_type::now();
    if (config.async) {
      auto const status   = api.read_async(file,
                                         base,
                                         &args.size,
                                         &args.file_offset,
                                         &args.buffer_offset,
                                         &args.bytes_done,
                                         slot.stream.get());
      auto const returned = clock_type::now();
      measured.submit.add(elapsed_us(slot.started, returned));
      cufile_check(status, "cuFileReadAsync");
    } else {
      args.bytes_done = api.read(file, base, args.size, args.file_offset, args.buffer_offset);
      measured.submit.add(elapsed_us(slot.started, clock_type::now()));
    }
    if (measured.submit.samples.back() > 100.0) { ++measured.slow_submits; }
    cuda_check(cudaEventRecord(slot.event.get(), slot.stream.get()), "cudaEventRecord(read)");
    slot.active     = true;
    slot.first_poll = true;
    ++inflight;
  };

  for (std::size_t i = 0; i < qd && (i == 0 || clock_type::now() < deadline); ++i) {
    submit(i);
  }
  while (inflight != 0) {
    for (std::size_t i = 0; i < qd; ++i) {
      auto& slot = *lanes[i];
      if (!slot.active) { continue; }
      ++measured.queries;
      auto const status = cudaEventQuery(slot.event.get());
      if (status == cudaErrorNotReady) {
        slot.first_poll = false;
        continue;
      }
      cuda_check(status, "cudaEventQuery(read)");
      // Never inspect bytes_done before the event recorded AFTER cuFileReadAsync succeeds.
      ssize_t const bytes_done = slot.args.get().bytes_done;
      measured.completion.add(elapsed_us(slot.started, clock_type::now()));
      ++measured.completed;
      if (slot.first_poll) { ++measured.ready_first_poll; }
      if (bytes_done != static_cast<ssize_t>(op_size)) {
        if (measured.mismatches < 5) {
          std::cerr << "bytes_done mismatch after event: expected=" << op_size
                    << " actual=" << bytes_done << " file_offset=" << slot.args.get().file_offset
                    << '\n';
        }
        ++measured.mismatches;
      } else {
        measured.bytes += op_size;
        if (!checker.check(static_cast<std::byte*>(arena) + i * slot_stride,
                           op_size,
                           slot.args.get().file_offset,
                           slot.stream.get())) {
          if (measured.data_mismatches < 5) {
            std::cerr << "data mismatch after event: lane=" << i
                      << " file_offset=" << slot.args.get().file_offset
                      << " buffer_offset=" << slot.args.get().buffer_offset << '\n';
          }
          ++measured.data_mismatches;
        }
      }
      slot.active = false;
      --inflight;
      // Refill this lane immediately; do not wait for any other operation in the window.
      if (clock_type::now() < deadline) { submit(i); }
    }
    if (inflight != 0 && config.poll_us != 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(config.poll_us));
    }
  }
  double const seconds = std::chrono::duration<double>(clock_type::now() - begin).count();
  measured.submit.sort();
  measured.completion.sort();
  bool const blocking =
    measured.submit.percentile(0.5) >= 0.5 * measured.completion.percentile(0.5);
  std::cout << "result," << (config.async ? "async" : "sync") << ',' << config.device << ','
            << registration_name(config.registration) << ',' << qd << ',' << op_size / mib << ','
            << seconds << ',' << measured.completed << ','
            << static_cast<double>(measured.bytes) / gib / seconds << ','
            << measured.submit.percentile(0.5) << ',' << measured.submit.percentile(0.99) << ','
            << measured.submit.maximum() << ',' << measured.completion.percentile(0.5) << ','
            << measured.completion.percentile(0.99) << ',' << measured.mismatches << ',' << blocking
            << ',' << measured.slow_submits << ',' << measured.queries << ','
            << measured.ready_first_poll << ',' << measured.data_mismatches << ',' << config.verify
            << std::endl;
  return measured.mismatches == 0 && measured.data_mismatches == 0;
}

void micro_result(std::string_view name, distribution& measured, std::size_t heads = 1)
{
  measured.sort();
  std::cout << "micro," << name << ',' << heads << ',' << measured.samples.size() << ','
            << measured.percentile(0.5) << ',' << measured.percentile(0.99) << ','
            << measured.maximum() << ',' << measured.percentile(0.5) / static_cast<double>(heads)
            << std::endl;
}

template <typename Fn>
void measure_calls(std::string_view name, std::size_t iterations, Fn&& fn, std::size_t heads = 1)
{
  distribution measured;
  measured.samples.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    auto const begin = clock_type::now();
    fn();
    measured.add(elapsed_us(begin, clock_type::now()));
  }
  micro_result(name, measured, heads);
}

// A controlled pending event, not a compute-load simulation. The callback calls no CUDA API.
// Release precedes stream destruction on every path, including exceptions during measurement.
class pending_gate {
 public:
  pending_gate()
  {
    cuda_check(cudaLaunchHostFunc(
                 _stream.get(),
                 [](void* context) {
                   auto& release = *static_cast<std::atomic<bool>*>(context);
                   while (!release.load(std::memory_order_acquire)) {
                     std::this_thread::yield();
                   }
                 },
                 &_release),
               "cudaLaunchHostFunc(gate)");
  }
  ~pending_gate()
  {
    _release.store(true, std::memory_order_release);
    static_cast<void>(cudaStreamSynchronize(_stream.get()));
  }
  pending_gate(pending_gate const&)            = delete;
  pending_gate& operator=(pending_gate const&) = delete;
  [[nodiscard]] cudaStream_t stream() const { return _stream.get(); }

 private:
  std::atomic<bool> _release{false};
  stream_handle _stream;
};

void run_microbench()
{
  constexpr std::size_t query_iterations   = 20000;
  constexpr std::size_t enqueue_iterations = 2000;
  std::cout << "# CUDA-only microbench: host wall-clock us (includes timer overhead); D2D=1MiB\n"
               "kind,component,heads,samples,p50_us,p99_us,max_us,p50_us_per_head\n";
  measure_calls("timer_baseline", query_iterations, [] {});
  std::vector<std::unique_ptr<lane>> heads;
  for (std::size_t i = 0; i < 160; ++i) {
    heads.push_back(std::make_unique<lane>());
  }
  measure_calls("event_query_ready", query_iterations, [&] {
    cuda_check(cudaEventQuery(heads.front()->event.get()), "cudaEventQuery(ready)");
  });
  for (std::size_t const count : {32u, 64u, 160u}) {
    measure_calls(
      "poll_ready_heads",
      2000,
      [&] {
        for (std::size_t i = 0; i < count; ++i) {
          cuda_check(cudaEventQuery(heads[i]->event.get()), "cudaEventQuery(ready heads)");
        }
      },
      count);
  }
  {
    // Declared AFTER heads: the gate is released before their destructors can synchronize.
    pending_gate gate;
    event_handle gate_event;
    cuda_check(cudaEventRecord(gate_event.get(), gate.stream()), "cudaEventRecord(gate)");
    for (auto& head : heads) {
      cuda_check(cudaStreamWaitEvent(head->stream.get(), gate_event.get(), 0),
                 "cudaStreamWaitEvent");
      cuda_check(cudaEventRecord(head->event.get(), head->stream.get()),
                 "cudaEventRecord(pending)");
    }
    auto require_pending = [](cudaEvent_t event) {
      auto const status = cudaEventQuery(event);
      if (status != cudaErrorNotReady) {
        cuda_check(status, "cudaEventQuery(pending)");
        throw std::runtime_error("Controlled not-ready event unexpectedly completed");
      }
    };
    measure_calls("event_query_not_ready", query_iterations, [&] {
      require_pending(heads.front()->event.get());
    });
    for (std::size_t const count : {32u, 64u, 160u}) {
      measure_calls(
        "poll_not_ready_heads",
        2000,
        [&] {
          for (std::size_t i = 0; i < count; ++i) {
            require_pending(heads[i]->event.get());
          }
        },
        count);
    }
  }
  for (auto& head : heads) {
    cuda_check(cudaStreamSynchronize(head->stream.get()), "cudaStreamSynchronize(head)");
  }
  auto& head = *heads.front();
  measure_calls("event_record_enqueue", enqueue_iterations, [&] {
    cuda_check(cudaEventRecord(head.event.get(), head.stream.get()), "cudaEventRecord(measure)");
  });
  cuda_check(cudaStreamSynchronize(head.stream.get()), "cudaStreamSynchronize(record)");
  device_buffer copy_arena(2 * mib);
  stream_handle copy_stream;  // Drain before copy_arena is freed, including exceptional exits.
  auto* base = static_cast<std::byte*>(copy_arena.get());
  cuda_check(cudaMemsetAsync(base, 0, 2 * mib, copy_stream.get()), "cudaMemsetAsync(microbench)");
  cuda_check(cudaStreamSynchronize(copy_stream.get()), "cudaStreamSynchronize(init microbench)");
  distribution d2d_enqueue;
  d2d_enqueue.samples.reserve(enqueue_iterations);
  for (std::size_t i = 0; i < enqueue_iterations; ++i) {
    auto const begin = clock_type::now();
    cuda_check(cudaMemcpyAsync(base + mib, base, mib, cudaMemcpyDeviceToDevice, copy_stream.get()),
               "cudaMemcpyAsync(D2D)");
    d2d_enqueue.add(elapsed_us(begin, clock_type::now()));
    // Keep queue-limit backpressure out of the enqueue sample, including this drain.
    if ((i + 1) % 256 == 0) {
      cuda_check(cudaStreamSynchronize(copy_stream.get()), "cudaStreamSynchronize(D2D batch)");
    }
  }
  cuda_check(cudaStreamSynchronize(copy_stream.get()), "cudaStreamSynchronize(D2D)");
  micro_result("d2d_enqueue_1mib", d2d_enqueue);
}

bool run_io(options const& config)
{
  if (config.prepare_file) { prepare_pattern_file(config); }
  auto const* api = cufile_api::get();
  if (api == nullptr) {
    throw std::runtime_error("libcufile.so.0 or required symbols unavailable");
  }
  if (config.async && !api->has_async_read) {
    throw std::runtime_error("cuFileReadAsync unavailable; use --mode sync for the sync baseline");
  }
  if (!cufile_api::open_driver()) { throw std::runtime_error("cuFileDriverOpen failed"); }
  file_descriptor fd(config.file);
  struct stat info{};
  if (::fstat(fd.get(), &info) != 0) {
    throw std::runtime_error("fstat: " + std::string(std::strerror(errno)));
  }
  if (!S_ISREG(info.st_mode) || info.st_size < static_cast<off_t>(config.size_bytes)) {
    throw std::invalid_argument("--file must be a regular file of at least --size-mib MiB");
  }
  file_handle file(*api, fd.get());
  auto const max_qd = *std::max_element(config.queue_depths.begin(), config.queue_depths.end());
  auto const stride = *std::max_element(config.op_bytes.begin(), config.op_bytes.end());
  if (stride > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) / max_qd) {
    throw std::invalid_argument("Arena size exceeds off_t");
  }
  std::size_t const arena_bytes = max_qd * stride;
  device_buffer arena(arena_bytes);
  cuda_check(cudaMemset(arena.get(), 0, arena_bytes), "cudaMemset(arena)");
  cuda_check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize(init arena)");
  // Declared before lanes: drain all streams before deregistering any buffer, including
  // exceptional exits. Each successful registration is owned before the next one starts.
  std::vector<std::unique_ptr<buffer_registration>> registrations;
  if (config.registration == registration_mode::arena) {
    registrations.push_back(std::make_unique<buffer_registration>(*api, arena.get(), arena_bytes));
  } else if (config.registration == registration_mode::slot) {
    registrations.reserve(max_qd);
    for (std::size_t i = 0; i < max_qd; ++i) {
      auto* const base = static_cast<std::byte*>(arena.get()) + i * stride;
      registrations.push_back(std::make_unique<buffer_registration>(*api, base, stride));
    }
  }
  content_checker checker(config, stride);
  std::vector<std::unique_ptr<lane>> lanes;
  for (std::size_t i = 0; i < max_qd; ++i) {
    lanes.push_back(std::make_unique<lane>(api, config.stream_register));
  }
  std::cout
    << "# file=" << std::quoted(config.file) << " size_mib=" << config.size_bytes / mib
    << " arena_mib=" << arena_bytes / mib << " cufile_version=" << cufile_api::version()
    << " registration=" << registration_name(config.registration)
    << " stream_registration=" << (config.stream_register && api->has_stream_register)
    << " poll_us=" << config.poll_us << " warmup_ops_per_lane_per_case=" << config.warmup_ops
    << " verify=" << config.verify
    << " bytes_done_checked=after_event direct_path=unverified\n"
       "# verification adds D2H and reference-read overhead; --verify 0 for throughput only\n"
       "# sync mode is a single-host-thread serial baseline, not QD concurrent I/O\n"
       "kind,mode,device,registration,qd,op_mib,elapsed_s,completed,gib_per_s,submit_p50_us,"
       "submit_p99_us,submit_max_us,completion_p50_us,completion_p99_us,bytes_done_mismatches,"
       "blocking_indicator,submits_over_100us,queries,ready_first_poll,data_mismatches,verify\n";
  bool success = true;
  for (auto const op_size : config.op_bytes) {
    for (auto const qd : config.queue_depths) {
      success =
        run_case(*api, file.get(), arena.get(), lanes, config, qd, op_size, stride, checker) &&
        success;
    }
  }
  return success;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const config = parse_options(argc, argv);
    cuda_check(cudaSetDevice(config.device), "cudaSetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, config.device), "cudaGetDeviceProperties");
    std::cout << std::fixed << std::setprecision(3) << "# device=" << config.device
              << " name=" << std::quoted(properties.name) << '\n';
    if (config.microbench) { run_microbench(); }
    return config.file.empty() || run_io(config) ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (std::exception const& error) {
    std::cerr << "cucascade_cufile_async_qd: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
