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

#include "transport_simulator_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace cucascade::io::cufile {
namespace {

bool valid_range(void* base, std::size_t size, off_t file_offset, off_t buffer_offset) noexcept
{
  if (file_offset < 0 || buffer_offset < 0 ||
      size > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()))
    return false;
  auto const limit   = std::numeric_limits<std::size_t>::max();
  auto const address = reinterpret_cast<std::uintptr_t>(base);
  return static_cast<std::size_t>(file_offset) <= limit - size &&
         static_cast<std::size_t>(buffer_offset) <= limit - size &&
         address <= std::numeric_limits<std::uintptr_t>::max() -
                      (static_cast<std::size_t>(buffer_offset) + size);
}

}  // namespace

struct transport_simulator::state {
  struct transfer {
    state* owner{nullptr};
    fault rule;
    std::size_t index{0};
    bool occupied{false};
    bool released{false};  // Protected by owner->mutex.
    bool completion_counted{false};
    bool enqueue_failed{false};
    bool write{false};
    void* base{nullptr};
    std::size_t* size{nullptr};
    off_t* file_offset{nullptr};
    off_t* buffer_offset{nullptr};
    std::size_t initial_size{0};
    off_t initial_file_offset{0};
    off_t initial_buffer_offset{0};
    ssize_t result{0};
    ssize_t* bytes_done{nullptr};
    cudaStream_t stream{nullptr};
    int device{0};
    simulator_copy_args* copy{nullptr};
    simulator_copy_args* device_copy{nullptr};
    bool inflight_counted{false};  // Protected by owner->mutex.
    std::atomic<bool> submit_returned{false};
    std::atomic<bool> complete{false};
  };

  struct enqueue_scope {
    transfer* item;
    ~enqueue_scope() { item->submit_returned.store(true, std::memory_order_release); }
  };

  mutable std::mutex mutex;
  std::condition_variable cv;
  options config;
  std::atomic<ssize_t> sync_result{std::numeric_limits<ssize_t>::min()};
  std::size_t image_size{0};
  std::size_t image_capacity{0};
  void* pinned_image{nullptr};
  void* pinned_args{nullptr};
  std::byte* image{nullptr};
  unsigned char* device_image{nullptr};
  std::vector<std::byte> cpu_image;  // CPU-only fault scripts remain usable without a device.
  std::vector<fault> faults;
  counters counts;
  std::vector<submission> calls;
  std::unique_ptr<transfer[]> transfers;
  std::map<void*, std::size_t, std::less<void*>> registrations;
  std::map<cudaStream_t, cudaError_t> stream_record_errors;
  std::map<std::size_t, cudaError_t> record_errors;
  std::map<std::size_t, cudaError_t> copy_errors;
  std::map<std::size_t, cudaError_t> wait_errors;
  status buffer_register;
  status buffer_deregister;
  status stream_register;
  status stream_deregister;

  ~state()
  {
    if (pinned_args != nullptr) static_cast<void>(cudaFreeHost(pinned_args));
    if (pinned_image != nullptr) static_cast<void>(cudaFreeHost(pinned_image));
  }

  void initialize(std::vector<std::byte> initial, options settings)
  {
    config     = settings;
    image_size = initial.size();
    image_capacity =
      config.image_capacity == 0 ? std::max(image_size, 1UL << 20) : config.image_capacity;
    if (image_capacity < image_size || config.max_inflight == 0 ||
        config.max_inflight > std::numeric_limits<std::size_t>::max() / sizeof(simulator_copy_args))
      throw std::invalid_argument("invalid simulator image/transfer capacity");
    transfers = std::make_unique<transfer[]>(config.max_inflight);
    calls.reserve(config.max_inflight);
    for (std::size_t i = 0; i < config.max_inflight; ++i)
      transfers[i].owner = this;

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      cpu_image = std::move(initial);
      image     = cpu_image.data();
      return;
    }
    auto check = [](cudaError_t error) {
      if (error != cudaSuccess)
        throw std::runtime_error("simulator pinned warm-up failed: " +
                                 std::string(cudaGetErrorString(error)));
    };
    auto constexpr flags = cudaHostAllocMapped | cudaHostAllocPortable;
    check(cudaHostAlloc(&pinned_image, image_capacity, flags));
    check(cudaHostAlloc(&pinned_args, config.max_inflight * sizeof(simulator_copy_args), flags));
    image = static_cast<std::byte*>(pinned_image);
    std::memset(image, 0, image_capacity);
    std::copy(initial.begin(), initial.end(), image);
    std::memset(pinned_args, 0, config.max_inflight * sizeof(simulator_copy_args));
    void* mapped_image = nullptr;
    void* mapped_args  = nullptr;
    check(cudaHostGetDevicePointer(&mapped_image, pinned_image, 0));
    check(cudaHostGetDevicePointer(&mapped_args, pinned_args, 0));
    device_image = static_cast<unsigned char*>(mapped_image);
    for (std::size_t i = 0; i < config.max_inflight; ++i) {
      transfers[i].copy        = static_cast<simulator_copy_args*>(pinned_args) + i;
      transfers[i].device_copy = static_cast<simulator_copy_args*>(mapped_args) + i;
    }
    // Load/JIT the tiny copy kernel outside measured submission (also avoids cold CUDA timing).
    cudaStream_t warm_up = nullptr;
    check(cudaStreamCreateWithFlags(&warm_up, cudaStreamNonBlocking));
    transfers[0].copy->released = 1;
    auto error                  = launch_simulator_gate(transfers[0].device_copy, warm_up);
    if (error == cudaSuccess) error = launch_simulator_copy(transfers[0].device_copy, warm_up);
    if (error == cudaSuccess) error = cudaStreamSynchronize(warm_up);
    static_cast<void>(cudaStreamDestroy(warm_up));
    check(error);
  }

  struct submit_scope {
    state& owner;
    explicit submit_scope(state& self) : owner(self)
    {
      std::lock_guard lock(owner.mutex);
      ++owner.counts.current_submit_calls;
      owner.counts.max_submit_concurrency =
        std::max(owner.counts.max_submit_concurrency, owner.counts.current_submit_calls);
    }
    ~submit_scope()
    {
      std::lock_guard lock(owner.mutex);
      --owner.counts.current_submit_calls;
    }
  };

  // mutex held. Recycle metadata only; never free pinned memory or issue CUDA calls.
  transfer* acquire()
  {
    for (std::size_t i = 0; i < config.max_inflight; ++i) {
      auto& item = transfers[i];
      if (!item.occupied || (item.complete.load(std::memory_order_acquire) &&
                             item.submit_returned.load(std::memory_order_acquire))) {
        item.occupied           = true;
        item.released           = false;
        item.completion_counted = false;
        item.enqueue_failed     = false;
        item.inflight_counted   = false;
        item.submit_returned.store(false, std::memory_order_relaxed);
        item.complete.store(false, std::memory_order_relaxed);
        item.rule = {};
        return &item;
      }
    }
    return nullptr;
  }

  bool registered_range(void* base, std::size_t size, off_t offset) const
  {
    auto const found = registrations.find(base);
    return found != registrations.end() && offset >= 0 &&
           static_cast<std::size_t>(offset) <= found->second &&
           size <= found->second - static_cast<std::size_t>(offset);
  }

  static void CUDART_CB prepare(void* context) noexcept
  {
    auto& item = *static_cast<transfer*>(context);
    auto& self = *item.owner;
    std::lock_guard lock(self.mutex);
    // cuFile reads these persistent pointer arguments when this stream reaches the operation.
    auto const size          = *item.size;
    auto const file_offset   = *item.file_offset;
    auto const buffer_offset = *item.buffer_offset;
    if (self.config.sample_arguments_at_submit &&
        (size != item.initial_size || file_offset != item.initial_file_offset ||
         buffer_offset != item.initial_buffer_offset))
      ++self.counts.argument_changes;
    auto& call               = self.calls[item.index];
    call.size                = size;
    call.file_offset         = file_offset;
    call.buffer_offset       = buffer_offset;
    item.copy->image         = self.device_image;
    item.copy->buffer        = static_cast<unsigned char*>(item.base);
    item.copy->write         = item.write;
    item.copy->file_offset   = 0;
    item.copy->buffer_offset = 0;
    item.copy->count         = 0;
    item.result              = -EINVAL;
    if (!valid_range(item.base, size, file_offset, buffer_offset)) return;
    if (self.config.strict_registration && !self.registered_range(item.base, size, buffer_offset)) {
      ++self.counts.extent_errors;
      return;
    }
    auto const offset = static_cast<std::size_t>(file_offset);
    auto count        = size;
    if (!item.write) {
      count = std::min(count, offset < self.image_size ? self.image_size - offset : 0);
    }
    if (item.rule.completion_result) {
      if (*item.rule.completion_result < 0) {
        item.result = *item.rule.completion_result;
        return;
      }
      count = std::min(count, static_cast<std::size_t>(*item.rule.completion_result));
    }
    if (item.write && (offset > self.image_capacity || count > self.image_capacity - offset)) {
      item.result = -ENOSPC;
      return;
    }
    item.result              = static_cast<ssize_t>(count);
    item.copy->file_offset   = offset;
    item.copy->buffer_offset = static_cast<std::size_t>(buffer_offset);
    item.copy->count         = count;
  }

  static void CUDART_CB finish(void* context) noexcept
  {
    auto& item = *static_cast<transfer*>(context);
    auto& self = *item.owner;
    {
      std::lock_guard lock(self.mutex);
      if (item.write && item.copy->count != 0)
        self.image_size = std::max(self.image_size, item.copy->file_offset + item.copy->count);
      if (item.write && item.result > 0 && self.config.mirror_write_fd >= 0) {
        std::size_t written{};
        while (written < static_cast<std::size_t>(item.result)) {
          auto const count = pwrite(self.config.mirror_write_fd,
                                    self.image + item.copy->file_offset + written,
                                    static_cast<std::size_t>(item.result) - written,
                                    static_cast<off_t>(item.copy->file_offset + written));
          if (count < 0 && errno == EINTR) continue;
          if (count <= 0) {
            item.result = -EIO;
            break;
          }
          written += static_cast<std::size_t>(count);
        }
      }
      if (!item.rule.preserve_bytes_done) *item.bytes_done = item.result;
      ++self.counts.completed;
      if (item.inflight_counted) {
        --self.counts.current_inflight;
        item.inflight_counted = false;
      }
      if (item.result >= 0) {
        auto& bytes = item.write ? self.counts.bytes_written : self.counts.bytes_read;
        bytes += static_cast<std::size_t>(item.result);
      }
    }
    // Last access: a subsequent submit may recycle metadata after this release publication.
    item.complete.store(true, std::memory_order_release);
  }
};

transport_simulator::transport_simulator(std::vector<std::byte> image)
  : transport_simulator(std::move(image), options{})
{
}

transport_simulator::transport_simulator(std::vector<std::byte> image, options config)
  : _state(std::make_unique<state>())
{
  _state->initialize(std::move(image), config);
}

transport_simulator::~transport_simulator() noexcept
{
  unblock_all();
  // Destruction is a test teardown drain, never a production hot-path operation.
  int previous     = 0;
  auto const saved = cudaGetDevice(&previous);
  bool quiescent   = true;
  for (std::size_t i = 0; i < _state->config.max_inflight; ++i) {
    auto const& item = _state->transfers[i];
    if (item.occupied && !item.complete.load(std::memory_order_acquire)) {
      if (cudaSetDevice(item.device) != cudaSuccess ||
          cudaStreamSynchronize(item.stream) != cudaSuccess)
        quiescent = false;
    }
  }
  if (saved == cudaSuccess) static_cast<void>(cudaSetDevice(previous));
  // Retain potentially live callback/kernel targets when CUDA cannot prove quiescence.
  if (!quiescent) static_cast<void>(_state.release());
}

void transport_simulator::add_fault(fault rule)
{
  if ((rule.file_offset && (*rule.file_offset < 0 || !_state->config.sample_arguments_at_submit)) ||
      rule.submit_latency.count() < 0 || rule.completion_latency.count() < 0 ||
      static_cast<std::uint64_t>(rule.completion_latency.count()) >
        std::numeric_limits<std::uint64_t>::max() / 1000)
    throw std::invalid_argument("negative simulator fault offset/latency");
  std::lock_guard lock(_state->mutex);
  _state->faults.push_back(std::move(rule));
}

void transport_simulator::clear_faults()
{
  std::lock_guard lock(_state->mutex);
  _state->faults.clear();
}

bool transport_simulator::unblock(std::size_t index) noexcept
{
  std::lock_guard lock(_state->mutex);
  for (std::size_t i = 0; i < _state->config.max_inflight; ++i) {
    auto& item = _state->transfers[i];
    if (item.occupied && item.index == index) {
      item.released = true;
      if (item.copy != nullptr)
        std::atomic_ref(item.copy->released).store(1, std::memory_order_release);
      if (item.completion_counted) {
        --_state->counts.blocked_completions;
        item.completion_counted = false;
      }
      _state->cv.notify_all();
      return true;
    }
  }
  return false;
}

void transport_simulator::unblock_all() noexcept
{
  std::lock_guard lock(_state->mutex);
  for (std::size_t i = 0; i < _state->config.max_inflight; ++i) {
    auto& item    = _state->transfers[i];
    item.released = true;
    if (item.copy != nullptr)
      std::atomic_ref(item.copy->released).store(1, std::memory_order_release);
    item.completion_counted = false;
  }
  _state->counts.blocked_completions = 0;
  _state->cv.notify_all();
}

cudaError_t transport_simulator::recover_stream(cudaStream_t stream) noexcept
{
  int device          = 0;
  auto const selected = cudaGetDevice(&device);
  if (selected != cudaSuccess) return selected;
  auto const error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return error;
  std::lock_guard lock(_state->mutex);
  for (std::size_t i = 0; i < _state->config.max_inflight; ++i) {
    auto& item = _state->transfers[i];
    if (item.occupied && item.stream == stream && item.device == device && item.enqueue_failed)
      item.complete.store(true, std::memory_order_release);
  }
  return cudaSuccess;
}

transport_simulator::counters transport_simulator::snapshot() const
{
  std::lock_guard lock(_state->mutex);
  return _state->counts;
}

std::vector<transport_simulator::submission> transport_simulator::submissions() const
{
  std::lock_guard lock(_state->mutex);
  return _state->calls;
}

void transport_simulator::mirror_writes_to(int fd)
{
  std::lock_guard lock(_state->mutex);
  _state->config.mirror_write_fd = fd;
}

std::vector<std::byte> transport_simulator::image() const
{
  std::lock_guard lock(_state->mutex);
  if (_state->image_size == 0) return {};
  return {_state->image, _state->image + _state->image_size};
}

void transport_simulator::set_registration_status(status buffer_register,
                                                  status buffer_deregister,
                                                  status stream_register,
                                                  status stream_deregister)
{
  std::lock_guard lock(_state->mutex);
  _state->buffer_register   = buffer_register;
  _state->buffer_deregister = buffer_deregister;
  _state->stream_register   = stream_register;
  _state->stream_deregister = stream_deregister;
}

void transport_simulator::fail_record(std::size_t index, cudaError_t error)
{
  std::lock_guard lock(_state->mutex);
  _state->record_errors[index] = error;
}

void transport_simulator::fail_copy(std::size_t index, cudaError_t error)
{
  std::lock_guard lock(_state->mutex);
  _state->copy_errors[index] = error;
}

void transport_simulator::fail_wait(std::size_t index, cudaError_t error)
{
  std::lock_guard lock(_state->mutex);
  _state->wait_errors[index] = error;
}

device_arena_hooks transport_simulator::arena_hooks(device_arena_hooks hooks)
{
  hooks.register_buffer = [this](int, void* base, std::size_t bytes) {
    return register_buffer(base, bytes).ok();
  };
  hooks.deregister_buffer = [this](int, void* base) { return deregister_buffer(base).ok(); };
  return hooks;
}

cufile_transport::status transport_simulator::register_buffer(void* base,
                                                              std::size_t size,
                                                              int) noexcept
{
  std::lock_guard lock(_state->mutex);
  ++_state->counts.buffer_registrations;
  if (!_state->buffer_register.ok()) return _state->buffer_register;
  if (base == nullptr || size == 0 ||
      reinterpret_cast<std::uintptr_t>(base) > std::numeric_limits<std::uintptr_t>::max() - size)
    return {-EINVAL};
  try {
    if (!_state->registrations.emplace(base, size).second) return {-EEXIST};
  } catch (...) {
    return {-ENOMEM};
  }
  return {};
}

cufile_transport::status transport_simulator::deregister_buffer(void* base) noexcept
{
  std::lock_guard lock(_state->mutex);
  ++_state->counts.buffer_deregistrations;
  if (!_state->buffer_deregister.ok()) return _state->buffer_deregister;
  return _state->registrations.erase(base) == 0 ? status{-EINVAL} : status{};
}

cufile_transport::status transport_simulator::register_stream(cudaStream_t, unsigned) noexcept
{
  std::lock_guard lock(_state->mutex);
  ++_state->counts.stream_registrations;
  return _state->stream_register;
}

cufile_transport::status transport_simulator::deregister_stream(cudaStream_t) noexcept
{
  std::lock_guard lock(_state->mutex);
  ++_state->counts.stream_deregistrations;
  return _state->stream_deregister;
}

cudaError_t transport_simulator::record_event(cudaEvent_t event, cudaStream_t stream) noexcept
{
  {
    std::lock_guard lock(_state->mutex);
    auto const index  = _state->counts.event_records++;
    auto const global = _state->record_errors.find(index);
    auto const local  = _state->stream_record_errors.find(stream);
    auto error        = global != _state->record_errors.end()         ? global->second
                        : local != _state->stream_record_errors.end() ? local->second
                                                                      : cudaSuccess;
    if (local != _state->stream_record_errors.end()) _state->stream_record_errors.erase(local);
    if (error != cudaSuccess) {
      ++_state->counts.record_errors;
      return error;
    }
  }
  auto const result = cudaEventRecord(event, stream);
  if (result != cudaSuccess) {
    std::lock_guard lock(_state->mutex);
    ++_state->counts.record_errors;
  }
  return result;
}

cudaError_t transport_simulator::memcpy_async(
  void* dst, void const* src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) noexcept
{
  {
    std::lock_guard lock(_state->mutex);
    auto const found = _state->copy_errors.find(_state->counts.copy_calls++);
    if (found != _state->copy_errors.end() && found->second != cudaSuccess) {
      ++_state->counts.copy_errors;
      return found->second;
    }
  }
  auto const result = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  if (result != cudaSuccess) {
    std::lock_guard lock(_state->mutex);
    ++_state->counts.copy_errors;
  }
  return result;
}

cudaError_t transport_simulator::stream_wait_event(cudaStream_t stream, cudaEvent_t event) noexcept
{
  {
    std::lock_guard lock(_state->mutex);
    auto const found = _state->wait_errors.find(_state->counts.wait_calls++);
    if (found != _state->wait_errors.end() && found->second != cudaSuccess) {
      ++_state->counts.wait_errors;
      return found->second;
    }
  }
  auto const result = cudaStreamWaitEvent(stream, event, 0);
  if (result != cudaSuccess) {
    std::lock_guard lock(_state->mutex);
    ++_state->counts.wait_errors;
  }
  return result;
}

void transport_simulator::set_sync_result(ssize_t result) noexcept
{
  _state->sync_result.store(result);
}

ssize_t transport_simulator::read_sync(
  void*, void* base, std::size_t size, off_t file_offset, off_t buffer_offset) noexcept
{
  if (!valid_range(base, size, file_offset, buffer_offset)) return -EINVAL;
  auto const forced = _state->sync_result.load();
  if (forced != std::numeric_limits<ssize_t>::min()) return forced;
  try {
    auto contents     = image();
    auto const offset = static_cast<std::size_t>(file_offset);
    if (offset >= contents.size()) return 0;
    auto const count = std::min(size, contents.size() - offset);
    cudaStream_t stream{};
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return -EIO;
    auto const copied = cudaMemcpyAsync(static_cast<std::byte*>(base) + buffer_offset,
                                        contents.data() + offset,
                                        count,
                                        cudaMemcpyHostToDevice,
                                        stream);
    auto const status = cudaStreamSynchronize(stream);
    static_cast<void>(cudaStreamDestroy(stream));
    return copied == cudaSuccess && status == cudaSuccess ? static_cast<ssize_t>(count) : -EIO;
  } catch (...) {
    return -ENOMEM;
  }
}

cufile_transport::status transport_simulator::read_async(void*,
                                                         void* base,
                                                         std::size_t* size,
                                                         off_t* file_offset,
                                                         off_t* buffer_offset,
                                                         ssize_t* bytes_done,
                                                         cudaStream_t stream) noexcept
{
  return submit(false, base, size, file_offset, buffer_offset, bytes_done, stream);
}

cufile_transport::status transport_simulator::write_async(void*,
                                                          void* base,
                                                          std::size_t* size,
                                                          off_t* file_offset,
                                                          off_t* buffer_offset,
                                                          ssize_t* bytes_done,
                                                          cudaStream_t stream) noexcept
{
  return submit(true, base, size, file_offset, buffer_offset, bytes_done, stream);
}

cufile_transport::status transport_simulator::submit(bool write,
                                                     void* base,
                                                     std::size_t* size,
                                                     off_t* file_offset,
                                                     off_t* buffer_offset,
                                                     ssize_t* bytes_done,
                                                     cudaStream_t stream) noexcept
{
  state::submit_scope active_call{*_state};
  if (base == nullptr || size == nullptr || file_offset == nullptr || buffer_offset == nullptr ||
      bytes_done == nullptr ||
      (_state->config.sample_arguments_at_submit &&
       !valid_range(base, *size, *file_offset, *buffer_offset)))
    return {-EINVAL};
  state::transfer* work = nullptr;
  try {
    {
      std::unique_lock lock(_state->mutex);
      if (_state->registrations.find(base) == _state->registrations.end()) ++_state->counts.bounce;
      if (_state->config.strict_registration &&
          (_state->registrations.find(base) == _state->registrations.end() ||
           (_state->config.sample_arguments_at_submit &&
            !_state->registered_range(base, *size, *buffer_offset)))) {
        ++_state->counts.extent_errors;
        ++_state->counts.submit_errors;
        return {-EINVAL};
      }
      work = _state->acquire();
      if (work == nullptr) {
        ++_state->counts.submit_errors;
        return {-EAGAIN};
      }
      work->index                 = _state->calls.size();
      work->write                 = write;
      work->base                  = base;
      work->size                  = size;
      work->file_offset           = file_offset;
      work->buffer_offset         = buffer_offset;
      work->initial_size          = _state->config.sample_arguments_at_submit ? *size : 0;
      work->initial_file_offset   = _state->config.sample_arguments_at_submit ? *file_offset : 0;
      work->initial_buffer_offset = _state->config.sample_arguments_at_submit ? *buffer_offset : 0;
      work->bytes_done            = bytes_done;
      work->stream                = stream;
      _state->calls.push_back({work->index,
                               write,
                               work->initial_file_offset,
                               work->initial_buffer_offset,
                               work->initial_size,
                               std::this_thread::get_id()});
      ++(write ? _state->counts.write_calls : _state->counts.read_calls);
      for (auto const& rule : _state->faults) {
        if ((!rule.index || *rule.index == work->index) &&
            (!rule.file_offset || *rule.file_offset == work->initial_file_offset) &&
            (!rule.write || *rule.write == write)) {
          work->rule = rule;
          break;
        }
      }
      if (work->rule.stuck_submit) {
        ++_state->counts.blocked_submissions;
        _state->cv.wait(lock, [&] { return work->released; });
        --_state->counts.blocked_submissions;
      }
    }
    state::enqueue_scope retained_until_return{work};
    std::this_thread::sleep_for(work->rule.submit_latency);
    if (!work->rule.submit_status.ok()) {
      std::lock_guard lock(_state->mutex);
      ++_state->counts.submit_errors;
      work->complete.store(true, std::memory_order_release);
      return work->rule.submit_status;
    }
    auto error = cudaGetDevice(&work->device);
    if (error == cudaSuccess && work->copy == nullptr) error = cudaErrorNoDevice;
    unsigned stream_flags = 0;
    if (error == cudaSuccess) error = cudaStreamGetFlags(stream, &stream_flags);
    if (error != cudaSuccess) {
      std::lock_guard lock(_state->mutex);
      ++_state->counts.submit_errors;
      work->complete.store(true, std::memory_order_release);
      return {-EIO, static_cast<int>(error)};
    }
    {
      std::lock_guard lock(_state->mutex);
      if (work->rule.record_error != cudaSuccess)
        _state->stream_record_errors[stream] = work->rule.record_error;
      work->copy->count = 0;
      work->copy->completion_latency_ns =
        static_cast<std::uint64_t>(work->rule.completion_latency.count()) * 1000;
      std::atomic_ref(work->copy->released)
        .store(work->released || !work->rule.stuck_completion ? 1U : 0U, std::memory_order_release);
      if (work->rule.stuck_completion && !work->released) {
        work->completion_counted = true;
        ++_state->counts.blocked_completions;
      }
      ++_state->counts.submitted;
      work->inflight_counted = true;
      ++_state->counts.current_inflight;
      _state->counts.max_inflight =
        std::max(_state->counts.max_inflight, _state->counts.current_inflight);
    }
    bool enqueued = false;
    error         = launch_simulator_gate(work->device_copy, stream, &enqueued);
    if (error == cudaSuccess) error = work->rule.prepare_error;
    if (error == cudaSuccess) error = cudaLaunchHostFunc(stream, state::prepare, work);
    if (error == cudaSuccess) error = launch_simulator_copy(work->device_copy, stream);
    if (error == cudaSuccess) error = cudaLaunchHostFunc(stream, state::finish, work);
    if (error != cudaSuccess) {
      // Keep partial enqueues until teardown/recovery synchronizes their stream.
      std::lock_guard lock(_state->mutex);
      ++_state->counts.submit_errors;
      work->enqueue_failed = true;
      if (!enqueued) work->complete.store(true, std::memory_order_release);
      if (!enqueued && work->completion_counted) {
        --_state->counts.blocked_completions;
        work->completion_counted = false;
      }
      if (work->inflight_counted) {
        --_state->counts.current_inflight;
        work->inflight_counted = false;
      }
      return {-EIO, static_cast<int>(error)};
    }
    return {};
  } catch (...) {
    std::lock_guard lock(_state->mutex);
    ++_state->counts.submit_errors;
    // All throwing operations precede the first enqueue; release their metadata on retry.
    if (work != nullptr) {
      work->complete.store(true, std::memory_order_release);
      work->submit_returned.store(true, std::memory_order_release);
    }
    return {-ENOMEM};
  }
}

}  // namespace cucascade::io::cufile
