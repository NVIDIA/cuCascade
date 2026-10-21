/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "disk_io.hpp"

#include <cucascade/data/encoded_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/disk_table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade {
namespace {

using memory::Tier;

/// Chunk size of the pageable bounce buffer used between GPUs on different devices.
constexpr std::size_t staging_size = 1ull << 20;

/// Chunk size of the pageable bounce buffer of DISK legs that cannot transfer directly (between
/// two files, or from a GPU when the target io context cannot write device memory). Each chunk is
/// one io request, so it is larger than the GPU bounce buffer.
constexpr std::size_t disk_staging_size = 8ull << 20;

void validate_metadata(const encoded_data_metadata& metadata)
{
  if (metadata.encoding.empty()) {
    throw std::invalid_argument("encoded representation requires a nonempty encoding name");
  }
}

void validate_tier(const memory::memory_space& space, Tier expected)
{
  if (space.get_tier() != expected) {
    throw std::invalid_argument("encoded representation factory received the wrong memory tier");
  }
}

void validate_stream(::cuda::stream_ref stream, int expected_device)
{
  int device = -1;
  CUCASCADE_CUDA_TRY(cudaStreamGetDevice(stream.get(), &device));
  if (device != expected_device) {
    CUCASCADE_FAIL("encoded representation stream belongs to a different CUDA device");
  }
}

std::size_t allocation_size(const memory::memory_space& space, std::size_t size)
{
  if (space.get_tier() != Tier::HOST) { return size; }
  auto block_size = space.get_memory_resource_of<Tier::HOST>()->get_block_size();
  if (!std::has_single_bit(block_size)) {
    throw std::invalid_argument("encoded host payload requires a power-of-two host block size");
  }
  if (size == 0) { return 0; }
  if (size > std::numeric_limits<std::size_t>::max() - (block_size - 1)) {
    throw std::overflow_error("encoded host payload allocation size overflow");
  }
  return ((size + block_size - 1) / block_size) * block_size;
}

void validate_reservation(const memory::memory_space& space,
                          std::size_t size,
                          memory::reservation* reservation)
{
  if (reservation == nullptr) { return; }
  if (space.get_tier() == Tier::GPU) {
    throw std::invalid_argument(
      "GPU reservations must be attached to the allocation tracker; use the memory space overload");
  }
  if (&reservation->get_memory_space() != &space) {
    throw std::invalid_argument("encoded relocation reservation belongs to another memory space");
  }
  if (reservation->size() < allocation_size(space, size)) {
    throw std::invalid_argument("encoded relocation reservation does not cover its allocation");
  }
}

/**
 * Drain submitted transfers before a failed operation destroys their buffers.
 *
 * The destructor only synchronizes on an exceptional path (finish() was not reached or itself
 * failed), so it is best effort: a failed synchronize means the stream or context is already in
 * an error state, and aborting here would hide the exception being propagated.
 */
class transfer_guard {
 public:
  explicit transfer_guard(::cuda::stream_ref stream) : _stream(stream) {}
  ~transfer_guard()
  {
    if (_pending && cudaStreamSynchronize(_stream.get()) != cudaSuccess) {
      // Clear a non-sticky error so it is not reported by an unrelated later CUDA call.
      (void)cudaGetLastError();
    }
  }
  transfer_guard(const transfer_guard&)            = delete;
  transfer_guard& operator=(const transfer_guard&) = delete;
  transfer_guard(transfer_guard&&)                 = delete;
  transfer_guard& operator=(transfer_guard&&)      = delete;

  void submitted() noexcept { _pending = true; }
  void finish()
  {
    if (_pending) {
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(_stream.get()));
      _pending = false;
    }
  }

 private:
  ::cuda::stream_ref _stream;
  bool _pending{false};
};

struct disk_storage {
  std::filesystem::path path;
  std::size_t offset{0};
  bool owned{false};
};

/** Remove a newly created file unless a representation has adopted it. */
struct temporary_file {
  explicit temporary_file(std::string file_path) : path(std::move(file_path)) {}
  temporary_file(const temporary_file&)            = delete;
  temporary_file& operator=(const temporary_file&) = delete;
  ~temporary_file()
  {
    if (!adopted) {
      std::error_code error;
      std::filesystem::remove(path, error);  // Best effort; must not hide the original failure.
    }
  }

  std::string path;
  bool adopted{false};
};

/** One write slice per host block, storing the first @p size bytes at file offset 0. */
std::vector<io::write_slice> host_write_slices(
  const memory::fixed_multiple_blocks_allocation& blocks, std::size_t size)
{
  std::vector<io::write_slice> slices;
  std::size_t offset = 0;
  for (std::size_t i = 0; offset < size; ++i) {
    auto block = blocks->at(i);
    auto count = std::min(block.size(), size - offset);
    slices.emplace_back(offset, count, reinterpret_cast<const std::uint8_t*>(block.data()));
    offset += count;
  }
  return slices;
}

/** One read slice per host block, loading @p size bytes that start at @p file_offset. */
std::vector<io::slice> host_read_slices(const memory::fixed_multiple_blocks_allocation& blocks,
                                        std::size_t size,
                                        std::size_t file_offset)
{
  std::vector<io::slice> slices;
  std::size_t offset = 0;
  for (std::size_t i = 0; offset < size; ++i) {
    auto block = blocks->at(i);
    auto count = std::min(block.size(), size - offset);
    slices.emplace_back(file_offset + offset, count, reinterpret_cast<std::uint8_t*>(block.data()));
    offset += count;
  }
  return slices;
}

}  // namespace

struct encoded_data_representation::storage {
  // Reservations outlive allocations, including destruction of partial results.
  std::unique_ptr<memory::reservation> disk_reservation;
  std::variant<std::monostate,
               memory::fixed_multiple_blocks_allocation,
               std::unique_ptr<rmm::device_buffer>,
               disk_storage>
    buffer;
  cudaEvent_t writer_event{nullptr};
};

encoded_data_representation::encoded_data_representation(memory::memory_space& space,
                                                         std::size_t payload_size,
                                                         encoded_data_metadata metadata)
  : idata_representation(space),
    _payload_size(payload_size),
    _metadata(std::move(metadata)),
    _storage(std::make_unique<storage>())
{
}

encoded_data_representation::~encoded_data_representation() noexcept
{
  if (_storage->writer_event != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(_storage->writer_event));
  }
  if (auto* disk = std::get_if<disk_storage>(&_storage->buffer); disk != nullptr && disk->owned) {
    try {
      std::filesystem::remove(disk->path);
    } catch (...) {
      // File cleanup is best effort, including during exception unwinding.
    }
  }
  if (auto* device = std::get_if<std::unique_ptr<rmm::device_buffer>>(&_storage->buffer)) {
    // RMM resources and stream-ordered frees belong to the buffer's device.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
    device->reset();
  }
}

std::size_t encoded_data_representation::relocation_reservation_bytes(
  const memory::memory_space& space, std::size_t payload_size)
{
  return allocation_size(space, payload_size);
}

std::unique_ptr<encoded_data_representation> encoded_data_representation::allocate(
  memory::memory_space& space,
  std::size_t payload_size,
  encoded_data_metadata metadata,
  ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  validate_metadata(metadata);
  validate_reservation(space, payload_size, reservation);
  auto result = std::unique_ptr<encoded_data_representation>(
    new encoded_data_representation(space, payload_size, std::move(metadata)));
  switch (space.get_tier()) {
    case Tier::HOST:
      // Check rounding before passing the size to the host resource.
      (void)allocation_size(space, payload_size);
      result->_storage->buffer =
        space.get_memory_resource_of<Tier::HOST>()->allocate_multiple_blocks(payload_size,
                                                                             reservation);
      break;
    case Tier::GPU: {
      validate_stream(stream, space.get_device_id());
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space.get_device_id()}};
      result->_storage->buffer =
        std::make_unique<rmm::device_buffer>(payload_size, stream, space.get_default_allocator());
      break;
    }
    case Tier::DISK: {
      // A caller-owned reservation covers the file for its whole lifetime; otherwise the
      // representation owns one, released only after the file is removed.
      if (reservation == nullptr) {
        result->_storage->disk_reservation = space.make_reservation_or_null(payload_size);
        if (!result->_storage->disk_reservation) {
          throw rmm::out_of_memory("insufficient capacity for encoded disk payload");
        }
      }
      // generate_disk_file_path() creates a fresh empty file under the mount path; the payload is
      // written at offset 0, the aligned start every batch file buffer uses.
      auto mount_path =
        std::filesystem::absolute(std::filesystem::path(space.get_disk_mount_path()));
      temporary_file file{memory::generate_disk_file_path(mount_path.string())};
      result->_storage->buffer = disk_storage{std::filesystem::path(file.path), 0, true};
      file.adopted             = true;
      break;
    }
    default: throw std::invalid_argument("unsupported encoded representation memory tier");
  }
  return result;
}

std::unique_ptr<encoded_data_representation> encoded_data_representation::from_host(
  std::span<const std::byte> payload,
  encoded_data_metadata metadata,
  memory::memory_space& space,
  memory::reservation* reservation)
{
  validate_tier(space, Tier::HOST);
  if (!payload.empty() && payload.data() == nullptr) {
    throw std::invalid_argument("null encoded host payload");
  }
  auto result        = allocate(space,
                         payload.size(),
                         std::move(metadata),
                         ::cuda::stream_ref{cudaStream_t{nullptr}},
                         reservation);
  std::size_t offset = 0;
  auto& allocation   = result->get_host_buffer();
  for (std::size_t i = 0; offset < payload.size(); ++i) {
    auto block = allocation->at(i);
    auto count = std::min(block.size(), payload.size() - offset);
    std::memcpy(block.data(), payload.data() + offset, count);
    offset += count;
  }
  return result;
}

std::unique_ptr<encoded_data_representation> encoded_data_representation::from_device(
  const void* payload,
  std::size_t payload_size,
  encoded_data_metadata metadata,
  memory::memory_space& space,
  ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  validate_tier(space, Tier::GPU);
  validate_stream(stream, space.get_device_id());
  if (payload_size != 0) {
    if (payload == nullptr) { throw std::invalid_argument("null encoded device payload"); }
    cudaPointerAttributes attributes{};
    CUCASCADE_CUDA_TRY(cudaPointerGetAttributes(&attributes, payload));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != space.get_device_id()) {
      throw std::invalid_argument("encoded device payload must reside on the target GPU");
    }
  }
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space.get_device_id()}};
  auto result = allocate(space, payload_size, std::move(metadata), stream, reservation);
  transfer_guard transfers{stream};
  if (payload_size != 0) {
    transfers.submitted();
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      result->device_data(), payload, payload_size, cudaMemcpyDeviceToDevice, stream.get()));
  }
  transfers.finish();
  result->record_writer_event(stream);
  return result;
}

std::unique_ptr<encoded_data_representation> encoded_data_representation::from_disk(
  std::filesystem::path path,
  std::size_t file_offset,
  std::size_t payload_size,
  encoded_data_metadata metadata,
  memory::memory_space& space,
  encoded_file_ownership ownership)
{
  validate_tier(space, Tier::DISK);
  validate_metadata(metadata);
  if (ownership != encoded_file_ownership::borrowed && ownership != encoded_file_ownership::owned) {
    throw std::invalid_argument("invalid encoded file ownership");
  }
  if (file_offset > std::numeric_limits<std::size_t>::max() - payload_size) {
    throw std::overflow_error("encoded disk range overflow");
  }
  if (file_offset + payload_size > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
    throw std::overflow_error("encoded disk range exceeds supported file offsets");
  }
  // Keep validation, later reads, and deletion tied to the same path across cwd changes.
  path = std::filesystem::absolute(path);
  if (!std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument("encoded disk payload requires a regular file");
  }
  if (file_offset + payload_size > std::filesystem::file_size(path)) {
    throw std::invalid_argument("encoded disk payload range exceeds its file");
  }
  auto result = std::unique_ptr<encoded_data_representation>(
    new encoded_data_representation(space, payload_size, std::move(metadata)));
  result->_storage->disk_reservation = space.make_reservation_or_null(payload_size);
  if (!result->_storage->disk_reservation) {
    throw rmm::out_of_memory("insufficient capacity for imported encoded disk payload");
  }
  result->_storage->buffer =
    disk_storage{std::move(path), file_offset, ownership == encoded_file_ownership::owned};
  return result;
}

std::size_t encoded_data_representation::get_size_in_bytes() const
{
  if (get_current_tier() == Tier::HOST) { return get_host_buffer()->size_bytes(); }
  return _payload_size;
}

std::size_t encoded_data_representation::get_uncompressed_data_size_in_bytes() const
{
  return _metadata.uncompressed_size;
}

std::size_t encoded_data_representation::get_payload_size() const noexcept { return _payload_size; }

const encoded_data_metadata& encoded_data_representation::get_metadata() const noexcept
{
  return _metadata;
}

const memory::fixed_multiple_blocks_allocation& encoded_data_representation::get_host_buffer() const
{
  validate_tier(get_memory_space(), Tier::HOST);
  return std::get<memory::fixed_multiple_blocks_allocation>(_storage->buffer);
}

const rmm::device_buffer& encoded_data_representation::get_device_buffer() const
{
  validate_tier(get_memory_space(), Tier::GPU);
  return *std::get<std::unique_ptr<rmm::device_buffer>>(_storage->buffer);
}

void* encoded_data_representation::device_data()
{
  return std::get<std::unique_ptr<rmm::device_buffer>>(_storage->buffer)->data();
}

const std::filesystem::path& encoded_data_representation::get_file_path() const
{
  validate_tier(get_memory_space(), Tier::DISK);
  return std::get<disk_storage>(_storage->buffer).path;
}

std::size_t encoded_data_representation::get_file_offset() const
{
  validate_tier(get_memory_space(), Tier::DISK);
  return std::get<disk_storage>(_storage->buffer).offset;
}

void encoded_data_representation::wait_for_writer() const
{
  if (get_current_tier() != Tier::GPU) { return; }
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
  if (_storage->writer_event != nullptr) {
    CUCASCADE_CUDA_TRY(cudaEventSynchronize(_storage->writer_event));
  } else {
    CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  }
}

std::unique_ptr<idata_representation> encoded_data_representation::relocate(
  const memory::memory_space* target,
  ::cuda::stream_ref stream,
  memory::reservation* reservation) const
{
  CUCASCADE_FUNC_RANGE();
  if (target == nullptr) { throw std::invalid_argument("null encoded relocation target"); }
  auto source_tier = get_current_tier();
  auto target_tier = target->get_tier();
  if (target_tier == Tier::GPU) {
    validate_stream(stream, target->get_device_id());
  } else if (source_tier == Tier::GPU) {
    validate_stream(stream, get_device_id());
  }
  validate_reservation(*target, _payload_size, reservation);
  wait_for_writer();
  // The idata_representation::relocate() contract takes a const target, but allocating in a space
  // draws from its (mutable) allocator, stream pool, and reservation accounting. Memory space
  // objects are not themselves const, so the cast is well defined.
  auto result = allocate(
    *const_cast<memory::memory_space*>(target), _payload_size, _metadata, stream, reservation);

  // Device operations below switch to their owning context; moves between HOST and DISK spaces
  // never touch CUDA.
  auto complete_device = [&] {
    if (target_tier == Tier::GPU) { result->record_writer_event(stream); }
  };
  if (_payload_size == 0) {
    complete_device();
    return result;
  }

  if (source_tier == Tier::HOST && target_tier == Tier::HOST) {
    auto& source_buffer = get_host_buffer();
    auto& target_buffer = result->get_host_buffer();
    std::size_t offset  = 0;
    while (offset < _payload_size) {
      auto source_block  = source_buffer->at(offset / source_buffer->block_size());
      auto target_block  = target_buffer->at(offset / target_buffer->block_size());
      auto source_offset = offset % source_buffer->block_size();
      auto target_offset = offset % target_buffer->block_size();
      auto count         = std::min({_payload_size - offset,
                                     source_block.size() - source_offset,
                                     target_block.size() - target_offset});
      std::memcpy(target_block.data() + target_offset, source_block.data() + source_offset, count);
      offset += count;
    }
    return result;
  }

  if (source_tier == Tier::DISK || target_tier == Tier::DISK) {
    relocate_disk(*result, stream);
    complete_device();
    return result;
  }

  // HOST endpoints are already pinned and allocated through the memory-space resource. Avoid an
  // extra copy in their direct GPU paths, and wait before either endpoint can be released.
  if (source_tier == Tier::HOST && target_tier == Tier::GPU) {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target->get_device_id()}};
    transfer_guard transfers{stream};
    std::size_t offset = 0;
    for (std::size_t i = 0; offset < _payload_size; ++i) {
      auto block = get_host_buffer()->at(i);
      auto count = std::min(block.size(), _payload_size - offset);
      transfers.submitted();
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<std::byte*>(result->device_data()) + offset,
                                         block.data(),
                                         count,
                                         cudaMemcpyHostToDevice,
                                         stream.get()));
      offset += count;
    }
    transfers.finish();
    complete_device();
    return result;
  }
  if (source_tier == Tier::GPU && target_tier == Tier::HOST) {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
    transfer_guard transfers{stream};
    std::size_t offset = 0;
    for (std::size_t i = 0; offset < _payload_size; ++i) {
      auto block = result->get_host_buffer()->at(i);
      auto count = std::min(block.size(), _payload_size - offset);
      transfers.submitted();
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(block.data(),
                        static_cast<const std::byte*>(get_device_buffer().data()) + offset,
                        count,
                        cudaMemcpyDeviceToHost,
                        stream.get()));
      offset += count;
    }
    transfers.finish();
    return result;
  }

  if (source_tier != Tier::GPU || target_tier != Tier::GPU) {
    throw std::invalid_argument("unsupported encoded relocation tier pair");
  }
  if (get_device_id() == target->get_device_id()) {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
    transfer_guard transfers{stream};
    transfers.submitted();
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(result->device_data(),
                                       get_device_buffer().data(),
                                       _payload_size,
                                       cudaMemcpyDeviceToDevice,
                                       stream.get()));
    transfers.finish();
    complete_device();
    return result;
  }

  // Between devices, bounce through bounded pageable staging rather than an untracked pinned
  // allocation, so the copy also works for GPU pairs without peer access. The caller's stream
  // belongs to the destination GPU; the device-to-host leg uses a stream of the source GPU, after
  // the source writer has completed. Each leg completes before the staging buffer is reused.
  std::vector<std::byte> staging(std::min(staging_size, _payload_size));
  auto source_stream = get_memory_space().acquire_stream();
  for (std::size_t offset = 0; offset < _payload_size;) {
    auto count = std::min(staging.size(), _payload_size - offset);
    {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
      transfer_guard transfers{source_stream};
      transfers.submitted();
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(staging.data(),
                        static_cast<const std::byte*>(get_device_buffer().data()) + offset,
                        count,
                        cudaMemcpyDeviceToHost,
                        source_stream.get()));
      transfers.finish();
    }
    {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target->get_device_id()}};
      transfer_guard transfers{stream};
      transfers.submitted();
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<std::byte*>(result->device_data()) + offset,
                                         staging.data(),
                                         count,
                                         cudaMemcpyHostToDevice,
                                         stream.get()));
      transfers.finish();
    }
    offset += count;
  }
  complete_device();
  return result;
}

void encoded_data_representation::relocate_disk(encoded_data_representation& result,
                                                ::cuda::stream_ref stream) const
{
  // Every leg goes through the io context of the DISK space(s) involved, and each helper waits for
  // its request before it returns or unwinds. So when this function is left, normally or by an
  // exception, no I/O touches either payload or either file any more, and a failed copy can be
  // destroyed (removing its file) without racing the transfer.
  auto source_tier = get_current_tier();
  auto target_tier = result.get_current_tier();
  auto& target     = result.get_memory_space();

  if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
    auto context = detail::resolve_disk_io_context(target);
    auto slices  = host_write_slices(get_host_buffer(), _payload_size);
    detail::write_disk_file_host(*context, result.get_file_path().string(), slices, _payload_size);
    return;
  }
  if (source_tier == Tier::DISK && target_tier == Tier::HOST) {
    auto context = detail::resolve_disk_io_context(get_memory_space());
    auto slices  = host_read_slices(result.get_host_buffer(), _payload_size, get_file_offset());
    detail::read_disk_file_host(*context, get_file_path().string(), slices);
    return;
  }
  if (source_tier == Tier::DISK && target_tier == Tier::GPU) {
    auto context = detail::resolve_disk_io_context(get_memory_space());
    // The io context binds device destinations to the current device. The destination is a
    // stream-ordered allocation on `stream`; the read is ordered after the work enqueued on
    // `stream` before it, and its bytes are in device memory once the helper returns, so the
    // writer event recorded on `stream` afterwards covers them.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target.get_device_id()}};
    std::array<io::slice, 1> slices{io::slice{
      get_file_offset(), _payload_size, static_cast<std::uint8_t*>(result.device_data())}};
    detail::read_disk_file_device(*context, get_file_path().string(), slices, stream);
    return;
  }
  if (source_tier == Tier::GPU && target_tier == Tier::DISK) {
    auto context    = detail::resolve_disk_io_context(target);
    const auto path = result.get_file_path().string();
    // The io context binds device sources to the current device. The source writer has completed
    // (wait_for_writer), and `stream` belongs to the source device.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
    const auto* source = static_cast<const std::uint8_t*>(get_device_buffer().data());
    if (context->supports_device_write()) {
      std::array<io::write_slice, 1> slices{io::write_slice{0, _payload_size, source}};
      detail::write_disk_file_device(*context, path, slices, stream, _payload_size);
      return;
    }
    // The context can only write host memory: stage through bounded pageable memory, completing
    // each device-to-host copy before its chunk is written and the buffer reused.
    std::vector<std::byte> staging(std::min(disk_staging_size, _payload_size));
    for (std::size_t offset = 0; offset < _payload_size;) {
      auto count = std::min(staging.size(), _payload_size - offset);
      transfer_guard transfers{stream};
      transfers.submitted();
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        staging.data(), source + offset, count, cudaMemcpyDeviceToHost, stream.get()));
      transfers.finish();
      std::array<io::write_slice, 1> slices{
        io::write_slice{offset, count, reinterpret_cast<const std::uint8_t*>(staging.data())}};
      detail::write_disk_file_host(*context, path, slices, _payload_size);
      offset += count;
    }
    return;
  }
  if (source_tier == Tier::DISK && target_tier == Tier::DISK) {
    // Between two files (possibly of the same space, as for clone()), through bounded pageable
    // staging: each chunk is read completely before it is written.
    auto source_context    = detail::resolve_disk_io_context(get_memory_space());
    auto target_context    = detail::resolve_disk_io_context(target);
    const auto source_path = get_file_path().string();
    const auto target_path = result.get_file_path().string();
    std::vector<std::byte> staging(std::min(disk_staging_size, _payload_size));
    auto* buffer = reinterpret_cast<std::uint8_t*>(staging.data());
    for (std::size_t offset = 0; offset < _payload_size;) {
      auto count = std::min(staging.size(), _payload_size - offset);
      std::array<io::slice, 1> reads{io::slice{get_file_offset() + offset, count, buffer}};
      detail::read_disk_file_host(*source_context, source_path, reads);
      std::array<io::write_slice, 1> writes{io::write_slice{offset, count, buffer}};
      detail::write_disk_file_host(*target_context, target_path, writes, _payload_size);
      offset += count;
    }
    return;
  }
  throw std::invalid_argument("unsupported encoded relocation tier pair");
}

std::unique_ptr<idata_representation> encoded_data_representation::clone(
  ::cuda::stream_ref stream) const
{
  return relocate(&get_memory_space(), stream);
}

void encoded_data_representation::record_writer_event(::cuda::stream_ref writer_stream)
{
  if (get_current_tier() != Tier::GPU) { return; }
  validate_stream(writer_stream, get_device_id());
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{get_device_id()}};
  cudaEvent_t event = std::exchange(_storage->writer_event, nullptr);
  try {
    if (event == nullptr) {
      CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUCASCADE_CUDA_TRY(cudaEventRecord(event, writer_stream.get()));
    _storage->writer_event = event;
  } catch (...) {
    if (event != nullptr) { CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(event)); }
    throw;
  }
}

cudaEvent_t encoded_data_representation::get_writer_event() const { return _storage->writer_event; }

void encoded_data_representation::rebind_stream(::cuda::stream_ref stream)
{
  if (get_current_tier() != Tier::GPU) { return; }
  validate_stream(stream, get_device_id());
  std::get<std::unique_ptr<rmm::device_buffer>>(_storage->buffer)->set_stream(stream);
}

}  // namespace cucascade
