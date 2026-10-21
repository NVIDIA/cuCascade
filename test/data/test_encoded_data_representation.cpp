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

#include "utils/disk_test_utils.hpp"
#include "utils/test_memory_resources.hpp"

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/data/encoded_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace cucascade;

namespace {

constexpr std::size_t block_size = 4096;

memory::host_memory_space_config host_config(int id = -1, std::size_t capacity = 1ull << 20)
{
  memory::host_memory_space_config config;
  config.numa_id                    = id;
  config.memory_capacity            = capacity;
  config.reservation_limit_fraction = 1.0;
  config.block_size                 = block_size;
  config.pool_size                  = 4;
  config.initial_number_pools       = 0;
  config.mr_factory_fn              = [](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
      memory::numa_region_pinned_host_memory_resource{-1, true}};
  };
  return config;
}

memory::gpu_memory_space_config gpu_config(int id = 0)
{
  memory::gpu_memory_space_config config;
  config.device_id       = id;
  config.memory_capacity = 32ull << 20;
  config.mr_factory_fn   = cucascade::test::make_shared_current_device_resource;
  return config;
}

struct temp_directory {
  temp_directory()
  {
    auto name = memory::generate_disk_file_path(std::filesystem::temp_directory_path().string());
    std::filesystem::remove(name);
    path = name;
    std::filesystem::create_directory(path);
  }
  ~temp_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  temp_directory(const temp_directory&)            = delete;
  temp_directory& operator=(const temp_directory&) = delete;
  std::filesystem::path path;
};

/// Write @p bytes at @p offset of @p path with POSIX pwrite, creating the file if needed.
void write_file(const std::filesystem::path& path,
                const std::vector<std::byte>& bytes,
                std::size_t offset = 0)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open"); }
  std::size_t done = 0;
  while (done < bytes.size()) {
    const auto written =
      ::pwrite(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(offset + done));
    if (written < 0) {
      const int error = errno;
      ::close(fd);
      throw std::system_error(error, std::generic_category(), "pwrite");
    }
    done += static_cast<std::size_t>(written);
  }
  ::close(fd);
}

/// Read @p size bytes at @p offset of @p path with POSIX pread.
std::vector<std::byte> read_file(const std::filesystem::path& path,
                                 std::size_t offset,
                                 std::size_t size)
{
  std::vector<std::byte> bytes(size);
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open"); }
  std::size_t done = 0;
  while (done < size) {
    const auto count =
      ::pread(fd, bytes.data() + done, size - done, static_cast<off_t>(offset + done));
    if (count <= 0) {
      const int error = count < 0 ? errno : EIO;
      ::close(fd);
      throw std::system_error(error, std::generic_category(), "pread");
    }
    done += static_cast<std::size_t>(count);
  }
  ::close(fd);
  return bytes;
}

std::size_t file_count(const std::filesystem::path& directory)
{
  return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator{directory},
                                                std::filesystem::directory_iterator{}));
}

/// A DISK memory space under @p directory without an io context (the converters adopt the
/// process-shared default).
std::unique_ptr<memory::memory_space> make_disk(const std::filesystem::path& directory,
                                                std::size_t capacity = 1ull << 24,
                                                int id               = 0)
{
  return std::make_unique<memory::memory_space>(
    memory::disk_memory_space_config{id, capacity, directory.string()});
}

/**
 * @brief An io context that forwards to a real one, with switches to fail requests or to hide
 *        device reads or (device) writes, and counters of what reached the real context.
 */
class scripted_ioctx final : public io::ioctx {
 public:
  explicit scripted_ioctx(std::shared_ptr<io::ioctx> inner) : _inner(std::move(inner)) {}

  [[nodiscard]] io::io_context_type type() const noexcept override { return _inner->type(); }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    return _inner->supports(path);
  }
  [[nodiscard]] bool supports_device_read() const noexcept override
  {
    return device_read && _inner->supports_device_read();
  }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override
  {
    return _inner->supports_host_to_device_read();
  }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override
  {
    return _inner->supports_vector_host_read();
  }
  [[nodiscard]] bool supports_device_range_read() const noexcept override
  {
    return _inner->supports_device_range_read();
  }
  [[nodiscard]] bool supports_write() const noexcept override
  {
    return write && _inner->supports_write();
  }
  [[nodiscard]] bool supports_device_write() const noexcept override
  {
    return device_write && _inner->supports_device_write();
  }
  [[nodiscard]] std::vector<io::byte_range> align_and_coalesce(
    std::span<const io::byte_range> ranges, std::optional<size_t> alignment) const noexcept override
  {
    return _inner->align_and_coalesce(ranges, alignment);
  }
  size_t host_read_io(const io::io_object& object,
                      size_t offset,
                      size_t size,
                      uint8_t* destination) override
  {
    return _inner->host_read_io(object, offset, size, destination);
  }
  exec::semi_future<size_t> mixed_readv_async_io(
    const io::io_object& object, std::vector<io::prepared_io_slice>&& slices) noexcept override
  {
    if (fail_read) { return failure("injected read failure"); }
    return _inner->mixed_readv_async_io(object, std::move(slices));
  }
  exec::semi_future<std::size_t> mixed_writev_async_io(
    const io::io_object& object, std::vector<io::prepared_io_slice>&& slices) noexcept override
  {
    for (const auto& slice : slices) {
      if (slice.has_device_request()) { device_write_slices.fetch_add(1); }
    }
    auto written = _inner->mixed_writev_async_io(object, std::move(slices));
    if (!fail_write) { return written; }
    // Let the bytes land first, so the failed copy leaves a written file behind to clean up.
    try {
      (void)std::move(written).get();
    } catch (...) {
      return exec::make_semi_future<std::size_t>(std::current_exception());
    }
    return failure("injected write failure");
  }

  std::atomic<bool> fail_read{false};
  std::atomic<bool> fail_write{false};
  std::atomic<bool> device_read{true};
  std::atomic<bool> write{true};
  std::atomic<bool> device_write{true};
  std::atomic<std::size_t> device_write_slices{0};
  std::atomic<std::size_t> write_opens{0};

 protected:
  std::shared_ptr<io::io_object> create_io_object(std::string path) override
  {
    return _inner->open_io_object(std::move(path));
  }
  std::shared_ptr<io::io_object> create_io_object_for_write(std::string path,
                                                            io::write_open_options options) override
  {
    write_opens.fetch_add(1);
    return _inner->open_io_object_for_write(std::move(path), options);
  }

 private:
  static exec::semi_future<std::size_t> failure(const char* what)
  {
    return exec::make_semi_future<std::size_t>(std::make_exception_ptr(std::runtime_error(what)));
  }

  std::shared_ptr<io::ioctx> _inner;
};

encoded_data_metadata make_metadata()
{
  return {"test/opaque-rle", 17, 999999, {std::byte{0}, std::byte{255}, std::byte{42}}};
}

std::vector<std::byte> payload(std::size_t size)
{
  std::vector<std::byte> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::byte>((i * 37 + i / 251) % 256);
  }
  return bytes;
}

/// The payload of a representation; a DISK payload is read with POSIX pread.
std::vector<std::byte> read_bytes(const encoded_data_representation& representation)
{
  std::vector<std::byte> result(representation.get_payload_size());
  if (result.empty()) { return result; }
  if (representation.get_current_tier() == memory::Tier::HOST) {
    const auto& blocks = representation.get_host_buffer();
    std::size_t offset = 0;
    for (std::size_t i = 0; i < blocks->size() && offset < result.size(); ++i) {
      const auto block = blocks->at(i);
      const auto count = std::min(block.size(), result.size() - offset);
      std::memcpy(result.data() + offset, block.data(), count);
      offset += count;
    }
  } else if (representation.get_current_tier() == memory::Tier::DISK) {
    result =
      read_file(representation.get_file_path(), representation.get_file_offset(), result.size());
  } else {
    REQUIRE(representation.get_current_tier() == memory::Tier::GPU);
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{representation.get_device_id()}};
    CUCASCADE_CUDA_TRY(cudaMemcpy(result.data(),
                                  representation.get_device_buffer().data(),
                                  result.size(),
                                  cudaMemcpyDeviceToHost));
  }
  return result;
}

}  // namespace

TEST_CASE("encoded payload preserves every byte across all tier pairs", "[encoded][relocation]")
{
  const auto backend = GENERATE(from_range(test::disk_test_backends));
  CAPTURE(test::to_string(backend));
  temp_directory directory;
  auto disk = test::make_disk_space(backend, 0, directory.path.string());
  if (disk == nullptr) { SKIP("the " << test::to_string(backend) << " backend cannot run here"); }
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  std::array<memory::memory_space*, 3> spaces{&host, &gpu, disk.get()};
  rmm::cuda_stream stream;
  const auto metadata = make_metadata();

  // Below, at, and just past one I/O block, and an unaligned multi-block payload.
  for (const auto size : {0u, 1u, 4095u, 4096u, 4097u, 12305u}) {
    const auto bytes = payload(size);
    auto seed        = encoded_data_representation::from_host(bytes, metadata, host);
    for (auto* source_space : spaces) {
      auto source = seed->relocate(source_space, stream);
      for (auto* target_space : spaces) {
        INFO("size=" << size << " source=" << static_cast<int>(source_space->get_tier())
                     << " target=" << static_cast<int>(target_space->get_tier()));
        auto result         = source->relocate(target_space, stream);
        const auto& encoded = result->cast<encoded_data_representation>();
        REQUIRE(&encoded.get_memory_space() == target_space);
        REQUIRE(encoded.get_metadata() == metadata);
        REQUIRE(encoded.get_payload_size() == bytes.size());
        REQUIRE(encoded.get_uncompressed_data_size_in_bytes() == metadata.uncompressed_size);
        REQUIRE(read_bytes(encoded) == bytes);
        REQUIRE(read_bytes(source->cast<encoded_data_representation>()) == bytes);
        if (target_space == &host) {
          REQUIRE(encoded.get_size_in_bytes() == ((bytes.size() + 4095) / 4096) * 4096);
        } else if (target_space == &gpu) {
          REQUIRE(encoded.get_size_in_bytes() == bytes.size());
          REQUIRE(encoded.get_writer_event() != nullptr);
        } else {
          REQUIRE(encoded.get_size_in_bytes() == bytes.size());
          REQUIRE(encoded.get_file_offset() == 0);
          REQUIRE(encoded.get_file_path().parent_path() == directory.path);
          REQUIRE(std::filesystem::file_size(encoded.get_file_path()) == bytes.size());
        }
      }
    }
  }
  // Every DISK copy removed its file and released its reservation.
  REQUIRE(disk->get_total_reserved_memory() == 0);
  REQUIRE(std::filesystem::is_empty(directory.path));
}

TEST_CASE("encoded clones own independent storage", "[encoded][relocation][ownership]")
{
  temp_directory directory;
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  auto disk = make_disk(directory.path);
  rmm::cuda_stream stream;
  const auto bytes = payload(4097);
  auto seed        = encoded_data_representation::from_host(bytes, make_metadata(), host);
  for (auto* space : std::array{&host, &gpu, disk.get()}) {
    auto original = seed->relocate(space, stream);
    auto clone    = original->clone(stream);
    auto& source  = original->cast<encoded_data_representation>();
    auto& copy    = clone->cast<encoded_data_representation>();
    REQUIRE(&copy.get_memory_space() == space);
    if (space == disk.get()) {
      REQUIRE(source.get_file_path() != copy.get_file_path());
      REQUIRE(std::filesystem::exists(copy.get_file_path()));
    } else if (space == &gpu) {
      REQUIRE(source.get_device_buffer().data() != copy.get_device_buffer().data());
    } else {
      REQUIRE(source.get_host_buffer()->at(0).data() != copy.get_host_buffer()->at(0).data());
    }
    original.reset();
    REQUIRE(read_bytes(copy) == bytes);
    const auto path = space == disk.get() ? copy.get_file_path() : std::filesystem::path{};
    clone.reset();
    if (!path.empty()) { REQUIRE_FALSE(std::filesystem::exists(path)); }
  }
}

TEST_CASE("encoded relocation reservation bytes match the destination footprint",
          "[encoded][relocation][reservation]")
{
  temp_directory directory;
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  memory::memory_space disk{
    memory::disk_memory_space_config{0, 1ull << 24, directory.path.string()}};
  using encoded = encoded_data_representation;
  REQUIRE(encoded::relocation_reservation_bytes(host, 0) == 0);
  REQUIRE(encoded::relocation_reservation_bytes(host, 1) == block_size);
  REQUIRE(encoded::relocation_reservation_bytes(host, block_size) == block_size);
  REQUIRE(encoded::relocation_reservation_bytes(host, block_size + 1) == 2 * block_size);
  REQUIRE(encoded::relocation_reservation_bytes(gpu, block_size + 1) == block_size + 1);
  REQUIRE(encoded::relocation_reservation_bytes(disk, block_size + 1) == block_size + 1);
  REQUIRE_THROWS_AS(
    encoded::relocation_reservation_bytes(host, std::numeric_limits<std::size_t>::max()),
    std::overflow_error);

  auto config       = host_config(-2);
  config.block_size = 1536;
  memory::memory_space odd_blocks{config};
  REQUIRE_THROWS_AS(encoded::relocation_reservation_bytes(odd_blocks, 1), std::invalid_argument);

  // A reservation of exactly relocation_reservation_bytes() is accepted; one block less is not.
  rmm::cuda_stream stream;
  memory::memory_space target{host_config(-3, 4 * block_size)};
  const auto bytes = payload(2 * block_size + 1);
  auto source      = encoded::from_host(bytes, make_metadata(), host);
  const auto need  = encoded::relocation_reservation_bytes(target, bytes.size());
  {
    auto short_reservation = target.make_reservation(need - block_size);
    REQUIRE_THROWS_AS(source->relocate(&target, stream, short_reservation.get()),
                      std::invalid_argument);
  }
  auto reservation = target.make_reservation(need);
  auto result      = source->relocate(&target, stream, reservation.get());
  REQUIRE(result->get_size_in_bytes() == need);
  REQUIRE(read_bytes(result->cast<encoded>()) == bytes);

  const auto disk_need = encoded::relocation_reservation_bytes(disk, bytes.size());
  {
    auto short_reservation = disk.make_reservation(disk_need - 1);
    REQUIRE_THROWS_AS(source->relocate(&disk, stream, short_reservation.get()),
                      std::invalid_argument);
  }
  auto disk_reservation = disk.make_reservation(disk_need);
  auto on_disk          = source->relocate(&disk, stream, disk_reservation.get());
  REQUIRE(read_bytes(on_disk->cast<encoded>()) == bytes);
}

TEST_CASE("encoded DISK legs serve aligned and unaligned ranges through each io context",
          "[encoded][relocation][disk][backend]")
{
  const auto backend = GENERATE(disk_io_backend::uring, disk_io_backend::cufile);
  CAPTURE(test::to_string(backend));
  auto context = test::try_make_disk_io_context(backend);
  if (context == nullptr) {
    SKIP("the " << test::to_string(backend) << " backend cannot run here");
  }
  temp_directory directory;
  auto disk = make_disk(directory.path);
  disk->set_io_context(context);
  memory::memory_space gpu{gpu_config()};
  memory::memory_space host{host_config(-1, 4ull << 20)};
  rmm::cuda_stream stream;
  // Whole I/O blocks and partial tails, at aligned and unaligned file offsets of imported ranges.
  for (const std::size_t size :
       {std::size_t{4096}, std::size_t{4097}, (std::size_t{1} << 20) + 4095}) {
    for (const std::size_t offset : {std::size_t{0}, std::size_t{37}, std::size_t{4096}}) {
      INFO("size=" << size << " offset=" << offset);
      const auto bytes = payload(size);
      const auto path  = directory.path / "imported.bin";
      write_file(path, payload(offset));
      write_file(path, bytes, offset);
      auto imported =
        encoded_data_representation::from_disk(path, offset, bytes.size(), make_metadata(), *disk);
      auto device = imported->relocate(&gpu, stream);
      REQUIRE(read_bytes(device->cast<encoded_data_representation>()) == bytes);
      auto written = device->relocate(disk.get(), stream);
      REQUIRE(read_bytes(written->cast<encoded_data_representation>()) == bytes);
      REQUIRE(std::filesystem::file_size(
                written->cast<encoded_data_representation>().get_file_path()) == bytes.size());
      auto copied = imported->relocate(disk.get(), stream);
      REQUIRE(read_bytes(copied->cast<encoded_data_representation>()) == bytes);
      auto restored = copied->relocate(&host, stream);
      REQUIRE(read_bytes(restored->cast<encoded_data_representation>()) == bytes);
      auto stored = restored->relocate(disk.get(), stream);
      REQUIRE(read_bytes(stored->cast<encoded_data_representation>()) == bytes);
      REQUIRE(disk->get_io_context() == context);
      imported.reset();
      std::filesystem::remove(path);
    }
  }
}

TEST_CASE("encoded GPU to DISK relocation stages through host memory without device writes",
          "[encoded][relocation][disk][gpu]")
{
  temp_directory directory;
  auto context = std::make_shared<scripted_ioctx>(make_disk_io_context());
  auto disk    = make_disk(directory.path, 64ull << 20);
  disk->set_io_context(context);
  auto config      = host_config(-1, 32ull << 20);
  config.pool_size = 256;
  memory::memory_space host{config};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream stream;
  // More than one staging chunk, and not a multiple of it.
  const auto bytes = payload((8ull << 20) + 4097);
  auto device =
    encoded_data_representation::from_host(bytes, make_metadata(), host)->relocate(&gpu, stream);

  context->device_write = false;
  auto staged           = device->relocate(disk.get(), stream);
  REQUIRE(context->device_write_slices == 0);
  REQUIRE(read_bytes(staged->cast<encoded_data_representation>()) == bytes);

  context->device_write = true;
  auto direct           = device->relocate(disk.get(), stream);
  REQUIRE(context->device_write_slices > 0);
  REQUIRE(read_bytes(direct->cast<encoded_data_representation>()) == bytes);
}

TEST_CASE("encoded multi-chunk DISK copies open their target file once",
          "[encoded][relocation][disk]")
{
  temp_directory directory;
  auto context = std::make_shared<scripted_ioctx>(make_disk_io_context());
  auto disk    = make_disk(directory.path, 64ull << 20);
  disk->set_io_context(context);
  auto config      = host_config(-1, 32ull << 20);
  config.pool_size = 256;
  memory::memory_space host{config};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream stream;
  // Three staging chunks, the last one partial.
  const auto bytes      = payload((16ull << 20) + 4097);
  auto host_source      = encoded_data_representation::from_host(bytes, make_metadata(), host);
  auto disk_source      = host_source->relocate(disk.get(), stream);
  auto device           = host_source->relocate(&gpu, stream);
  context->device_write = false;
  for (idata_representation* source :
       std::array<idata_representation*, 2>{disk_source.get(), device.get()}) {
    INFO("source tier=" << static_cast<int>(source->get_current_tier()));
    const auto opens_before = context->write_opens.load();
    auto copy               = source->relocate(disk.get(), stream);
    REQUIRE(context->write_opens == opens_before + 1);
    REQUIRE(read_bytes(copy->cast<encoded_data_representation>()) == bytes);
  }
}

TEST_CASE("encoded relocation rejects an io context without the leg's capability up front",
          "[encoded][relocation][disk][failure]")
{
  temp_directory directory;
  auto context = std::make_shared<scripted_ioctx>(make_disk_io_context());
  auto disk    = make_disk(directory.path);
  disk->set_io_context(context);
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream stream;
  const auto bytes       = payload(8193);
  auto host_source       = encoded_data_representation::from_host(bytes, make_metadata(), host);
  auto gpu_source        = host_source->relocate(&gpu, stream);
  auto disk_source       = host_source->relocate(disk.get(), stream);
  auto* host_allocator   = host.get_memory_resource_of<memory::Tier::HOST>();
  auto* gpu_allocator    = gpu.get_memory_resource_of<memory::Tier::GPU>();
  const auto host_before = host_allocator->get_total_allocated_bytes();
  const auto gpu_before  = gpu_allocator->get_total_allocated_bytes();
  const auto opens       = context->write_opens.load();

  // No DISK copy is created (or reserved) for a context that cannot write, from any source.
  context->write = false;
  for (idata_representation* source : std::array<idata_representation*, 3>{
         host_source.get(), gpu_source.get(), disk_source.get()}) {
    INFO("source tier=" << static_cast<int>(source->get_current_tier()));
    REQUIRE_THROWS_AS(source->relocate(disk.get(), stream), std::logic_error);
    REQUIRE(file_count(directory.path) == 1);
    REQUIRE(disk->get_total_reserved_memory() == bytes.size());
  }
  REQUIRE(context->write_opens == opens);
  context->write = true;

  // No GPU buffer is allocated for a context that cannot read into device memory; reads into
  // host memory are unaffected.
  context->device_read = false;
  REQUIRE_THROWS_AS(disk_source->relocate(&gpu, stream), std::logic_error);
  REQUIRE(gpu_allocator->get_total_allocated_bytes() == gpu_before);
  auto restored = disk_source->relocate(&host, stream);
  REQUIRE(read_bytes(restored->cast<encoded_data_representation>()) == bytes);
  restored.reset();
  REQUIRE(host_allocator->get_total_allocated_bytes() == host_before);
}

TEST_CASE("encoded relocation removes a failed DISK copy and keeps its source",
          "[encoded][relocation][disk][failure]")
{
  temp_directory directory;
  auto context = std::make_shared<scripted_ioctx>(make_disk_io_context());
  auto disk    = make_disk(directory.path);
  disk->set_io_context(context);
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream stream;
  const auto bytes       = payload(8193);
  auto host_source       = encoded_data_representation::from_host(bytes, make_metadata(), host);
  auto gpu_source        = host_source->relocate(&gpu, stream);
  auto disk_source       = host_source->relocate(disk.get(), stream);
  const auto disk_path   = disk_source->cast<encoded_data_representation>().get_file_path();
  auto* host_allocator   = host.get_memory_resource_of<memory::Tier::HOST>();
  auto* gpu_allocator    = gpu.get_memory_resource_of<memory::Tier::GPU>();
  const auto host_before = host_allocator->get_total_allocated_bytes();
  const auto gpu_before  = gpu_allocator->get_total_allocated_bytes();

  // Writes: the bytes land, then the request fails; the new file must be gone afterwards.
  context->fail_write = true;
  for (const bool device_write : {true, false}) {
    context->device_write = device_write;
    for (idata_representation* source : std::array<idata_representation*, 3>{
           host_source.get(), gpu_source.get(), disk_source.get()}) {
      INFO("source tier=" << static_cast<int>(source->get_current_tier())
                          << " device_write=" << device_write);
      REQUIRE_THROWS_WITH(source->relocate(disk.get(), stream), "injected write failure");
      REQUIRE(read_bytes(source->cast<encoded_data_representation>()) == bytes);
      REQUIRE(file_count(directory.path) == 1);
      REQUIRE(disk->get_total_reserved_memory() == bytes.size());
    }
  }
  context->fail_write   = false;
  context->device_write = true;

  // Reads: no destination allocation survives the failure.
  context->fail_read = true;
  REQUIRE_THROWS_WITH(disk_source->relocate(&host, stream), "injected read failure");
  REQUIRE_THROWS_WITH(disk_source->relocate(&gpu, stream), "injected read failure");
  REQUIRE_THROWS_WITH(disk_source->clone(stream), "injected read failure");
  REQUIRE(host_allocator->get_total_allocated_bytes() == host_before);
  REQUIRE(gpu_allocator->get_total_allocated_bytes() == gpu_before);
  REQUIRE(file_count(directory.path) == 1);
  REQUIRE(disk->get_total_reserved_memory() == bytes.size());
  context->fail_read = false;

  REQUIRE(read_bytes(disk_source->cast<encoded_data_representation>()) == bytes);
  disk_source.reset();
  REQUIRE_FALSE(std::filesystem::exists(disk_path));
  REQUIRE(disk->get_total_reserved_memory() == 0);
}

TEST_CASE("encoded relocation from a truncated DISK range fails without a destination",
          "[encoded][relocation][disk][failure]")
{
  temp_directory directory;
  auto disk = make_disk(directory.path);
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream stream;
  const auto bytes = payload(8193);
  const auto path  = directory.path / "input.bin";
  write_file(path, bytes);
  auto source =
    encoded_data_representation::from_disk(path, 0, bytes.size(), make_metadata(), *disk);
  std::filesystem::resize_file(path, 3);
  auto* host_allocator   = host.get_memory_resource_of<memory::Tier::HOST>();
  auto* gpu_allocator    = gpu.get_memory_resource_of<memory::Tier::GPU>();
  const auto host_before = host_allocator->get_total_allocated_bytes();
  const auto gpu_before  = gpu_allocator->get_total_allocated_bytes();
  for (auto* target : std::array<memory::memory_space*, 3>{&host, &gpu, disk.get()}) {
    INFO("target tier=" << static_cast<int>(target->get_tier()));
    REQUIRE_THROWS(source->relocate(target, stream));
  }
  REQUIRE(host_allocator->get_total_allocated_bytes() == host_before);
  REQUIRE(gpu_allocator->get_total_allocated_bytes() == gpu_before);
  REQUIRE(disk->get_total_reserved_memory() == bytes.size());
  REQUIRE(file_count(directory.path) == 1);

  write_file(path, bytes);
  auto restored = source->relocate(&host, stream);
  REQUIRE(read_bytes(restored->cast<encoded_data_representation>()) == bytes);
}

TEST_CASE("encoded disk ranges preserve borrowed files and remove owned files",
          "[encoded][disk][ownership]")
{
  temp_directory directory;
  memory::memory_space disk{
    memory::disk_memory_space_config{0, 1ull << 24, directory.path.string()}};
  const auto bytes  = payload(4097);
  const auto path   = directory.path / "input.bin";
  const auto prefix = payload(37);
  write_file(path, prefix);
  write_file(path, bytes, prefix.size());
  {
    auto source = encoded_data_representation::from_disk(
      path, prefix.size(), bytes.size(), make_metadata(), disk);
    REQUIRE(source->get_file_offset() == prefix.size());
    REQUIRE(source->get_payload_size() == bytes.size());
    REQUIRE(source->get_size_in_bytes() == bytes.size());
    REQUIRE(source->get_metadata() == make_metadata());
  }
  REQUIRE(std::filesystem::exists(path));
  {
    auto owner = encoded_data_representation::from_disk(
      path, prefix.size(), bytes.size(), make_metadata(), disk, encoded_file_ownership::owned);
    REQUIRE(std::filesystem::exists(path));
  }
  REQUIRE_FALSE(std::filesystem::exists(path));
}

TEST_CASE("encoded factories validate ranges tiers and metadata", "[encoded][validation]")
{
  temp_directory directory;
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  memory::memory_space disk{
    memory::disk_memory_space_config{0, 1ull << 24, directory.path.string()}};
  rmm::cuda_stream stream;
  const auto bytes = payload(17);
  auto metadata    = make_metadata();
  const auto path  = directory.path / "input.bin";
  write_file(path, bytes);
  REQUIRE_THROWS_AS(encoded_data_representation::from_host(bytes, metadata, disk),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(encoded_data_representation::from_device(nullptr, 1, metadata, gpu, stream),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
    encoded_data_representation::from_device(bytes.data(), bytes.size(), metadata, gpu, stream),
    std::invalid_argument);
  REQUIRE_THROWS(encoded_data_representation::from_disk(path, 16, 2, metadata, disk));
  REQUIRE_THROWS(encoded_data_representation::from_disk(
    path, std::numeric_limits<std::size_t>::max(), 2, metadata, disk));
  REQUIRE_THROWS(encoded_data_representation::from_disk(directory.path, 0, 0, metadata, disk));
  REQUIRE_THROWS(encoded_data_representation::from_disk(path, 0, 1, metadata, host));
  REQUIRE_THROWS_AS(encoded_data_representation::from_disk(
                      path, 0, 1, metadata, disk, static_cast<encoded_file_ownership>(7)),
                    std::invalid_argument);
  metadata.encoding.clear();
  REQUIRE_THROWS_AS(encoded_data_representation::from_host(bytes, metadata, host),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(encoded_data_representation::from_disk(path, 0, 1, metadata, disk),
                    std::invalid_argument);
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(disk.get_total_reserved_memory() == 0);
}

TEST_CASE("encoded relocation leaves no destination allocation behind on failure",
          "[encoded][relocation][failure]")
{
  memory::memory_space host{host_config()};
  memory::memory_space too_small{host_config(-2, block_size)};
  rmm::cuda_stream stream;
  const auto bytes = payload(8193);
  auto source      = encoded_data_representation::from_host(bytes, make_metadata(), host);
  REQUIRE_THROWS(source->relocate(&too_small, stream));
  REQUIRE(too_small.get_memory_resource_of<memory::Tier::HOST>()->get_total_allocated_bytes() == 0);
  REQUIRE(too_small.get_total_reserved_memory() == 0);
  REQUIRE(read_bytes(*source) == bytes);
  REQUIRE_THROWS_AS(source->relocate(nullptr, stream), std::invalid_argument);
  REQUIRE(read_bytes(*source) == bytes);
}

TEST_CASE("encoded host reservation is consumed once and may outlive its handle",
          "[encoded][relocation][reservation]")
{
  memory::memory_space source_space{host_config()};
  memory::memory_space target_space{host_config(-2, 4 * block_size)};
  rmm::cuda_stream stream;
  const auto bytes = payload(8193);
  auto source      = encoded_data_representation::from_host(bytes, make_metadata(), source_space);
  auto reservation = target_space.make_reservation(3 * block_size);
  auto result      = source->relocate(&target_space, stream, reservation.get());
  REQUIRE(target_space.get_total_reserved_memory() == 3 * block_size);
  REQUIRE(read_bytes(result->cast<encoded_data_representation>()) == bytes);
  reservation.reset();
  REQUIRE(read_bytes(result->cast<encoded_data_representation>()) == bytes);
  REQUIRE(target_space.get_memory_resource_of<memory::Tier::HOST>()->get_total_allocated_bytes() ==
          3 * block_size);
  result.reset();
  REQUIRE(target_space.get_total_reserved_memory() == 0);
  REQUIRE(target_space.get_memory_resource_of<memory::Tier::HOST>()->get_total_allocated_bytes() ==
          0);

  auto wrong = source_space.make_reservation(3 * block_size);
  REQUIRE_THROWS_AS(source->relocate(&target_space, stream, wrong.get()), std::invalid_argument);
  wrong.reset();
  auto insufficient = target_space.make_reservation(block_size);
  REQUIRE_THROWS_AS(source->relocate(&target_space, stream, insufficient.get()),
                    std::invalid_argument);
  REQUIRE(target_space.get_total_reserved_memory() == block_size);
}

TEST_CASE("encoded GPU writer events order relocation on another stream", "[encoded][gpu][stream]")
{
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream writer;
  rmm::cuda_stream reader;
  const auto bytes = payload(65537);
  rmm::device_buffer buffer{bytes.size(), writer};
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
    buffer.data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice, writer.value()));
  auto source = encoded_data_representation::from_device(
    buffer.data(), buffer.size(), make_metadata(), gpu, writer);
  REQUIRE(read_bytes(*source) == bytes);
  auto* data = const_cast<void*>(source->get_device_buffer().data());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(data, 0xA7, bytes.size(), writer.value()));
  source->record_writer_event(writer);
  REQUIRE(source->get_writer_event() != nullptr);
  source->rebind_stream(reader);
  auto result = source->relocate(&host, reader);
  REQUIRE(read_bytes(result->cast<encoded_data_representation>()) ==
          std::vector<std::byte>(bytes.size(), std::byte{0xA7}));
}

namespace {

/// Hold back later work on a stream long enough that an unordered reader would observe stale bytes.
void delay_stream(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

}  // namespace

TEST_CASE("encoded relocation waits for a pending GPU writer for every target",
          "[encoded][gpu][stream]")
{
  int count = 0;
  CUCASCADE_CUDA_TRY(cudaGetDeviceCount(&count));
  temp_directory directory;
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  auto disk = make_disk(directory.path);
  std::unique_ptr<memory::memory_space> other_gpu;
  if (count >= 2) {
    memory::gpu_memory_space_config config;
    config.device_id       = 1;
    config.memory_capacity = 32ull << 20;
    other_gpu              = std::make_unique<memory::memory_space>(config);
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  rmm::cuda_stream writer;
  rmm::cuda_stream reader;
  const auto bytes = payload(65537);
  const std::vector<std::byte> written(bytes.size(), std::byte{0xC3});
  auto seed = encoded_data_representation::from_host(bytes, make_metadata(), host);

  std::vector<memory::memory_space*> targets{&gpu, &host, disk.get()};
  if (other_gpu) { targets.push_back(other_gpu.get()); }
  for (auto* target : targets) {
    INFO("target tier=" << static_cast<int>(target->get_tier())
                        << " device=" << target->get_device_id());
    auto source = seed->relocate(&gpu, writer);
    REQUIRE(read_bytes(source->cast<encoded_data_representation>()) == bytes);
    // The overwrite is still queued behind the delay when relocation starts on another stream.
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer.value(), delay_stream, nullptr));
    auto* data =
      const_cast<void*>(source->cast<encoded_data_representation>().get_device_buffer().data());
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(data, 0xC3, bytes.size(), writer.value()));
    source->record_writer_event(writer);
    source->rebind_stream(reader);

    auto check = [&](::cuda::stream_ref stream) {
      auto result = source->relocate(target, stream);
      REQUIRE(&result->get_memory_space() == target);
      REQUIRE(read_bytes(result->cast<encoded_data_representation>()) == written);
    };
    if (target == other_gpu.get()) {
      // A stream on the destination device, destroyed only after the result is released.
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{1}};
      rmm::cuda_stream target_stream;
      check(target_stream);
    } else {
      check(reader);
    }
  }
}

TEST_CASE("encoded DISK to GPU relocation is ordered after earlier work on its stream",
          "[encoded][relocation][disk][gpu][stream]")
{
  const auto backend = GENERATE(disk_io_backend::uring, disk_io_backend::cufile);
  CAPTURE(test::to_string(backend));
  auto context = test::try_make_disk_io_context(backend);
  if (context == nullptr) {
    SKIP("the " << test::to_string(backend) << " backend cannot run here");
  }
  temp_directory directory;
  auto disk = make_disk(directory.path);
  disk->set_io_context(context);
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{gpu.get_device_id()}};
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
  const auto bytes = payload(65537);
  const auto path  = directory.path / "input.bin";
  write_file(path, bytes);
  auto source =
    encoded_data_representation::from_disk(path, 0, bytes.size(), make_metadata(), *disk);

  // Leave a delayed overwrite of a payload-sized block queued on the stream and free the block in
  // stream order, so the destination allocated on the same stream may reuse it. A read that is not
  // ordered after the queued work would be overwritten once the delay ends.
  {
    rmm::device_buffer stale{bytes.size(), stream, gpu.get_default_allocator()};
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(stream.value(), delay_stream, nullptr));
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(stale.data(), 0x5A, bytes.size(), stream.value()));
  }
  auto result = source->relocate(&gpu, stream);
  auto& copy  = result->cast<encoded_data_representation>();
  REQUIRE(copy.get_writer_event() != nullptr);

  // Observe the copy only through its writer event, from a stream that does not synchronize with
  // the relocation stream.
  std::vector<std::byte> observed(bytes.size());
  CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(reader.value(), copy.get_writer_event(), 0));
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(observed.data(),
                                     copy.get_device_buffer().data(),
                                     observed.size(),
                                     cudaMemcpyDeviceToHost,
                                     reader.value()));
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(reader.value()));
  REQUIRE(observed == bytes);
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.value()));
}

TEST_CASE("encoded GPU relocation uses a reservation already attached to its stream",
          "[encoded][relocation][gpu][reservation]")
{
  memory::memory_space host{host_config()};
  memory::memory_space gpu{gpu_config()};
  rmm::cuda_stream stream;
  const auto bytes = payload(65536);
  auto source      = encoded_data_representation::from_host(bytes, make_metadata(), host);
  auto reservation = gpu.make_reservation(bytes.size());
  auto* allocator  = gpu.get_memory_resource_of<memory::Tier::GPU>();
  REQUIRE_THROWS_AS(source->relocate(&gpu, stream, reservation.get()), std::invalid_argument);
  REQUIRE(allocator->attach_reservation_to_tracker(
    stream, std::move(reservation), std::make_unique<memory::fail_reservation_limit_policy>()));
  auto device = source->relocate(&gpu, stream);
  REQUIRE(gpu.get_total_reserved_memory() == bytes.size());
  REQUIRE(allocator->get_total_allocated_bytes() == bytes.size());
  REQUIRE(read_bytes(device->cast<encoded_data_representation>()) == bytes);
  device.reset();
  stream.synchronize();
  allocator->reset_stream_reservation(stream);
  REQUIRE(gpu.get_total_reserved_memory() == 0);
}

TEST_CASE("encoded disk imports reserve their payload bytes", "[encoded][disk][reservation]")
{
  temp_directory directory;
  const auto bytes = payload(4097);
  memory::memory_space disk{
    memory::disk_memory_space_config{0, bytes.size() * 2, directory.path.string()}};
  memory::memory_space too_small{
    memory::disk_memory_space_config{1, bytes.size() - 1, directory.path.string()}};
  const auto path = directory.path / "imported.bin";
  write_file(path, bytes);
  REQUIRE_THROWS(encoded_data_representation::from_disk(
    path, 0, bytes.size(), make_metadata(), too_small, encoded_file_ownership::owned));
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(too_small.get_total_reserved_memory() == 0);
  auto imported =
    encoded_data_representation::from_disk(path, 0, bytes.size(), make_metadata(), disk);
  REQUIRE(disk.get_total_reserved_memory() == bytes.size());
  auto second =
    encoded_data_representation::from_disk(path, 0, bytes.size(), make_metadata(), disk);
  REQUIRE(disk.get_total_reserved_memory() == bytes.size() * 2);
  REQUIRE_THROWS(
    encoded_data_representation::from_disk(path, 0, bytes.size(), make_metadata(), disk));
  REQUIRE(disk.get_total_reserved_memory() == bytes.size() * 2);
  imported.reset();
  REQUIRE(disk.get_total_reserved_memory() == bytes.size());
  REQUIRE(std::filesystem::exists(path));
  second.reset();
  REQUIRE(disk.get_total_reserved_memory() == 0);
}

TEST_CASE("encoded DISK copies reserve their payload or use the caller's reservation",
          "[encoded][relocation][disk][reservation]")
{
  temp_directory directory;
  const auto bytes = payload(4097);
  auto disk        = make_disk(directory.path, bytes.size() * 2);
  memory::memory_space host{host_config()};
  rmm::cuda_stream stream;
  const auto path = directory.path / "imported.bin";
  write_file(path, bytes);
  auto imported =
    encoded_data_representation::from_disk(path, 0, bytes.size(), make_metadata(), *disk);
  auto clone = imported->clone(stream);
  REQUIRE(disk->get_total_reserved_memory() == bytes.size() * 2);
  // The space is full: the copy fails before creating a file.
  REQUIRE_THROWS_AS(imported->clone(stream), rmm::out_of_memory);
  REQUIRE(file_count(directory.path) == 2);
  REQUIRE(read_bytes(*imported) == bytes);
  imported.reset();
  REQUIRE(disk->get_total_reserved_memory() == bytes.size());
  REQUIRE(std::filesystem::exists(path));
  clone.reset();
  REQUIRE(disk->get_total_reserved_memory() == 0);
  REQUIRE(file_count(directory.path) == 1);

  // A caller-owned reservation backs the copy and outlives its file.
  auto source      = encoded_data_representation::from_host(bytes, make_metadata(), host);
  auto reservation = disk->make_reservation(bytes.size());
  auto relocated   = source->relocate(disk.get(), stream, reservation.get());
  REQUIRE(disk->get_total_reserved_memory() == bytes.size());
  REQUIRE(read_bytes(relocated->cast<encoded_data_representation>()) == bytes);
  const auto relocated_path = relocated->cast<encoded_data_representation>().get_file_path();
  relocated.reset();
  REQUIRE_FALSE(std::filesystem::exists(relocated_path));
  REQUIRE(disk->get_total_reserved_memory() == bytes.size());
  reservation.reset();
  REQUIRE(disk->get_total_reserved_memory() == 0);

  auto other_disk = make_disk(directory.path, bytes.size(), 1);
  auto foreign    = other_disk->make_reservation(bytes.size());
  REQUIRE_THROWS_AS(source->relocate(disk.get(), stream, foreign.get()), std::invalid_argument);
  REQUIRE(file_count(directory.path) == 1);
}

TEST_CASE("encoded GPU relocation supports another device", "[encoded][gpu][multigpu]")
{
  int count = 0;
  CUCASCADE_CUDA_TRY(cudaGetDeviceCount(&count));
  if (count < 2) { SKIP("Two CUDA devices are required"); }
  memory::memory_space host{host_config(-1, 8ull << 20)};
  memory::memory_space gpu0{gpu_config(0)};
  memory::gpu_memory_space_config other_config;
  other_config.device_id       = 1;
  other_config.memory_capacity = 32ull << 20;
  memory::memory_space gpu1{other_config};
  temp_directory directory;
  auto disk = make_disk(directory.path);
  // Larger than one staging chunk, and not a multiple of it.
  const auto bytes = payload((3ull << 20) + 65537);
  auto source      = encoded_data_representation::from_host(bytes, make_metadata(), host);
  rmm::cuda_set_device_raii first_device{rmm::cuda_device_id{0}};
  rmm::cuda_stream first_stream;
  auto first = source->relocate(&gpu0, first_stream);
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{1}};
    rmm::cuda_stream stream;
    REQUIRE_THROWS(source->relocate(&gpu1, first_stream));
    REQUIRE_THROWS_AS(encoded_data_representation::from_device(
                        first->cast<encoded_data_representation>().get_device_buffer().data(),
                        bytes.size(),
                        make_metadata(),
                        gpu1,
                        stream),
                      std::invalid_argument);
    auto second = first->relocate(&gpu1, stream);
    REQUIRE(read_bytes(second->cast<encoded_data_representation>()) == bytes);
    REQUIRE(second->get_device_id() == 1);
    auto back = second->relocate(&gpu0, first_stream);
    REQUIRE(read_bytes(back->cast<encoded_data_representation>()) == bytes);
    REQUIRE(back->get_device_id() == 0);
    // DISK legs use a stream of the GPU taking part.
    REQUIRE_THROWS(second->relocate(disk.get(), first_stream));
    auto on_disk = second->relocate(disk.get(), stream);
    REQUIRE(read_bytes(on_disk->cast<encoded_data_representation>()) == bytes);
    REQUIRE_THROWS(on_disk->relocate(&gpu1, first_stream));
    auto from_disk = on_disk->relocate(&gpu1, stream);
    REQUIRE(read_bytes(from_disk->cast<encoded_data_representation>()) == bytes);
    REQUIRE(from_disk->get_device_id() == 1);
  }
  first.reset();
}

TEST_CASE("encoded relocation spans staging chunks and different host block sizes",
          "[encoded][relocation][boundary]")
{
  temp_directory directory;
  auto source_config       = host_config(-1, 32ull << 20);
  auto target_config       = host_config(-2, 32ull << 20);
  source_config.block_size = 2048;
  target_config.block_size = 65536;
  source_config.pool_size  = 128;
  memory::memory_space source_host{source_config};
  memory::memory_space target_host{target_config};
  memory::memory_space gpu{gpu_config()};
  auto disk = make_disk(directory.path, 64ull << 20);
  rmm::cuda_stream stream;
  const auto bytes = payload((8ull << 20) + 33);
  auto original    = encoded_data_representation::from_host(bytes, make_metadata(), source_host);
  auto host_copy   = original->relocate(&target_host, stream);
  REQUIRE(read_bytes(host_copy->cast<encoded_data_representation>()) == bytes);
  auto device_copy = host_copy->relocate(&gpu, stream);
  REQUIRE(read_bytes(device_copy->cast<encoded_data_representation>()) == bytes);
  auto host_again = device_copy->relocate(&source_host, stream);
  REQUIRE(read_bytes(host_again->cast<encoded_data_representation>()) == bytes);
  auto host_back = host_again->relocate(&target_host, stream);
  REQUIRE(read_bytes(host_back->cast<encoded_data_representation>()) == bytes);
  // DISK legs, including a file-to-file copy larger than one staging chunk.
  auto disk_copy = host_copy->relocate(disk.get(), stream);
  REQUIRE(read_bytes(disk_copy->cast<encoded_data_representation>()) == bytes);
  auto disk_clone = disk_copy->clone(stream);
  REQUIRE(read_bytes(disk_clone->cast<encoded_data_representation>()) == bytes);
  auto device_again = disk_clone->relocate(&gpu, stream);
  REQUIRE(read_bytes(device_again->cast<encoded_data_representation>()) == bytes);
  auto disk_again = device_again->relocate(disk.get(), stream);
  auto host_last  = disk_again->relocate(&source_host, stream);
  REQUIRE(read_bytes(host_last->cast<encoded_data_representation>()) == bytes);
}

TEST_CASE("encoded host allocation rejects unsupported block alignment",
          "[encoded][relocation][validation]")
{
  auto config       = host_config();
  config.block_size = 1536;
  memory::memory_space host{config};
  REQUIRE_THROWS_AS(encoded_data_representation::from_host(payload(4097), make_metadata(), host),
                    std::invalid_argument);
  REQUIRE(host.get_total_reserved_memory() == 0);
}

TEST_CASE("encoded host allocations remain safe when reservations are replaced",
          "[encoded][relocation][reservation][regression]")
{
  memory::memory_space host{host_config()};
  const auto bytes = payload(8193);
  for (int iteration = 0; iteration < 32; ++iteration) {
    auto original_reservation = host.make_reservation(3 * block_size);
    auto original             = encoded_data_representation::from_host(
      bytes, make_metadata(), host, original_reservation.get());
    original_reservation.reset();
    auto replacement = host.make_reservation(3 * block_size);
    // The allocator commonly reuses the expired reservation arena address immediately. The old
    // allocation must neither grow through that new arena nor debit its tracking when released.
    REQUIRE_FALSE(original->get_host_buffer()->grow_by(block_size));
    REQUIRE(read_bytes(*original) == bytes);
    REQUIRE(replacement->size() == 3 * block_size);
    original.reset();
    REQUIRE(host.get_total_reserved_memory() == 3 * block_size);
    auto fresh =
      encoded_data_representation::from_host(bytes, make_metadata(), host, replacement.get());
    REQUIRE(host.get_total_reserved_memory() == 3 * block_size);
    REQUIRE(read_bytes(*fresh) == bytes);
    fresh.reset();
    replacement.reset();
    REQUIRE(host.get_total_reserved_memory() == 0);
  }
}
