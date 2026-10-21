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

#include "utils/test_memory_resources.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/encoded_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

using namespace cucascade;

namespace {

memory::host_memory_space_config make_host_config(int id, std::size_t capacity = 1ull << 20)
{
  memory::host_memory_space_config config;
  config.numa_id              = id;
  config.memory_capacity      = capacity;
  config.block_size           = 4096;
  config.pool_size            = 4;
  config.initial_number_pools = 0;
  config.mr_factory_fn        = [](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
      memory::numa_region_pinned_host_memory_resource{-1, true}};
  };
  return config;
}

memory::gpu_memory_space_config make_gpu_config()
{
  memory::gpu_memory_space_config config;
  config.device_id       = 0;
  config.memory_capacity = 32ull << 20;
  config.mr_factory_fn   = cucascade::test::make_shared_current_device_resource;
  return config;
}

struct temporary_directory {
  temporary_directory()
  {
    path = memory::generate_disk_file_path(std::filesystem::temp_directory_path().string());
    std::filesystem::remove(path);
    std::filesystem::create_directory(path);
  }
  ~temporary_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  temporary_directory(const temporary_directory&)            = delete;
  temporary_directory& operator=(const temporary_directory&) = delete;
  std::filesystem::path path;
};

/// Write @p bytes to a new file at @p path with POSIX pwrite.
void write_file(const std::filesystem::path& path, const std::vector<std::byte>& bytes)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open"); }
  std::size_t done = 0;
  while (done < bytes.size()) {
    const auto written =
      ::pwrite(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
    if (written < 0) {
      const int error = errno;
      ::close(fd);
      throw std::system_error(error, std::generic_category(), "pwrite");
    }
    done += static_cast<std::size_t>(written);
  }
  ::close(fd);
}

std::vector<std::byte> host_bytes(const encoded_data_representation& source)
{
  std::vector<std::byte> result(source.get_payload_size());
  const auto& blocks = source.get_host_buffer();
  std::size_t offset = 0;
  for (std::size_t i = 0; offset < result.size(); ++i) {
    const auto block = blocks->at(i);
    const auto size  = std::min(block.size(), result.size() - offset);
    std::memcpy(result.data() + offset, block.data(), size);
    offset += size;
  }
  return result;
}

std::vector<std::byte> rle_encode(const std::vector<std::byte>& input)
{
  std::vector<std::byte> result;
  for (std::size_t i = 0; i < input.size();) {
    std::size_t count = 1;
    while (i + count < input.size() && count < 255 && input[i + count] == input[i]) {
      ++count;
    }
    result.push_back(static_cast<std::byte>(count));
    result.push_back(input[i]);
    i += count;
  }
  return result;
}

class decoded_representation : public idata_representation {
 public:
  decoded_representation(memory::memory_space& space, std::vector<std::byte> bytes)
    : idata_representation(space), bytes(std::move(bytes))
  {
  }
  std::size_t get_size_in_bytes() const override { return bytes.size(); }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return bytes.size(); }
  std::unique_ptr<idata_representation> clone(::cuda::stream_ref) override
  {
    return std::make_unique<decoded_representation>(get_memory_space(), bytes);
  }
  std::vector<std::byte> bytes;
};

class malformed_relocation : public decoded_representation {
 public:
  malformed_relocation(memory::memory_space& space, bool return_null)
    : decoded_representation(space, {std::byte{99}}), _return_null(return_null)
  {
  }
  std::unique_ptr<idata_representation> relocate(const memory::memory_space*,
                                                 ::cuda::stream_ref stream,
                                                 memory::reservation*) override
  {
    return _return_null ? nullptr : clone(stream);
  }

 private:
  bool _return_null;
};

struct relocation_observer {
  int started{0};
  int completed{0};
  int replaced{0};
  bool success{false};
  const idata_representation* completed_data{nullptr};
};

class relocation_probe : public idata_batch_probe {
 public:
  explicit relocation_probe(relocation_observer& observer) : observer(observer) {}
  void relocation_started(const idata_representation&,
                          const memory::memory_space*) noexcept override
  {
    ++observer.started;
  }
  void relocation_completed(const idata_representation& data, bool success) noexcept override
  {
    ++observer.completed;
    observer.success        = success;
    observer.completed_data = &data;
  }
  void data_replaced(const idata_representation&) noexcept override { ++observer.replaced; }
  relocation_observer& observer;
};

struct stream_gate {
  std::atomic<bool> entered{false};
  std::atomic<bool> released{false};
};

void gate_callback(void* argument)
{
  auto& gate = *static_cast<stream_gate*>(argument);
  gate.entered.store(true, std::memory_order_release);
  gate.entered.notify_all();
  gate.released.wait(false, std::memory_order_acquire);
}

struct release_gate {
  stream_gate& gate;
  ~release_gate()
  {
    gate.released.store(true, std::memory_order_release);
    gate.released.notify_all();
  }
};

}  // namespace

TEST_CASE("data batch relocates compressed bytes and decodes only on explicit conversion",
          "[data_batch][relocation][encoded]")
{
  temporary_directory directory;
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space other_host{make_host_config(-2)};
  memory::memory_space gpu{make_gpu_config()};
  memory::memory_space disk{
    memory::disk_memory_space_config{0, 1ull << 24, directory.path.string()}};
  rmm::cuda_stream stream;
  std::vector<std::byte> plain(8193, std::byte{0xAA});
  std::fill(plain.begin() + 4096, plain.end(), std::byte{0x3D});
  const auto compressed = rle_encode(plain);
  REQUIRE(compressed.size() < plain.size() / 10);
  const encoded_data_metadata metadata{"rle/test", 1, plain.size(), {std::byte{23}, std::byte{0}}};
  auto batch =
    data_batch::make(777, encoded_data_representation::from_host(compressed, metadata, host));
  int decoder_calls = 0;
  representation_converter_registry registry;
  registry.register_converter<encoded_data_representation, decoded_representation>(
    [&](idata_representation& source,
        const memory::memory_space* target,
        ::cuda::stream_ref conversion_stream,
        memory::reservation*) -> std::unique_ptr<idata_representation> {
      ++decoder_calls;
      auto on_host       = source.relocate(target, conversion_stream);
      const auto encoded = host_bytes(on_host->cast<encoded_data_representation>());
      std::vector<std::byte> decoded;
      for (std::size_t i = 0; i < encoded.size(); i += 2) {
        decoded.insert(decoded.end(), std::to_integer<std::size_t>(encoded[i]), encoded.at(i + 1));
      }
      return std::make_unique<decoded_representation>(const_cast<memory::memory_space&>(*target),
                                                      std::move(decoded));
    });

  {
    auto mutable_batch = batch->to_mutable();
    std::filesystem::path disk_path;
    for (auto* target : std::array{&disk, &gpu, &other_host, &gpu, &disk, &host}) {
      mutable_batch.relocate_to(target, stream);
      const auto& encoded = mutable_batch.get_data()->cast<encoded_data_representation>();
      REQUIRE(&encoded.get_memory_space() == target);
      REQUIRE(encoded.get_metadata() == metadata);
      REQUIRE(encoded.get_payload_size() == compressed.size());
      REQUIRE(mutable_batch.get_batch_id() == 777);
      REQUIRE(decoder_calls == 0);
      if (target == &disk) {
        // Leaving DISK removed the previous copy's file.
        if (!disk_path.empty()) { REQUIRE_FALSE(std::filesystem::exists(disk_path)); }
        disk_path = encoded.get_file_path();
        REQUIRE(std::filesystem::file_size(disk_path) == compressed.size());
      } else if (target != &gpu) {
        REQUIRE(host_bytes(encoded) == compressed);
      }
    }
    REQUIRE_FALSE(std::filesystem::exists(disk_path));
    REQUIRE(disk.get_total_reserved_memory() == 0);
    REQUIRE(gpu.get_memory_resource_of<memory::Tier::GPU>()->get_total_allocated_bytes() == 0);
    REQUIRE(other_host.get_memory_resource_of<memory::Tier::HOST>()->get_total_allocated_bytes() ==
            0);
    mutable_batch.convert_to<decoded_representation>(registry, &host, stream);
    REQUIRE(mutable_batch.get_data()->cast<decoded_representation>().bytes == plain);
    REQUIRE(decoder_calls == 1);
  }
}

TEST_CASE("data batch preserves source and reports relocation failures",
          "[data_batch][relocation][failure]")
{
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space too_small{make_host_config(-2, 4096)};
  memory::memory_space large{make_host_config(-3)};
  rmm::cuda_stream stream;
  const std::vector<std::byte> bytes(4097, std::byte{0xB5});
  auto source    = encoded_data_representation::from_host(bytes, {"opaque", 1, 999999, {}}, host);
  auto* original = source.get();
  relocation_observer observer;
  auto batch = data_batch::make(1, std::move(source), std::make_unique<relocation_probe>(observer));
  auto mutable_batch = batch->to_mutable();
  REQUIRE_THROWS(mutable_batch.relocate_to(&too_small, stream));
  REQUIRE(mutable_batch.get_data() == original);
  REQUIRE(host_bytes(original->cast<encoded_data_representation>()) == bytes);
  REQUIRE(too_small.get_total_reserved_memory() == 0);
  REQUIRE(observer.started == 1);
  REQUIRE(observer.completed == 1);
  REQUIRE_FALSE(observer.success);
  REQUIRE(observer.completed_data == original);
  REQUIRE(observer.replaced == 0);
  mutable_batch.relocate_to(&large, stream);
  REQUIRE(&mutable_batch.get_data()->get_memory_space() == &large);
  REQUIRE(host_bytes(mutable_batch.get_data()->cast<encoded_data_representation>()) == bytes);
  REQUIRE(observer.started == 2);
  REQUIRE(observer.completed == 2);
  REQUIRE(observer.success);
  REQUIRE(observer.completed_data == mutable_batch.get_data());
  REQUIRE(observer.replaced == 1);
  REQUIRE(host.get_memory_resource_of<memory::Tier::HOST>()->get_total_allocated_bytes() == 0);
}

TEST_CASE("data batch keeps its DISK source when disk I/O fails",
          "[data_batch][relocation][disk][failure]")
{
  temporary_directory directory;
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space disk{
    memory::disk_memory_space_config{0, 1ull << 24, directory.path.string()}};
  rmm::cuda_stream stream;
  const auto path = directory.path / "compressed.bin";
  const std::vector<std::byte> bytes(4097, std::byte{0xB5});
  write_file(path, bytes);
  auto source =
    encoded_data_representation::from_disk(path, 0, bytes.size(), {"opaque", 1, 999999, {}}, disk);
  auto* original = source.get();
  relocation_observer observer;
  auto batch = data_batch::make(1, std::move(source), std::make_unique<relocation_probe>(observer));
  // A truncated file makes the read short.
  std::filesystem::resize_file(path, 3);
  auto mutable_batch = batch->to_mutable();
  REQUIRE_THROWS(mutable_batch.relocate_to(&host, stream));
  REQUIRE(mutable_batch.get_data() == original);
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(host.get_memory_resource_of<memory::Tier::HOST>()->get_total_allocated_bytes() == 0);
  REQUIRE(observer.started == 1);
  REQUIRE(observer.completed == 1);
  REQUIRE_FALSE(observer.success);
  REQUIRE(observer.completed_data == original);
  REQUIRE(observer.replaced == 0);
  write_file(path, bytes);
  mutable_batch.relocate_to(&host, stream);
  REQUIRE(host_bytes(mutable_batch.get_data()->cast<encoded_data_representation>()) == bytes);
  REQUIRE(observer.started == 2);
  REQUIRE(observer.completed == 2);
  REQUIRE(observer.success);
  REQUIRE(observer.replaced == 1);
  // The borrowed file stays; only the import's reservation is released.
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(disk.get_total_reserved_memory() == 0);
}

TEST_CASE("data batch relocation rejects null and unsupported targets and skips same space",
          "[data_batch][relocation][validation]")
{
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space other{make_host_config(-2)};
  rmm::cuda_stream stream;
  relocation_observer observer;
  auto batch = data_batch::make(
    1,
    std::make_unique<decoded_representation>(host, std::vector<std::byte>{std::byte{42}}),
    std::make_unique<relocation_probe>(observer));
  auto mutable_batch   = batch->to_mutable();
  const auto* original = mutable_batch.get_data();
  REQUIRE_THROWS_AS(mutable_batch.relocate_to(nullptr, stream), std::invalid_argument);
  REQUIRE_NOTHROW(mutable_batch.relocate_to(&host, stream));
  REQUIRE(observer.started == 0);
  REQUIRE_THROWS(mutable_batch.relocate_to(&other, stream));
  REQUIRE(mutable_batch.get_data() == original);
  REQUIRE(observer.started == 1);
  REQUIRE(observer.completed == 1);
  REQUIRE_FALSE(observer.success);
}

TEST_CASE("data batch relocation accepts target reservation without double counting",
          "[data_batch][relocation][reservation]")
{
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space target{make_host_config(-2)};
  rmm::cuda_stream stream;
  const std::vector<std::byte> bytes(4097, std::byte{77});
  auto batch = data_batch::make(
    1, encoded_data_representation::from_host(bytes, {"opaque", 1, 99999, {}}, host));
  auto reservation = target.make_reservation(8192);
  {
    auto mutable_batch = batch->to_mutable();
    mutable_batch.relocate_to(*reservation, stream);
    REQUIRE(&mutable_batch.get_data()->get_memory_space() == &target);
    REQUIRE(target.get_total_reserved_memory() == 8192);
  }
  reservation.reset();
  REQUIRE(host_bytes(batch->to_read_only().get_data()->cast<encoded_data_representation>()) ==
          bytes);
  batch.reset();
  REQUIRE(target.get_total_reserved_memory() == 0);
}

TEST_CASE("data batch rejects a relocation hook returning a null or wrong destination",
          "[data_batch][relocation][failure]")
{
  const auto return_null = GENERATE(false, true);
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space target{make_host_config(-2)};
  rmm::cuda_stream stream;
  relocation_observer observer;
  auto batch         = data_batch::make(1,
                                std::make_unique<malformed_relocation>(host, return_null),
                                std::make_unique<relocation_probe>(observer));
  auto mutable_batch = batch->to_mutable();
  auto* original     = mutable_batch.get_data();
  REQUIRE_THROWS_AS(mutable_batch.relocate_to(&target, stream), std::logic_error);
  REQUIRE(mutable_batch.get_data() == original);
  REQUIRE(original->cast<malformed_relocation>().bytes == std::vector<std::byte>{std::byte{99}});
  REQUIRE(observer.completed == 1);
  REQUIRE_FALSE(observer.success);
  REQUIRE(observer.replaced == 0);
}

TEST_CASE("data batch rejects a GPU reservation handle before changing its source",
          "[data_batch][relocation][gpu][reservation]")
{
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space gpu{make_gpu_config()};
  rmm::cuda_stream stream;
  const std::vector<std::byte> bytes(4096, std::byte{0x9A});
  auto batch = data_batch::make(
    1, encoded_data_representation::from_host(bytes, {"opaque", 1, 8192, {}}, host));
  auto reservation   = gpu.make_reservation(bytes.size());
  auto mutable_batch = batch->to_mutable();
  auto* original     = mutable_batch.get_data();
  REQUIRE_THROWS_AS(mutable_batch.relocate_to(*reservation, stream), std::invalid_argument);
  REQUIRE(mutable_batch.get_data() == original);
  REQUIRE(host_bytes(original->cast<encoded_data_representation>()) == bytes);
  REQUIRE(gpu.get_total_reserved_memory() == bytes.size());
}

TEST_CASE("encoded batch relocation waits for recorded asynchronous GPU readers",
          "[data_batch][relocation][gpu][reader_event]")
{
  memory::memory_space host{make_host_config(-1)};
  memory::memory_space gpu{make_gpu_config()};
  rmm::cuda_stream transfer_stream;
  rmm::cuda_stream reader_stream;
  const std::vector<std::byte> bytes(65537, std::byte{91});
  auto source = encoded_data_representation::from_host(bytes, {"opaque", 1, 999999, {}}, host);
  auto batch  = data_batch::make(1, source->relocate(&gpu, transfer_stream));
  stream_gate gate;
  release_gate release{gate};
  {
    auto reader = batch->to_read_only();
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(reader_stream.value(), gate_callback, &gate));
    reader.record_reader_event(reader_stream);
  }
  gate.entered.wait(false, std::memory_order_acquire);
  REQUIRE_FALSE(batch->try_to_mutable().has_value());
  std::atomic<bool> finished{false};
  std::exception_ptr error;
  std::jthread worker([&] {
    try {
      auto mutable_batch = batch->to_mutable();
      mutable_batch.relocate_to(&host, transfer_stream);
      finished.store(true, std::memory_order_release);
    } catch (...) {
      error = std::current_exception();
    }
  });
  REQUIRE_FALSE(finished.load(std::memory_order_acquire));
  gate.released.store(true, std::memory_order_release);
  gate.released.notify_all();
  worker.join();
  if (error) { std::rethrow_exception(error); }
  REQUIRE(finished.load(std::memory_order_acquire));
  REQUIRE(host_bytes(batch->to_read_only().get_data()->cast<encoded_data_representation>()) ==
          bytes);
}
