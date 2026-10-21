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

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/data/encoded_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <benchmark/benchmark.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr std::int64_t MiB = 1024LL * 1024;

/// Directory the DISK space creates its files in: $CUCASCADE_BENCH_DISK_DIR, else /tmp.
std::filesystem::path benchmark_disk_dir()
{
  char const* dir = std::getenv("CUCASCADE_BENCH_DISK_DIR");
  return dir != nullptr && *dir != '\0' ? std::filesystem::path{dir}
                                        : std::filesystem::path{"/tmp"};
}

struct temporary_file {
  std::filesystem::path path;
  temporary_file()                                 = default;
  temporary_file(const temporary_file&)            = delete;
  temporary_file& operator=(const temporary_file&) = delete;
  ~temporary_file()
  {
    std::error_code error;
    if (!path.empty()) { std::filesystem::remove(path, error); }
  }
};

/// Ask the kernel to drop the file's cached pages without changing global cache settings.
void drop_file_cache(const std::filesystem::path& path)
{
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    ::fdatasync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
  }
}

const char* tier_name(Tier tier)
{
  switch (tier) {
    case Tier::HOST: return "HOST";
    case Tier::GPU: return "GPU";
    case Tier::DISK: return "DISK";
    default: return "unknown";
  }
}

/// One slice per host block covering the first @p size bytes, at file offsets from 0.
template <typename Slice, typename Pointer>
std::vector<Slice> block_slices(const fixed_multiple_blocks_allocation& blocks, std::size_t size)
{
  std::vector<Slice> slices;
  std::size_t offset = 0;
  for (std::size_t i = 0; offset < size; ++i) {
    const auto block = blocks->at(i);
    const auto count = std::min(block.size(), size - offset);
    slices.emplace_back(offset, count, reinterpret_cast<Pointer>(block.data()));
    offset += count;
  }
  return slices;
}

/// The raw baseline of one DISK leg: one vectored request through @p context, awaited.
void raw_disk_transfer(io::ioctx& context,
                       const std::filesystem::path& path,
                       Tier source_tier,
                       Tier target_tier,
                       const fixed_multiple_blocks_allocation& blocks,
                       rmm::device_buffer& device,
                       std::size_t size,
                       rmm::cuda_stream& stream)
{
  io::write_open_options options;
  options.mode      = io::write_mode::open_existing;
  auto* device_data = static_cast<std::uint8_t*>(device.data());
  if (target_tier == Tier::DISK) {
    auto object = context.open_io_object_for_write(path.string(), options);
    if (source_tier == Tier::GPU) {
      const io::write_slice slice{0, size, device_data};
      (void)context.device_writev_async(*object, {&slice, 1}, stream).get();
    } else {
      auto slices = block_slices<io::write_slice, const std::uint8_t*>(blocks, size);
      (void)context.host_writev_async(*object, slices).get();
    }
    return;
  }
  auto object = context.open_io_object(path.string());
  if (target_tier == Tier::GPU) {
    const io::slice slice{0, size, device_data};
    (void)context.device_readv_async_io(*object, {&slice, 1}, stream).get();
  } else {
    auto slices = block_slices<io::slice, std::uint8_t*>(blocks, size);
    (void)context.host_readv_async_io(*object, slices).get();
  }
}

// Relocation includes destination allocation (and file creation for DISK targets); raw copies
// reuse preallocated buffers and one file. Payloads are opaque fixture bytes, so this benchmark
// measures no codec work. DISK legs use the process-shared default io context.
void encoded_relocation(benchmark::State& state, Tier source_tier, Tier target_tier, bool raw)
{
  const bool uses_disk = source_tier == Tier::DISK || target_tier == Tier::DISK;
  const auto disk_dir  = benchmark_disk_dir();
  if (uses_disk && !std::filesystem::is_directory(disk_dir)) {
    state.SkipWithMessage("DISK directory is absent: " + disk_dir.string() +
                          " (set CUCASCADE_BENCH_DISK_DIR)");
    return;
  }

  try {
    const auto size = static_cast<std::size_t>(state.range(0));
    rmm::cuda_stream stream;
    gpu_memory_space_config gpu_config;
    gpu_config.device_id       = 0;
    gpu_config.memory_capacity = 2ull << 30;
    memory_space gpu(gpu_config);

    host_memory_space_config host_config;
    host_config.memory_capacity      = 2ull << 30;
    host_config.pool_size            = 16;
    host_config.initial_number_pools = 1;
    host_config.make_portable        = true;
    memory_space host(host_config);

    std::unique_ptr<memory_space> disk;
    std::shared_ptr<io::ioctx> context;
    if (uses_disk) {
      disk_memory_space_config disk_config;
      disk_config.disk_id         = 0;
      disk_config.memory_capacity = 4ull << 30;
      disk_config.mount_paths     = disk_dir.string();
      disk                        = std::make_unique<memory_space>(disk_config);
      context = disk->get_or_init_io_context([] { return default_disk_io_context(); });
      if (target_tier == Tier::DISK && !context->supports_write()) {
        state.SkipWithMessage("the default DISK io context cannot write");
        return;
      }
    }
    auto* source_space = source_tier == Tier::HOST  ? &host
                         : source_tier == Tier::GPU ? &gpu
                                                    : disk.get();
    auto* target_space = target_tier == Tier::HOST  ? &host
                         : target_tier == Tier::GPU ? &gpu
                                                    : disk.get();

    std::string label = std::string(tier_name(source_tier)) + "->" + tier_name(target_tier);
    label += raw ? "; raw/preallocated" : "; relocation/allocation-included";
    if (uses_disk) {
      label += "; disk=" + disk_dir.string();
      if (target_tier == Tier::DISK) { label += "; writeback-not-timed"; }
    }
    state.SetLabel(label);

    std::vector<std::byte> payload(size, std::byte{0x5a});
    if (!raw) {
      encoded_data_metadata metadata{"benchmark/opaque", 1, size, {}};
      auto seed   = encoded_data_representation::from_host(payload, metadata, host);
      auto source = seed->relocate(source_space, stream);
      seed.reset();
      // The source stays available for every iteration: each relocation is a copy, as the batch
      // API makes before replacing the old representation.
      for (auto unused : state) {
        (void)unused;
        state.PauseTiming();
        if (source_tier == Tier::DISK) {
          drop_file_cache(source->cast<encoded_data_representation>().get_file_path());
        }
        state.ResumeTiming();
        auto result = source->relocate(target_space, stream);
        benchmark::DoNotOptimize(result.get());
        state.PauseTiming();
        result.reset();
        state.ResumeTiming();
      }
    } else {
      // Match the relocation host block layout and GPU allocator, while keeping allocation and
      // file creation outside the measured raw-copy baseline.
      auto blocks = host.get_memory_resource_of<Tier::HOST>()->allocate_multiple_blocks(size);
      std::size_t offset = 0;
      for (std::size_t i = 0; offset < size; ++i) {
        const auto block = blocks->at(i);
        const auto count = std::min(block.size(), size - offset);
        std::memcpy(block.data(), payload.data() + offset, count);
        offset += count;
      }
      rmm::device_buffer device(size, stream, gpu.get_default_allocator());
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        device.data(), payload.data(), size, cudaMemcpyHostToDevice, stream.value()));
      stream.synchronize();
      temporary_file file;
      if (uses_disk) {
        file.path = generate_disk_file_path(disk_dir.string());
        raw_disk_transfer(
          *context, file.path, Tier::HOST, Tier::DISK, blocks, device, size, stream);
      }

      for (auto unused : state) {
        (void)unused;
        state.PauseTiming();
        if (source_tier == Tier::DISK) { drop_file_cache(file.path); }
        state.ResumeTiming();
        if (uses_disk) {
          raw_disk_transfer(
            *context, file.path, source_tier, target_tier, blocks, device, size, stream);
        } else {
          offset = 0;
          for (std::size_t i = 0; offset < size; ++i) {
            const auto block = blocks->at(i);
            const auto count = std::min(block.size(), size - offset);
            auto* device_ptr = static_cast<std::byte*>(device.data()) + offset;
            if (target_tier == Tier::GPU) {
              CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
                device_ptr, block.data(), count, cudaMemcpyHostToDevice, stream.value()));
            } else {
              CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
                block.data(), device_ptr, count, cudaMemcpyDeviceToHost, stream.value()));
            }
            offset += count;
          }
        }
        stream.synchronize();
      }
    }
    state.SetBytesProcessed(state.iterations() * state.range(0));
    state.counters["encoded_bytes"] = static_cast<double>(size);
    state.counters["logical_bytes"] = static_cast<double>(size);
  } catch (const std::exception& error) {
    state.SkipWithMessage(error.what());
  }
}

#define RELOCATION_BENCHMARK(source, target)                                                     \
  BENCHMARK_CAPTURE(encoded_relocation, source##_to_##target, Tier::source, Tier::target, false) \
    ->Arg(MiB)                                                                                   \
    ->Arg(64 * MiB)                                                                              \
    ->Arg(64 * MiB + 17)                                                                         \
    ->UseRealTime();                                                                             \
  BENCHMARK_CAPTURE(                                                                             \
    encoded_relocation, raw_##source##_to_##target, Tier::source, Tier::target, true)            \
    ->Arg(MiB)                                                                                   \
    ->Arg(64 * MiB)                                                                              \
    ->Arg(64 * MiB + 17)                                                                         \
    ->UseRealTime()

RELOCATION_BENCHMARK(HOST, GPU);
RELOCATION_BENCHMARK(GPU, HOST);
RELOCATION_BENCHMARK(HOST, DISK);
RELOCATION_BENCHMARK(DISK, HOST);
RELOCATION_BENCHMARK(GPU, DISK);
RELOCATION_BENCHMARK(DISK, GPU);

#undef RELOCATION_BENCHMARK

}  // namespace
