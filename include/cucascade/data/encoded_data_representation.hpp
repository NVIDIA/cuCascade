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

#pragma once

#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cucascade {

/** @brief Encoding and logical layout information retained verbatim during relocation. */
struct encoded_data_metadata {
  std::string encoding;              ///< Application-defined, nonempty format or codec name.
  uint32_t version{1};               ///< Application-defined encoding version.
  std::size_t uncompressed_size{0};  ///< Logical size before encoding.
  std::vector<std::byte> metadata;   ///< Opaque schema, codec settings, or chunk descriptors.

  bool operator==(const encoded_data_metadata&) const = default;
};

/** @brief Whether destroying a disk representation removes the entire backing file. */
enum class encoded_file_ownership { borrowed, owned };

/**
 * @brief Encoded bytes whose encoding is independent of their physical memory tier.
 *
 * Factories copy host and device inputs, or adopt an existing disk range. Relocation and cloning
 * copy exactly get_payload_size() bytes and preserve all encoding metadata without interpreting it.
 * They finish all transfers before returning. A decoder can be registered through the ordinary
 * representation converter registry using get_metadata() and the tier-specific storage accessor.
 *
 * The class is built into the cucascade-cudf library, but its interface has no cuDF dependency.
 * Relocation between HOST and GPU spaces, and between GPU spaces, is supported. Relocation to or
 * from a DISK space is not supported yet and throws std::logic_error before touching either side;
 * from_disk() adopts a file range without any I/O.
 *
 * Memory spaces must outlive their representations. For a disk range adopted with owned ownership,
 * destruction removes the entire file, including bytes outside the adopted range. Borrowed files
 * are never removed.
 *
 * Direct use requires external synchronization, as with other data representations. When exposing
 * device bytes to an asynchronous writer, record_writer_event() after enqueueing its writes. A
 * reader must wait on get_writer_event() before accessing the bytes on another stream. Keep the
 * representation and its allocation stream alive until external work finishes, or rebind_stream()
 * after establishing the required ordering.
 */
class encoded_data_representation : public idata_representation {
 public:
  /** @brief Copy a contiguous host payload into blocks belonging to a HOST memory space. */
  [[nodiscard]] static std::unique_ptr<encoded_data_representation> from_host(
    std::span<const std::byte> payload,
    encoded_data_metadata metadata,
    memory::memory_space& space,
    memory::reservation* reservation = nullptr);

  /**
   * @brief Copy device bytes into a GPU memory space on the same CUDA device.
   *
   * The source must be ready for reading on @p stream, which must belong to @p space. The copy is
   * complete on return, including when the supplied stream is the default stream. A non-null
   * reservation is rejected: transfer GPU reservations to the allocator's tracking API before
   * allocation, then call this factory with the default null reservation.
   */
  [[nodiscard]] static std::unique_ptr<encoded_data_representation> from_device(
    const void* payload,
    std::size_t payload_size,
    encoded_data_metadata metadata,
    memory::memory_space& space,
    ::cuda::stream_ref stream,
    memory::reservation* reservation = nullptr);

  /**
   * @brief Adopt an existing regular file range without reading or decoding its contents.
   *
   * Range validation and reserving disk capacity occur before ownership is taken. Both borrowed
   * and owned ranges retain a reservation for their payload bytes. Metadata is kept in memory and
   * is not serialized to the file. The default borrows the file; owned opts into its deletion.
   */
  [[nodiscard]] static std::unique_ptr<encoded_data_representation> from_disk(
    std::filesystem::path path,
    std::size_t file_offset,
    std::size_t payload_size,
    encoded_data_metadata metadata,
    memory::memory_space& space,
    encoded_file_ownership ownership = encoded_file_ownership::borrowed);

  /**
   * @brief Bytes a caller-owned reservation in @p space must cover to hold @p payload_size encoded
   *        bytes there: the payload rounded up to whole host blocks for HOST, the payload itself
   *        for GPU and DISK.
   *
   * relocate() rejects a HOST reservation smaller than this. relocate() never accepts a GPU
   * reservation pointer, so the GPU value is for sizing a reservation that the caller attaches to
   * the target allocator's tracker before relocating. The DISK value describes the footprint of a
   * DISK copy; relocation to or from DISK is not supported yet.
   *
   * @throws std::invalid_argument if a HOST space's block size is not a power of two.
   * @throws std::overflow_error if rounding up to whole blocks overflows.
   */
  [[nodiscard]] static std::size_t relocation_reservation_bytes(const memory::memory_space& space,
                                                                std::size_t payload_size);

  ~encoded_data_representation() noexcept override;

  encoded_data_representation(const encoded_data_representation&)            = delete;
  encoded_data_representation& operator=(const encoded_data_representation&) = delete;
  encoded_data_representation(encoded_data_representation&&)                 = delete;
  encoded_data_representation& operator=(encoded_data_representation&&)      = delete;

  /** @brief Storage footprint, including the unused tail of allocated HOST blocks. */
  [[nodiscard]] std::size_t get_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_payload_size() const noexcept;
  [[nodiscard]] const encoded_data_metadata& get_metadata() const noexcept;

  /** @brief HOST allocation. Only the first get_payload_size() logical bytes are payload. */
  [[nodiscard]] const memory::fixed_multiple_blocks_allocation& get_host_buffer() const;
  [[nodiscard]] const rmm::device_buffer& get_device_buffer() const;
  [[nodiscard]] const std::filesystem::path& get_file_path() const;
  [[nodiscard]] std::size_t get_file_offset() const;

  /**
   * @brief Make a complete byte-preserving copy in a target space, leaving the source untouched.
   *
   * For transfers to GPU, @p stream belongs to the destination device. For transfers from GPU to
   * HOST, it belongs to the source device. HOST-only transfers do not use the stream. DISK sources
   * and targets are not supported yet.
   * An optional reservation must belong to the target space and cover
   * relocation_reservation_bytes(). HOST allocations draw from that reservation. GPU allocations
   * follow the target allocator's existing reservation tracker; pass no reservation pointer and
   * attach owned GPU reservations to the tracker before calling. A non-null GPU reservation
   * pointer is rejected.
   *
   * @throws std::logic_error if the source or the target is a DISK space (not supported yet).
   * @throws std::invalid_argument on a null target, a stream on the wrong device, or a reservation
   *         that belongs to another space, is too small, or is a GPU reservation.
   */
  [[nodiscard]] std::unique_ptr<idata_representation> relocate(
    const memory::memory_space* target,
    ::cuda::stream_ref stream,
    memory::reservation* reservation = nullptr) const override;

  [[nodiscard]] std::unique_ptr<idata_representation> clone(
    ::cuda::stream_ref stream) const override;
  void record_writer_event(::cuda::stream_ref writer_stream) override;
  [[nodiscard]] cudaEvent_t get_writer_event() const override;
  void rebind_stream(::cuda::stream_ref stream) override;

 private:
  struct storage;
  encoded_data_representation(memory::memory_space& space,
                              std::size_t payload_size,
                              encoded_data_metadata metadata);

  static std::unique_ptr<encoded_data_representation> allocate(memory::memory_space& space,
                                                               std::size_t payload_size,
                                                               encoded_data_metadata metadata,
                                                               ::cuda::stream_ref stream,
                                                               memory::reservation* reservation);
  void wait_for_writer() const;
  [[nodiscard]] void* device_data();

  std::size_t _payload_size;
  encoded_data_metadata _metadata;
  std::unique_ptr<storage> _storage;
};

}  // namespace cucascade
