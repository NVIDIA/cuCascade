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

#pragma once

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace cucascade::io::s3rdma {

// ---------------------------------------------------------------------------
// s3rdma_ioctx (placeholder)
// ---------------------------------------------------------------------------

/**
 * @brief Non-constructible placeholder for the S3-over-RDMA I/O context.
 *
 * The planned backend uses RDMA for S3 range-read payloads. Data lands in
 * registered GPU memory and is copied on device to the caller's destination.
 * LIST and HEAD use the configured HTTP(S) control plane; glob resolution
 * uses LIST results.
 *
 * Integration uses @c io_context_type::s3rdma and
 * @c io_context_registry::replace_ioctx, composes @c rest::object_store_lister
 * with its own page fetch for LIST, and overrides
 * @c templated_ioctx::on_device_dispatch_failure.
 *
 * This declaration has no implementation or factory registration. Its deleted
 * constructor prevents accidental use.
 *
 * @see https://github.com/sirius-db/sirius/blob/dev/experimental/s3-rdma-transport-design.md
 */
class s3rdma_ioctx : public ioctx {
 public:
  s3rdma_ioctx() = delete;

  [[nodiscard]] io_context_type type() const noexcept override;

  void shutdown() noexcept override;

  [[nodiscard]] bool supports(std::string_view path) const noexcept override;

  [[nodiscard]] bool supports_device_read() const noexcept override;
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override;
  [[nodiscard]] bool supports_vector_host_read() const noexcept override;
  [[nodiscard]] cache::prefetching_stage preferred_prefetching_stage() const noexcept override;

  [[nodiscard]] std::vector<byte_range> align_and_coalesce(
    std::span<const byte_range> ranges,
    std::optional<size_t> alignment = std::nullopt) const noexcept override;

  size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) override;

  exec::semi_future<size_t> host_read_async_io(const io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst) noexcept override;

  exec::semi_future<size_t> device_read_async_io(const io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 rmm::cuda_stream_view stream) noexcept override;

  exec::semi_future<size_t> host_to_device_read_async_io(
    const io_object& obj,
    std::span<io_object_segment> slices,
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) noexcept override;

  exec::semi_future<size_t> host_read_ranges_async_io(
    const io_object& obj, std::span<io_object_segment> segments) noexcept override;

 protected:
  std::shared_ptr<io_object> create_io_object(std::string path) override;
};

static_assert(!std::is_default_constructible_v<s3rdma_ioctx>,
              "s3rdma_ioctx is a placeholder and must stay non-constructible "
              "until the backend implementation lands");

}  // namespace cucascade::io::s3rdma
