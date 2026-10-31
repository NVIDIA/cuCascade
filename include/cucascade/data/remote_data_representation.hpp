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
#include <cucascade/memory/disk_table.hpp>

#include <memory>
#include <string>

namespace cucascade {

namespace detail {
/// The object a REMOTE representation's bytes live in, as the converter that wrote it left it:
/// what reading it back needs, and the removal of the object once released. Defined by the
/// cucascade-cudf layer (the core library only holds and releases it).
struct remote_object;
}  // namespace detail

/**
 * @brief A batch spilled to a REMOTE memory space: one object in an object store.
 *
 * The object holds the column buffers at the DISK file layout (4 KiB-aligned offsets from 0, gaps
 * read back as zeros); the column metadata stays in memory, as for disk_data_representation.
 * Representations are created by the HOST/GPU -> REMOTE converters of the cucascade-cudf layer
 * (see cucascade/cudf/builtin_converters.hpp). Destroying one removes its object with an
 * asynchronous DeleteObject through the io context that wrote it (best effort: an object left
 * behind by a failed delete or a crash is collected by the bucket's lifecycle rule, see
 * docs/io/remote-tier-design.md).
 */
class remote_data_representation : public idata_representation {
 public:
  /**
   * @param object_table Column layout; `file_path` is the object URI, `data_size` its size.
   * @param memory_space The REMOTE memory space the object belongs to.
   * @param object The written object (from the REMOTE converters); may be null, in which case the
   *        representation neither removes the object nor can be converted back.
   */
  remote_data_representation(std::unique_ptr<memory::disk_table_allocation> object_table,
                             memory::memory_space& memory_space,
                             std::shared_ptr<detail::remote_object const> object);

  /// Releases the object handle; the last release issues the (asynchronous) removal. Noexcept.
  ~remote_data_representation() noexcept override;

  remote_data_representation(remote_data_representation const&)            = delete;
  remote_data_representation& operator=(remote_data_representation const&) = delete;

  /// Bytes of the object.
  [[nodiscard]] std::size_t get_size_in_bytes() const override;

  /// @copydoc idata_representation::get_uncompressed_data_size_in_bytes
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;

  /// Cloning is not supported (it would need a server-side copy). @throws cucascade::logic_error.
  std::unique_ptr<idata_representation> clone(::cuda::stream_ref stream) const override;

  /// The column layout; `file_path` is the object URI.
  [[nodiscard]] const memory::disk_table_allocation& get_object_table() const;

  /// The object URI, e.g. "s3://bucket/prefix/run/batch_....cucascade".
  [[nodiscard]] const std::string& get_object_uri() const;

  /// The written object (internal: used by the REMOTE converters to read it back).
  [[nodiscard]] std::shared_ptr<detail::remote_object const> const& get_remote_object()
    const noexcept
  {
    return _object;
  }

 private:
  std::unique_ptr<memory::disk_table_allocation> _object_table;
  std::shared_ptr<detail::remote_object const> _object;
};

}  // namespace cucascade
