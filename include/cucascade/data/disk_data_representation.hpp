/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace cucascade {

/**
 * @brief Data representation for a table stored on disk as a single file.
 *
 * Owns a disk_table_allocation that tracks the file path, column metadata,
 * and total data size. When this representation is destroyed, the backing
 * file is deleted (RAII lifecycle).
 *
 * Cloning is not supported for disk representations -- use a converter to
 * materialize the data in a different tier instead.
 */
class disk_data_representation : public idata_representation {
 public:
  /**
   * @brief Construct a disk_data_representation.
   *
   * @param disk_table Allocation describing the on-disk file and column layout.
   * @param memory_space The disk memory space where the data resides.
   */
  disk_data_representation(std::unique_ptr<memory::disk_table_allocation> disk_table,
                           memory::memory_space& memory_space);

  /**
   * @brief Destructor. Deletes the backing file on disk (best-effort, noexcept).
   */
  ~disk_data_representation() noexcept;

  /**
   * @brief Get the total number of bytes stored in the disk file.
   *
   * @return std::size_t The data size in bytes.
   */
  [[nodiscard]] std::size_t get_size_in_bytes() const override;

  /**
   * @copydoc idata_representation::get_uncompressed_data_size_in_bytes
   */
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;

  /**
   * @brief Cloning is not supported for disk representations.
   *
   * @throws cucascade::logic_error always.
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  /**
   * @brief Access the underlying disk table allocation.
   *
   * @return Const reference to the disk_table_allocation.
   */
  [[nodiscard]] const memory::disk_table_allocation& get_disk_table() const;

 private:
  std::unique_ptr<memory::disk_table_allocation> _disk_table;
};

}  // namespace cucascade
