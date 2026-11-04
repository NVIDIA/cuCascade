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

#include <cucascade/cudf/column_admission.hpp>
#include <cucascade/cudf/datasource.hpp>

#include <cudf/io/datasource.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cucascade::io {

/**
 * @brief A @c cudf::io::datasource that advises every sizeable read to the cache before issuing it.
 *
 * The cache keeps only the chunks that a request advised, and libcudf reads exact column-chunk
 * ranges without advising anything. This wrapper closes that gap for readers that do not call
 * @c datasource::fadvise themselves, such as libcudf's parquet reader: each read of at least
 * @c min_advise_bytes is passed to a @ref column_admission policy, and the chunk-aligned ranges it
 * admits are advised, at the priority it chose, before the read runs. Smaller reads, and reads
 * the policy refuses, go straight to the wrapped datasource and are not cached.
 *
 * Each advised read gets its own duplicate of the wrapped datasource, since an advise attaches
 * to one datasource's request handle. The duplicates live until this object is destroyed so that
 * asynchronous reads stay valid.
 */
class advising_datasource : public cudf::io::datasource {
 public:
  /**
   * @param base The datasource to read through.
   * @param admission Policy that decides what to advise. Shared between all the files of a scan.
   * @param path Object path of the file, the key into @p admission.
   * @param min_advise_bytes Reads smaller than this are not advised. 0 advises nothing.
   */
  advising_datasource(std::unique_ptr<cucascade::io::datasource> base,
                      std::shared_ptr<column_admission> admission,
                      std::string path,
                      std::size_t min_advise_bytes);

  [[nodiscard]] size_t size() const override;
  [[nodiscard]] bool supports_device_read() const override;
  [[nodiscard]] bool is_device_read_preferred(size_t size) const override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;
  std::future<std::unique_ptr<buffer>> host_read_async(size_t offset, size_t size) override;

  std::unique_ptr<buffer> device_read(size_t offset, size_t size, cudf_stream_type stream) override;
  size_t device_read(size_t offset, size_t size, uint8_t* dst, cudf_stream_type stream) override;
  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cudf_stream_type stream) override;

 private:
  /// The datasource a read of [offset, offset + size) must go through: the wrapped one, or a
  /// fresh duplicate that has advised the admitted ranges.
  cucascade::io::datasource& route(size_t offset, size_t size);

  std::unique_ptr<cucascade::io::datasource> _base;
  std::shared_ptr<column_admission> _admission;
  std::string _path;
  std::size_t _min_advise_bytes;
  std::mutex _mutex;
  std::vector<std::unique_ptr<cucascade::io::datasource>> _readers;
};

}  // namespace cucascade::io
