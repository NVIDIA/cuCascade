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

#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/column_admission.hpp>
#include <cucascade/io/io_context.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cucascade::io {

/**
 * @brief Reads one object through the prefetching cache, advising what it reads first.
 *
 * The cache keeps only the chunks that a request advised, and a reader such as libcudf's parquet
 * reader reads exact column-chunk ranges without advising anything. This class closes that gap:
 * each read of at least @c min_advise_bytes is passed to a @ref column_admission policy, and the
 * chunk-aligned ranges it admits are advised to the cache, at the priority it chose, before the
 * read runs. Smaller reads, and reads the policy refuses, go straight to the cache and are not
 * kept.
 *
 * It has no libcudf dependency: a class that subclasses @c cudf::io::datasource wraps it, and is
 * built together with the libcudf it runs with.
 *
 * Each advised read gets its own request handle, since an advise attaches to one handle. The
 * handles live until this object is destroyed so that asynchronous reads stay valid. Safe to call
 * from several threads.
 */
class advising_reader {
 public:
  /**
   * @param context The io context that serves the reads and owns the cache.
   * @param object The opened object.
   * @param admission Policy that decides what to advise. Shared between all the files of a scan.
   * @param path Object path of the file, the key into @p admission.
   * @param min_advise_bytes Reads smaller than this are not advised. 0 advises nothing.
   */
  advising_reader(std::shared_ptr<ioctx> context,
                  std::shared_ptr<io_object> object,
                  std::shared_ptr<column_admission> admission,
                  std::string path,
                  std::size_t min_advise_bytes);
  ~advising_reader();

  advising_reader(advising_reader const&)            = delete;
  advising_reader& operator=(advising_reader const&) = delete;

  /// Size of the object in bytes.
  [[nodiscard]] std::size_t size() const;

  /// Whether reads can go straight into device memory.
  [[nodiscard]] bool supports_device_read() const;

  /// Read [offset, offset + size) into host memory and wait for it.
  [[nodiscard]] std::size_t host_read(std::size_t offset, std::size_t size, std::uint8_t* dst);

  /// Start reading [offset, offset + size) into host memory.
  [[nodiscard]] std::future<std::size_t> host_read_async(std::size_t offset,
                                                         std::size_t size,
                                                         std::uint8_t* dst);

  /// Start reading [offset, offset + size) into device memory. The read is not ordered with the
  /// work already on @p stream; synchronize the stream before using the data.
  [[nodiscard]] std::future<std::size_t> device_read_async(std::size_t offset,
                                                           std::size_t size,
                                                           std::uint8_t* dst,
                                                           ::cuda::stream_ref stream);

 private:
  /// The request handle a read of [offset, offset + size) must go through: a new one that has
  /// advised the admitted ranges, or null if the read is small, refused, or there is no cache.
  cache::cache_handle* route(std::size_t offset, std::size_t size);

  std::shared_ptr<ioctx> _context;
  std::shared_ptr<io_object> _object;
  std::shared_ptr<column_admission> _admission;
  std::string _path;
  std::size_t _min_advise_bytes;
  std::mutex _mutex;
  std::vector<std::unique_ptr<cache::cache_handle>> _handles;
};

}  // namespace cucascade::io
