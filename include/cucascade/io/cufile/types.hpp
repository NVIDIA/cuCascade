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

#include <cucascade/io/io_request.hpp>

#include <kvikio/shim/cufile_h_wrapper.hpp>

#include <sys/types.h>

#include <cstddef>

namespace cucascade::io::cufile {

/**
 * @brief Argument block for @c cuFileReadAsync.
 *
 * cuFile's async API takes the size and offsets by pointer so the caller may
 * refine them right up to execution time (see kvikIO's @c StreamFuture): the
 * pointers must stay valid until the stream has drained past the read, and
 * @c bytes_done is written by the stream, not at submission.  The reactor owns
 * one of these per in-flight cuFile operation and frees it only after the
 * operation's CUDA event has fired.
 */
struct cufile_stream_args {
  std::size_t size{0};
  off_t file_offset{0};
  off_t devPtr_offset{0};
  ssize_t bytes_done{0};
};

}  // namespace cucascade::io::cufile
