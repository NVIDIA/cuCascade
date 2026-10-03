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
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/types.hpp>

#include <sys/uio.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cucascade::io::rest {

class upload_session;

/// What one REST transfer does.  @c get is a ranged data GET (reads); the
/// others implement whole-object uploads (see @c upload_session).
enum class rest_op_kind : std::uint8_t {
  get,           ///< ranged GET into the read's destination / staging
  put_object,    ///< single PUT of a whole staged object (commit, small objects)
  initiate_mpu,  ///< POST ?uploads= (CreateMultipartUpload)
  upload_part,   ///< PUT ?partNumber=N&uploadId=ID of one staged part
  complete_mpu,  ///< POST ?uploadId=ID with the part list (commit)
  abort_mpu,     ///< DELETE ?uploadId=ID (failure clean-up, best effort)
};

/// Read-callback source of one upload: libcurl pulls the request body from
/// @c buffers (staged part blocks, in object order) at a running cursor.  The
/// seek callback rewinds it (libcurl rewinds a body to resend it, e.g. on a
/// redirect or a reused connection that died), so the staged bytes are read as
/// many times as needed; they stay untouched until the upload succeeded.
struct buf_source {
  std::vector<iovec> buffers;   ///< body bytes, in order
  std::size_t size{0};          ///< sum of buffers[i].iov_len
  std::size_t cursor{0};        ///< body bytes handed to libcurl so far
  std::size_t active{0};        ///< buffer index of @c cursor
  std::size_t block_offset{0};  ///< offset of @c cursor inside buffers[active]

  /// Move the cursor to @p position (<= size).
  void seek(std::size_t position) noexcept
  {
    cursor       = position;
    active       = 0;
    block_offset = 0;
    while (active < buffers.size() && position >= buffers[active].iov_len) {
      position -= buffers[active].iov_len;
      ++active;
    }
    block_offset = position;
  }

  /// Copy up to @p capacity bytes at the cursor into @p out; returns the count.
  std::size_t read(char* out, std::size_t capacity) noexcept
  {
    std::size_t copied = 0;
    while (copied < capacity && active < buffers.size()) {
      auto const& block = buffers[active];
      auto const n      = std::min(block.iov_len - block_offset, capacity - copied);
      if (n > 0) {
        std::memcpy(out + copied, static_cast<char const*>(block.iov_base) + block_offset, n);
      }
      copied += n;
      block_offset += n;
      if (block_offset >= block.iov_len) {
        ++active;
        block_offset = 0;
      }
    }
    cursor += copied;
    return copied;
  }
};

/// Write-callback target for one in-flight transfer.  libcurl hands the
/// response body to the reactor in arbitrarily-sized pieces; @c buf_sink
/// scatters them across @c buffers (the chunk's destination iovecs, in file
/// order) at a running cursor — so a single contiguous ranged GET that fuses
/// several adjacent segments lands each segment in its own buffer — and ALWAYS
/// reports the full incoming size back to curl (never a short count, which
/// would abort the transfer with CURLE_WRITE_ERROR).  Bytes beyond @c capacity
/// are counted in @c total_received but not stored, so the reactor can detect a
/// server that ignored the Range header (e.g. returned the whole object).
struct buf_sink {
  std::span<iovec> buffers;  // destination buffers, in file order
  std::size_t capacity{0};   // Σ buffers[i].iov_len
  std::size_t active{0};     // index of the buffer currently being filled
  std::size_t cursor{0};     // bytes written into buffers[active]
  std::size_t written{0};    // total bytes written across all buffers
  std::size_t total_received{0};

  void reset() noexcept
  {
    active         = 0;
    cursor         = 0;
    written        = 0;
    total_received = 0;
  }
};

/// Response headers the reactor inspects on completion.  @c content_range
/// validates that a 206 honored the requested byte range; @c retry_after drives
/// the retry delay when the server asks the client to back off.
struct header_capture {
  std::string content_range;
  std::string retry_after;
  /// ETag of the final response block (uploads: the part's ETag).
  std::string etag;

  void reset() noexcept
  {
    content_range.clear();
    retry_after.clear();
    etag.clear();
  }
};

/// REST-specific retry envelope around a backend-neutral physical operation.
/// The operation owns any CuCascade staging allocation through staging_owner,
/// so its iovecs remain stable across retries and until a CUDA event drains.
struct rest_io_op_request {
  object_ref object;
  std::unique_ptr<io_op_request> op;
  std::size_t attempt{0};
  std::size_t auth_attempt{0};
  bool needs_staging{false};
  std::size_t logical_bytes{0};
  /// Engine bookkeeping: id of the grouped request the operation was planned
  /// from (@c request_meta::id) and its scheduling class.
  std::uint64_t group_id{0};
  request_class cls{request_class::read};
  /// Engine bookkeeping: the operation's terminal state was accounted for.
  bool settled{false};

  // -- uploads (kind != get) ----------------------------------------------------
  rest_op_kind kind{rest_op_kind::get};
  /// Upload session the operation belongs to (null for reads).
  std::shared_ptr<upload_session> session;
  /// Part number of an @c upload_part.
  std::uint32_t part_number{0};
  /// Request body of a PUT (staged bytes) ...
  buf_source source;
  /// ... or of a POST (CompleteMultipartUpload XML; empty for Initiate).
  std::string request_body;
  /// Response body of the current attempt (control-plane XML).
  std::string response_body;

  [[nodiscard]] bool is_device() const noexcept
  {
    return op != nullptr && op->device_copy != nullptr;
  }

  [[nodiscard]] cudaError_t copy_h2d_async(cudaEvent_t event = nullptr) const noexcept
  {
    if (!is_device()) return cudaSuccess;
    return op->device_copy->copy_async(op->io_rng, op->iovecs, event);
  }
};

}  // namespace cucascade::io::rest
