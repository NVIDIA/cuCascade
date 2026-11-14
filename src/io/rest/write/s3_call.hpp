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

#include <cucascade/io/rest/authorizer.hpp>

#include <sys/uio.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

/**
 * @file s3_call.hpp
 * @brief One logical S3 write request of the REST reactor with its retry envelope (the write
 *        path's analogue of the `data_op` of a data GET; docs/io/rest-transport.md "Writes").
 *
 * Worker-confined.  A call is owned by the worker's queues or by the connection slot running
 * it, never by the job it serves: the call owns its job through @c keepalive (a part upload, a
 * flush, a CreateMultipartUpload and the work waiting for it), so the last call of a job that
 * settles releases the job, and with it the upload session reference.
 */

namespace cucascade::io::rest::detail {

class upload_session;

/// Memory of the zero bodies (parts never written): 1 MiB of static, immutable zeros.
inline constexpr std::size_t zero_block_bytes = std::size_t{1} << 20;

/// @return the static zero block (never freed, never written: its lifetime cannot end while a
/// request is still sending from it).
[[nodiscard]] std::span<std::byte const> zero_block() noexcept;

/**
 * @brief A request body of @p bytes zeros as a transport body generator: windows of at most
 *        @p max_iovecs iovecs over @ref zero_block, the last one cut to the remaining size.
 *
 * Each call of the result yields the next window (an empty span at the end).  Build one per
 * attempt: the state lives in the closure.
 */
[[nodiscard]] std::function<std::span<iovec const>()> zero_body_generator(std::uint64_t bytes,
                                                                          std::size_t max_iovecs);

struct s3_call {
  enum class kind : std::uint8_t {
    put_object,       ///< PUT <key> (single PUT commit)
    create_upload,    ///< POST <key>?uploads
    upload_part,      ///< PUT <key>?partNumber=N&uploadId=U
    complete_upload,  ///< POST <key>?uploadId=U
    head_object,      ///< HEAD <key> (commit-outcome recovery after a lost Complete)
  };

  kind k{kind::put_object};
  object_ref object;
  std::string query;          ///< pre-encoded sub-resource query (s3::part_query, ...)
  bool if_none_match{false};  ///< create_new: `If-None-Match: *` on the PUT / Complete

  /// The body: the caller's iovecs, a part buffer, or (zero_body) zeros from zero_block().
  /// Copied into each attempt's exchange; the memory stays valid until the call settled.
  std::vector<iovec> body;
  std::uint64_t body_length{0};
  bool zero_body{false};

  std::size_t attempt{0};           ///< transient retries
  std::size_t auth_attempt{0};      ///< 403 re-sign retries (PUT / UploadPart)
  std::size_t complete_attempt{0};  ///< `200 OK` + `<Error>` re-sends of Complete
  bool stale_retried{false};        ///< the free stale-connection retry was used
  bool fresh{false};                ///< run on a fresh connection (the stale retry)
  /// An earlier attempt went out whole and its answer was lost: the server may have acted on
  /// it (a conditional commit that then answers 412 cannot be told from someone else's).
  bool sent_unanswered{false};

  // ---- outcome (valid in on_done) ----
  std::uint16_t status{0};
  std::string amz_code;      ///< the S3 <Code> of an error answer
  std::string etag;          ///< the ETag header (PUT / UploadPart / HEAD)
  std::string value;         ///< Create: the UploadId; Complete: the object ETag of the result
  std::error_code ec;        ///< the failure, mapped for s3::upload_state
  std::exception_ptr error;  ///< std::system_error(ec, "rest_reactor::<what>: ... for b/k")

  /// Runs on the worker thread once the call is final (success or error).  noexcept by contract.
  std::function<void(s3_call&)> on_done;
  /// The job this call serves (part, flush, create); released with the call.
  std::shared_ptr<void> keepalive;
  /// Commit requests (PUT, Complete, the recovery HEAD): the object's session, which remembers a
  /// lost commit answer across flushes.
  std::shared_ptr<upload_session> session;
  char const* what{"rest_reactor"};

  [[nodiscard]] bool has_body() const noexcept
  {
    return k == kind::put_object || k == kind::create_upload || k == kind::upload_part ||
           k == kind::complete_upload;
  }
  [[nodiscard]] request_method method() const noexcept
  {
    switch (k) {
      case kind::put_object:
      case kind::upload_part: return request_method::PUT;
      case kind::create_upload:
      case kind::complete_upload: return request_method::POST;
      case kind::head_object: return request_method::HEAD;
    }
    return request_method::PUT;
  }
  /// PUT / UploadPart carry data: a stall detector bounds them, a 403 is re-signed.
  [[nodiscard]] bool data_plane() const noexcept
  {
    return k == kind::put_object || k == kind::upload_part;
  }
};

}  // namespace cucascade::io::rest::detail
