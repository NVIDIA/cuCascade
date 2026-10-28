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

#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cucascade::io {

/**
 * @brief Raised by @c s3_request_authorizer implementations when credential
 *        acquisition or signing fails.
 *
 * Surfaces from:
 *   - missing or malformed static credentials (caught at provider construction
 *     or first call),
 *   - misconfigured endpoint / region,
 *   - underlying signing-library failure (HMAC / SHA256),
 *   - upstream credential broker errors (in future refresh-aware impls).
 *
 * Backends translate this into the broader IO error path of the caller.
 * Future IO-layer error types share this header.
 */
class credential_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Opening a local file paired descriptors of two different files.
 *
 * A local reactor opens a buffered descriptor and reopens it for @c O_DIRECT;
 * when the two do not name the same file (device and inode) on every attempt,
 * for example because the path is being replaced concurrently, the open fails
 * with this error rather than read the data of one file and the footer of
 * another.
 */
class descriptor_mismatch_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief A read found the object to be a different version from the one its
 *        datasource was opened on.
 *
 * Raised by the REST reactor when a conditional range GET is refused (412) or
 * when an accepted response does not carry the validator the open recorded.
 * Nothing from such a response is published; the error is terminal for the
 * read and is not retried.  @c observed_tag() is the response's ETag as
 * received, or empty when the response carried none or was refused (412).
 */
class object_changed_error : public std::runtime_error {
 public:
  object_changed_error(std::string object_path, std::string expected_tag, std::string observed_tag)
    : std::runtime_error(describe(object_path, observed_tag)),
      _object_path(std::move(object_path)),
      _expected_tag(std::move(expected_tag)),
      _observed_tag(std::move(observed_tag))
  {
  }

  [[nodiscard]] std::string const& object_path() const noexcept { return _object_path; }
  [[nodiscard]] std::string const& expected_tag() const noexcept { return _expected_tag; }
  [[nodiscard]] std::string const& observed_tag() const noexcept { return _observed_tag; }

 private:
  static std::string describe(std::string const& path, std::string const& observed)
  {
    return "object " + path + " changed since it was opened" +
           (observed.empty() ? " (no matching validator in the response)"
                             : " (response validator differs)");
  }

  std::string _object_path;
  std::string _expected_tag;
  std::string _observed_tag;
};

/**
 * @brief The backend does not implement this operation yet.
 *
 * Reported by every write and flush entry point of a backend that serves
 * reads only (the curl REST context and reactor), and by
 * the @c ioctx write front ends when the backend cannot write at all.  A
 * @c std::system_error with @c std::errc::not_supported, so callers that catch
 * @c std::system_error and compare its code keep working; the message names
 * the backend and operation and says "not yet implemented".
 */
class not_implemented_error : public std::system_error {
 public:
  /// @p what names the backend and the operation, e.g. "rest: host_write".
  explicit not_implemented_error(std::string const& what)
    : std::system_error(std::make_error_code(std::errc::not_supported),
                        what + ": not yet implemented")
  {
  }
};

}  // namespace cucascade::io
