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

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/authorizer.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <utility>

namespace cucascade::io::rest {

/**
 * @brief Test-only @c request_authorizer that returns a canned
 *        @c authorized_request or throws.
 *
 * Header-only so test suites can include it without a separate library target.
 * Produces deterministic output for unit tests of code that consumes
 * @c request_authorizer through the abstract base class — typical pattern is:
 *
 * @code
 *   auto provider = std::make_shared<mock_authorizer>(
 *     authorized_request{"https://canned/url", {}});
 *   // ... wire it into a reactor_context and drive a read ...
 *   CHECK(provider->call_count() == 1);
 *   CHECK(provider->last_bucket() == "mybucket");
 * @endcode
 *
 * Default behavior: every call to @c authorize returns the
 * @c authorized_request passed to the constructor verbatim (independent of
 * @p obj / @p method) so tests can verify "the reactor passed our URL +
 * headers to libcurl unchanged". To exercise error paths, call @c set_throw to
 * make subsequent calls throw @c cucascade::io::credential_error.
 *
 * Thread safety: counters are atomic; @c last_bucket / @c last_key are
 * guarded by an internal mutex. Safe to share across threads when tests
 * exercise concurrent reactor paths.
 */
class mock_authorizer final : public request_authorizer {
 public:
  explicit mock_authorizer(authorized_request canned) : _canned(std::move(canned)) {}

  authorized_request authorize(object_ref const& obj,
                               request_method method,
                               std::chrono::seconds timeout) override
  {
    ++_call_count;
    if (method == request_method::GET) ++_get_count;
    if (method == request_method::HEAD) ++_head_count;
    {
      std::scoped_lock lk{_last_mtx};
      _last_bucket  = obj.bucket;
      _last_key     = obj.key;
      _last_timeout = timeout;
      _last_method  = method;
      _last_query.clear();
    }
    if (_should_throw.load()) {
      std::string msg;
      {
        std::scoped_lock lk{_last_mtx};
        msg = _throw_msg.empty() ? std::string{"mock_authorizer: forced failure"} : _throw_msg;
      }
      throw credential_error(msg);
    }
    return _canned;
  }

  /// Any-method request (write path). Returns the canned URL with
  /// @c spec.canonical_query appended verbatim (after '?', or '&' when the canned
  /// URL already has a query) and the canned headers followed by
  /// @c spec.extra_headers, so a loopback server can route on method + query.
  authorized_request authorize_request(request_spec const& spec,
                                       std::chrono::seconds timeout) override
  {
    ++_call_count;
    ++_request_count;
    if (spec.method == request_method::GET) ++_get_count;
    if (spec.method == request_method::HEAD) ++_head_count;
    {
      std::scoped_lock lk{_last_mtx};
      _last_bucket  = spec.object.bucket;
      _last_key     = spec.object.key;
      _last_timeout = timeout;
      _last_method  = spec.method;
      _last_query   = spec.canonical_query;
    }
    if (_should_throw.load()) {
      std::string msg;
      {
        std::scoped_lock lk{_last_mtx};
        msg = _throw_msg.empty() ? std::string{"mock_authorizer: forced failure"} : _throw_msg;
      }
      throw credential_error(msg);
    }
    authorized_request out = _canned;
    if (!spec.canonical_query.empty()) {
      out.url += out.url.find('?') == std::string::npos ? '?' : '&';
      out.url += spec.canonical_query;
    }
    out.headers.insert(out.headers.end(), spec.extra_headers.begin(), spec.extra_headers.end());
    return out;
  }

  /// Subsequent calls throw @c credential_error with @p msg (or default).
  void set_throw(std::string msg = {})
  {
    {
      std::scoped_lock lk{_last_mtx};
      _throw_msg = std::move(msg);
    }
    _should_throw.store(true);
  }

  /// Stop throwing.
  void clear_throw()
  {
    _should_throw.store(false);
    {
      std::scoped_lock lk{_last_mtx};
      _throw_msg.clear();
    }
  }

  [[nodiscard]] int call_count() const noexcept { return _call_count.load(); }
  [[nodiscard]] int get_count() const noexcept { return _get_count.load(); }
  [[nodiscard]] int head_count() const noexcept { return _head_count.load(); }
  /// Number of @c authorize_request calls (any method).
  [[nodiscard]] int request_count() const noexcept { return _request_count.load(); }

  [[nodiscard]] request_method last_method() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_method;
  }
  [[nodiscard]] std::string last_query() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_query;
  }

  [[nodiscard]] std::string last_bucket() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_bucket;
  }
  [[nodiscard]] std::string last_key() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_key;
  }
  [[nodiscard]] std::chrono::seconds last_timeout() const
  {
    std::scoped_lock lk{_last_mtx};
    return _last_timeout;
  }

 private:
  authorized_request _canned;
  std::atomic<int> _call_count{0};
  std::atomic<int> _get_count{0};
  std::atomic<int> _head_count{0};
  std::atomic<int> _request_count{0};
  std::atomic<bool> _should_throw{false};
  mutable std::mutex _last_mtx;
  std::string _last_bucket;
  std::string _last_key;
  std::chrono::seconds _last_timeout{0};
  request_method _last_method{request_method::GET};
  std::string _last_query;
  std::string _throw_msg;
};

}  // namespace cucascade::io::rest
