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

// The page/entry types live under rest/s3 but are S3-PROTOCOL shapes
// (ListObjectsV2 responses), not REST-transport shapes — any backend that
// lists an S3-compatible store speaks them, whatever its data plane.
#include <cucascade/io/rest/s3/list_parser.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::io::rest {

// ---------------------------------------------------------------------------
// object_store_lister
// ---------------------------------------------------------------------------

/**
 * @brief Backend-ioctx-owned control-plane component for object-store listing.
 *
 * Each S3-capable ioctx composes one instance and injects its own page-fetch
 * operation.  The lister owns the ListObjectsV2 protocol logic — canonical
 * query construction, page-size clamping, response parsing, continuation-token
 * advancement, the anti-loop guards, and the @c max_scanned / @c max_matches
 * caps.  Transport, retries, signing, and admission policy stay behind the
 * injected fetch.
 */
class object_store_lister {
 public:
  /// Fetches one raw ListObjectsV2 response body for @p bucket.
  /// @p canonical_query is the fully encoded query string in SigV4 canonical
  /// order; @p prefix repeats the (unencoded) prefix inside it for fetchers
  /// that address by prefix.
  using page_fetch_fn = std::function<std::string(
    std::string_view bucket, std::string_view prefix, std::string_view canonical_query)>;

  /// @p error_context prefixes the messages of exceptions the lister itself
  /// throws; exceptions from the fetch, the parser, or the sink propagate
  /// unwrapped.
  object_store_lister(page_fetch_fn fetch,
                      std::size_t max_scanned,
                      std::size_t max_matches,
                      std::string error_context)
    : _fetch(std::move(fetch)),
      _max_scanned(max_scanned),
      _max_matches(max_matches),
      _error_context(std::move(error_context))
  {
  }

  /// Stream ListObjectsV2 pages under @p prefix to @p sink, one call per
  /// page.  @p sink returns false to stop early.  @p page_size is clamped
  /// to [1,1000] (0 and >1000 mean 1000).  Throws (never truncates) on a
  /// truncated page without a continuation token, and once more than
  /// @p max_scanned entries have been scanned across pages.
  void list_objects_paged(std::string_view bucket,
                          std::string_view prefix,
                          std::size_t page_size,
                          std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                          std::optional<std::size_t> max_scanned = std::nullopt);

  /// Whole-listing convenience over @c list_objects_paged: every object under
  /// @p prefix, in document order, with sizes.  Throws (never truncates) when
  /// the accumulated entries would exceed @p max_keys.
  [[nodiscard]] std::vector<s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt);

  /// The configured matched cap for glob resolution.
  [[nodiscard]] std::size_t list_max_matches() const noexcept { return _max_matches; }

 private:
  page_fetch_fn _fetch;
  std::size_t _max_scanned;
  std::size_t _max_matches;
  std::string _error_context;
};

}  // namespace cucascade::io::rest
