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

#include <cucascade/io/rest/object_store_lister.hpp>
#include <cucascade/io/rest/s3/sigv4.hpp>

#include <stdexcept>
#include <string>

namespace cucascade::io::rest {

void object_store_lister::list_objects_paged(
  std::string_view bucket,
  std::string_view prefix,
  std::size_t page_size,
  std::function<bool(s3::list_objects_v2_page const&)> const& sink,
  std::optional<std::size_t> max_scanned)
{
  std::size_t const clamped     = (page_size == 0 || page_size > 1000) ? 1000 : page_size;
  std::size_t const scanned_cap = max_scanned.value_or(_max_scanned);

  std::size_t scanned = 0;
  std::string token;
  bool truncated = false;
  do {
    // SigV4 canonical order = byte order of the encoded keys; for these params
    // that is continuation-token < list-type < max-keys < prefix.
    std::string query;
    if (!token.empty()) {
      query += "continuation-token=";
      query += s3::uri_encode(token, /*encode_slash=*/true);
      query += '&';
    }
    query += "list-type=2&max-keys=";
    query += std::to_string(clamped);
    query += "&prefix=";
    query += s3::uri_encode(prefix, /*encode_slash=*/true);

    auto const page = s3::parse_list_objects_v2(_fetch(bucket, prefix, query));

    scanned += page.entries.size();
    if (scanned > scanned_cap) {
      throw std::runtime_error(_error_context + ": scanned more than " +
                               std::to_string(scanned_cap) + " objects under s3://" +
                               std::string(bucket) + "/" + std::string(prefix) +
                               " — narrow the glob prefix");
    }
    if (page.is_truncated && page.next_continuation_token.empty()) {
      throw std::runtime_error(_error_context +
                               ": truncated ListObjectsV2 page without a continuation token for "
                               "s3://" +
                               std::string(bucket) + "/" + std::string(prefix));
    }
    // A truncated page must contain entries and advance the token. Together
    // with scanned_cap these bound pagination for non-conforming backends that
    // would otherwise loop on empty or non-advancing pages.
    if (page.is_truncated && page.entries.empty()) {
      throw std::runtime_error(_error_context +
                               ": truncated ListObjectsV2 page with no entries for s3://" +
                               std::string(bucket) + "/" + std::string(prefix));
    }
    if (page.is_truncated && page.next_continuation_token == token) {
      throw std::runtime_error(_error_context +
                               ": ListObjectsV2 continuation token did not advance for s3://" +
                               std::string(bucket) + "/" + std::string(prefix));
    }
    truncated = page.is_truncated;
    token     = page.next_continuation_token;
    if (!sink(page)) { return; }
  } while (truncated);
}

std::vector<s3::list_entry> object_store_lister::list_objects(std::string_view bucket,
                                                              std::string_view prefix,
                                                              std::size_t page_size,
                                                              std::optional<std::size_t> max_keys)
{
  std::size_t const keys_cap = max_keys.value_or(_max_matches);
  std::vector<s3::list_entry> out;
  list_objects_paged(bucket, prefix, page_size, [&](s3::list_objects_v2_page const& page) {
    if (out.size() + page.entries.size() > keys_cap) {
      throw std::runtime_error(_error_context + ": more than " + std::to_string(keys_cap) +
                               " objects under s3://" + std::string(bucket) + "/" +
                               std::string(prefix) + " — narrow the glob prefix");
    }
    out.insert(out.end(), page.entries.begin(), page.entries.end());
    return true;
  });
  return out;
}

}  // namespace cucascade::io::rest
