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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace cucascade::io::rest::s3 {

//===----------------------------------------------------------------------===//
// Minimal XML helpers for S3 responses (hand-rolled, no XML dependency)
//===----------------------------------------------------------------------===//

/**
 * @brief Single-pass unescape of the five predefined XML entities.
 *
 * Unknown sequences (e.g. numeric character references, which S3 does not emit
 * for the fields cuCascade reads) pass through verbatim.
 */
[[nodiscard]] std::string xml_unescape(std::string_view s);

/// Escape @c & @c < @c > @c " @c ' as the predefined XML entities.
[[nodiscard]] std::string xml_escape(std::string_view s);

/// Trim leading / trailing ASCII whitespace.
[[nodiscard]] std::string_view xml_trim(std::string_view s) noexcept;

/**
 * @brief Raw (still escaped) text between the first @c <tag> at or after
 *        @p from and the following @c </tag>.
 *
 * Only matches attribute-free open tags (S3 response elements carry none) and
 * does not handle nesting of the same tag. @c std::nullopt when the element is
 * absent or unterminated.
 */
[[nodiscard]] std::optional<std::string_view> xml_element_text(std::string_view xml,
                                                               std::string_view tag,
                                                               std::size_t from = 0);

/**
 * @brief Name of the document's root element (prologue, comments, and
 *        whitespace skipped), or empty when @p xml holds no element.
 */
[[nodiscard]] std::string_view xml_root_name(std::string_view xml) noexcept;

//===----------------------------------------------------------------------===//
// S3 error / multipart-upload documents
//===----------------------------------------------------------------------===//

/// Parsed S3 @c <Error> document.
struct s3_error {
  std::string code;        ///< @c <Code>, e.g. @c "NoSuchUpload" (may be empty)
  std::string message;     ///< @c <Message> (may be empty)
  std::string request_id;  ///< @c <RequestId> (may be empty)
};

/**
 * @brief Parse an S3 error body.
 *
 * Returns the error iff the document's root element is @c <Error> — this also
 * detects the S3 quirk of an HTTP 200 CompleteMultipartUpload response whose
 * body is an error. Fields are XML-unescaped and trimmed. Never throws on
 * malformed input (returns @c std::nullopt for a non-error / non-XML body).
 */
[[nodiscard]] std::optional<s3_error> parse_s3_error(std::string_view xml);

/**
 * @brief Extract the upload id from a CreateMultipartUpload response
 *        (@c <InitiateMultipartUploadResult><UploadId>...</UploadId>).
 *
 * @return The XML-unescaped, trimmed, non-empty upload id.
 * @throw std::runtime_error for an @c <Error> body (message carries the S3 code),
 *        a different root element, or a missing / empty @c <UploadId>.
 */
[[nodiscard]] std::string parse_initiate_multipart_upload(std::string_view xml);

/// One uploaded part, as listed in a CompleteMultipartUpload request.
struct part_record {
  std::uint32_t part_number{0};  ///< 1..10000
  std::string etag;              ///< ETag header value of the UploadPart response, quotes included
};

/// Highest part number S3 accepts.
inline constexpr std::uint32_t max_part_number = 10'000;

/**
 * @brief Build the CompleteMultipartUpload request body.
 *
 * @code
 * <CompleteMultipartUpload xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
 *   <Part><PartNumber>1</PartNumber><ETag>&quot;etag&quot;</ETag></Part>...
 * </CompleteMultipartUpload>
 * @endcode
 *
 * @param parts  Parts in strictly ascending part-number order (S3 requirement).
 * @throw std::invalid_argument when @p parts is empty, a part number is outside
 *        [1, @c max_part_number], numbers are not strictly ascending, or an ETag
 *        is empty.
 */
[[nodiscard]] std::string build_complete_multipart_body(std::span<part_record const> parts);

}  // namespace cucascade::io::rest::s3
