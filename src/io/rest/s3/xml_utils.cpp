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

#include <cucascade/io/rest/s3/xml_utils.hpp>

#include <cctype>
#include <stdexcept>
#include <string>

namespace cucascade::io::rest::s3 {

namespace {

bool is_xml_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

/// Unescaped, trimmed text of @p tag inside @p xml, or empty when absent.
std::string field(std::string_view xml, std::string_view tag)
{
  auto const raw = xml_element_text(xml, tag);
  return raw.has_value() ? xml_unescape(xml_trim(*raw)) : std::string{};
}

}  // namespace

std::string xml_unescape(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '&') {
      if (s.compare(i, 5, "&amp;") == 0) {
        out += '&';
        i += 5;
        continue;
      }
      if (s.compare(i, 4, "&lt;") == 0) {
        out += '<';
        i += 4;
        continue;
      }
      if (s.compare(i, 4, "&gt;") == 0) {
        out += '>';
        i += 4;
        continue;
      }
      if (s.compare(i, 6, "&quot;") == 0) {
        out += '"';
        i += 6;
        continue;
      }
      if (s.compare(i, 6, "&apos;") == 0) {
        out += '\'';
        i += 6;
        continue;
      }
    }
    out += s[i];
    ++i;
  }
  return out;
}

std::string xml_escape(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char const c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out += c; break;
    }
  }
  return out;
}

std::string_view xml_trim(std::string_view s) noexcept
{
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

std::optional<std::string_view> xml_element_text(std::string_view xml,
                                                 std::string_view tag,
                                                 std::size_t from)
{
  std::string const open  = "<" + std::string{tag} + ">";
  std::string const close = "</" + std::string{tag} + ">";
  auto const o            = xml.find(open, from);
  if (o == std::string_view::npos) { return std::nullopt; }
  auto const s = o + open.size();
  auto const c = xml.find(close, s);
  if (c == std::string_view::npos) { return std::nullopt; }
  return xml.substr(s, c - s);
}

std::string_view xml_root_name(std::string_view xml) noexcept
{
  std::size_t i = 0;
  while (i < xml.size()) {
    while (i < xml.size() && is_xml_space(xml[i])) {
      ++i;
    }
    if (i >= xml.size() || xml[i] != '<') { return {}; }
    // Skip the prologue / processing instructions and comments.
    if (xml.compare(i, 2, "<?") == 0) {
      auto const end = xml.find("?>", i + 2);
      if (end == std::string_view::npos) { return {}; }
      i = end + 2;
      continue;
    }
    if (xml.compare(i, 4, "<!--") == 0) {
      auto const end = xml.find("-->", i + 4);
      if (end == std::string_view::npos) { return {}; }
      i = end + 3;
      continue;
    }
    auto const name_begin = i + 1;
    auto name_end         = name_begin;
    while (name_end < xml.size() && !is_xml_space(xml[name_end]) && xml[name_end] != '>' &&
           xml[name_end] != '/') {
      ++name_end;
    }
    if (name_end >= xml.size()) { return {}; }  // unterminated open tag
    return xml.substr(name_begin, name_end - name_begin);
  }
  return {};
}

std::optional<s3_error> parse_s3_error(std::string_view xml)
{
  if (xml_root_name(xml) != "Error") { return std::nullopt; }
  s3_error err;
  err.code       = field(xml, "Code");
  err.message    = field(xml, "Message");
  err.request_id = field(xml, "RequestId");
  return err;
}

std::string parse_initiate_multipart_upload(std::string_view xml)
{
  if (auto const err = parse_s3_error(xml); err.has_value()) {
    throw std::runtime_error("parse_initiate_multipart_upload: S3 error " +
                             (err->code.empty() ? std::string{"<no code>"} : err->code) +
                             (err->message.empty() ? std::string{} : ": " + err->message));
  }
  auto const root = xml_root_name(xml);
  if (root != "InitiateMultipartUploadResult") {
    throw std::runtime_error(
      "parse_initiate_multipart_upload: not an InitiateMultipartUploadResult (root '" +
      std::string{root} + "')");
  }
  auto upload_id = field(xml, "UploadId");
  if (upload_id.empty()) {
    throw std::runtime_error("parse_initiate_multipart_upload: missing or empty <UploadId>");
  }
  return upload_id;
}

std::string build_complete_multipart_body(std::span<part_record const> parts)
{
  if (parts.empty()) { throw std::invalid_argument("build_complete_multipart_body: no parts"); }
  std::string out =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">";
  std::uint32_t prev = 0;
  for (auto const& part : parts) {
    if (part.part_number == 0 || part.part_number > max_part_number) {
      throw std::invalid_argument("build_complete_multipart_body: part number " +
                                  std::to_string(part.part_number) + " outside [1, 10000]");
    }
    if (part.part_number <= prev) {
      throw std::invalid_argument(
        "build_complete_multipart_body: part numbers must be strictly ascending (" +
        std::to_string(part.part_number) + " after " + std::to_string(prev) + ")");
    }
    if (part.etag.empty()) {
      throw std::invalid_argument("build_complete_multipart_body: empty ETag for part " +
                                  std::to_string(part.part_number));
    }
    prev = part.part_number;
    out += "<Part><PartNumber>";
    out += std::to_string(part.part_number);
    out += "</PartNumber><ETag>";
    out += xml_escape(part.etag);
    out += "</ETag></Part>";
  }
  out += "</CompleteMultipartUpload>";
  return out;
}

}  // namespace cucascade::io::rest::s3
