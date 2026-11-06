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

// S3 response fixtures. The 206 has the field set captured from the live endpoint in
// gpuio docs/research/dev_box_probes.md §7 (noaa-ghcn-pds, us-east-1: 206 Partial Content,
// Accept-Ranges, Content-Range bytes 0-1048575/11441354, Content-Length, ETag, x-amz-request-id,
// x-amz-id-2, Server: AmazonS3; no Connection header, no chunked) plus the fields S3 always adds
// (Date, Last-Modified, Content-Type, x-amz-server-side-encryption, x-amz-version-id,
// Cache-Control): 13 fields, 513 bytes. Identifier values are representative, not captured.
// Error bodies follow the documented S3 XML error format (s3_line_rate.md §2.1, §2.8).

#pragma once

#include <string_view>

namespace s3_fixtures {

inline constexpr std::string_view head_206 =
  "HTTP/1.1 206 Partial Content\r\n"
  "x-amz-id-2: Xr9GhZqE0QnQ1wF2c5Vx3mI1yP7uN0bK4sJ8tL6dA2hR5eC9oW3vB1gT7kM0zY4pS6nD8fH2jU5qE=\r\n"
  "x-amz-request-id: 5RZ8XK1S2V6QH9TB\r\n"
  "Date: Mon, 12 Oct 2026 17:21:03 GMT\r\n"
  "Last-Modified: Tue, 03 Mar 2026 09:14:55 GMT\r\n"
  "ETag: \"6b1f5c2b7f1a9e0d3c4b5a6978879a5c\"\r\n"
  "x-amz-server-side-encryption: AES256\r\n"
  "x-amz-version-id: null\r\n"
  "Cache-Control: no-cache\r\n"
  "Accept-Ranges: bytes\r\n"
  "Content-Range: bytes 0-1048575/11441354\r\n"
  "Content-Type: binary/octet-stream\r\n"
  "Server: AmazonS3\r\n"
  "Content-Length: 1048576\r\n"
  "\r\n";

inline constexpr std::string_view head_200 =
  "HTTP/1.1 200 OK\r\n"
  "x-amz-id-2: Xr9GhZqE0QnQ1wF2c5Vx3mI1yP7uN0bK4sJ8tL6dA2hR5eC9oW3vB1gT7kM0zY4pS6nD8fH2jU5qE=\r\n"
  "x-amz-request-id: 5RZ8XK1S2V6QH9TC\r\n"
  "Date: Mon, 12 Oct 2026 17:21:04 GMT\r\n"
  "Last-Modified: Tue, 03 Mar 2026 09:14:55 GMT\r\n"
  "ETag: \"6b1f5c2b7f1a9e0d3c4b5a6978879a5c\"\r\n"
  "Accept-Ranges: bytes\r\n"
  "Content-Type: binary/octet-stream\r\n"
  "Server: AmazonS3\r\n"
  "Content-Length: 11441354\r\n"
  "\r\n";

// S3 error bodies are commonly sent chunked.
inline constexpr std::string_view resp_403 =
  "HTTP/1.1 403 Forbidden\r\n"
  "x-amz-request-id: 0B4RH6A1QX7C2M8D\r\n"
  "x-amz-id-2: q3V1n6L0m9Kc7Hj5Bf2Ds8Ap4Zw1Xy0Tv6Ru3Qe7Ws9Ot2Pi5Lk8Mj4Nh1Gf=\r\n"
  "Content-Type: application/xml\r\n"
  "Transfer-Encoding: chunked\r\n"
  "Date: Mon, 12 Oct 2026 17:21:05 GMT\r\n"
  "Server: AmazonS3\r\n"
  "\r\n"
  "e4\r\n"
  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
  "<Error><Code>AccessDenied</Code><Message>Access Denied</Message>"
  "<RequestId>0B4RH6A1QX7C2M8D</RequestId>"
  "<HostId>q3V1n6L0m9Kc7Hj5Bf2Ds8Ap4Zw1Xy0Tv6Ru3Qe7Ws9Ot2Pi5Lk8Mj4Nh1Gf=</HostId></Error>\r\n"
  "0\r\n"
  "\r\n";

inline constexpr std::string_view resp_403_skewed =
  "HTTP/1.1 403 Forbidden\r\n"
  "x-amz-request-id: 0B4RH6A1QX7C2M8E\r\n"
  "Content-Type: application/xml\r\n"
  "Date: Mon, 12 Oct 2026 17:21:05 GMT\r\n"
  "Content-Length: 141\r\n"
  "\r\n"
  "<Error><Code>RequestTimeTooSkewed</Code><Message>The difference between the request time "
  "and the current time is too large.</Message></Error>";

inline constexpr std::string_view resp_503_slowdown =
  "HTTP/1.1 503 Slow Down\r\n"
  "x-amz-request-id: 7T2K9WQ4M1D8F3ZA\r\n"
  "x-amz-id-2: Lk8Mj4Nh1Gf=\r\n"
  "Retry-After: 2\r\n"
  "Content-Type: application/xml\r\n"
  "Content-Length: 126\r\n"
  "Date: Mon, 12 Oct 2026 17:21:06 GMT\r\n"
  "Server: AmazonS3\r\n"
  "\r\n"
  "<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message>"
  "<RequestId>7T2K9WQ4M1D8F3ZA</RequestId></Error>";

inline constexpr std::string_view resp_412 =
  "HTTP/1.1 412 Precondition Failed\r\n"
  "x-amz-request-id: 1H7G3D9S0A2F4K6L\r\n"
  "Content-Type: application/xml\r\n"
  "Content-Length: 157\r\n"
  "Date: Mon, 12 Oct 2026 17:21:07 GMT\r\n"
  "Server: AmazonS3\r\n"
  "\r\n"
  "<Error><Code>PreconditionFailed</Code><Message>At least one of the pre-conditions you "
  "specified did not hold</Message><Condition>If-Match</Condition></Error>";

inline constexpr std::string_view resp_416 =
  "HTTP/1.1 416 Requested Range Not Satisfiable\r\n"
  "x-amz-request-id: 3M5N7B9V1C2X4Z6Q\r\n"
  "Content-Range: bytes */11441354\r\n"
  "Content-Type: application/xml\r\n"
  "Content-Length: 198\r\n"
  "Date: Mon, 12 Oct 2026 17:21:08 GMT\r\n"
  "Server: AmazonS3\r\n"
  "\r\n"
  "<Error><Code>InvalidRange</Code><Message>The requested range is not satisfiable</Message>"
  "<RangeRequested>bytes=11441354-11441400</RangeRequested><ActualObjectSize>11441354"
  "</ActualObjectSize></Error>";

inline constexpr std::string_view resp_404 =
  "HTTP/1.1 404 Not Found\r\n"
  "x-amz-request-id: 9P8O7I6U5Y4T3R2E\r\n"
  "x-amz-id-2: Ws9Ot2Pi5Lk8=\r\n"
  "Content-Type: application/xml\r\n"
  "Transfer-Encoding: chunked\r\n"
  "Date: Mon, 12 Oct 2026 17:21:09 GMT\r\n"
  "Server: AmazonS3\r\n"
  "\r\n"
  "96\r\n"
  "<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message>"
  "<Key>missing.bin</Key><RequestId>9P8O7I6U5Y4T3R2E</RequestId></Error>\r\n"
  "0\r\n"
  "\r\n";

}  // namespace s3_fixtures
