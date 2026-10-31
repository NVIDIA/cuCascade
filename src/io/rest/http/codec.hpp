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

/// @file codec.hpp
/// @brief Umbrella header for the HTTP/1.1 codec (gpuio docs/arch/API.md §A.12; no I/O).

#pragma once

#include "io/rest/http/chunked_decoder.hpp"
#include "io/rest/http/classify.hpp"
#include "io/rest/http/content_range.hpp"
#include "io/rest/http/fields.hpp"
#include "io/rest/http/iovec_sink.hpp"
#include "io/rest/http/message.hpp"
#include "io/rest/http/request_head.hpp"
#include "io/rest/http/response_parser.hpp"
