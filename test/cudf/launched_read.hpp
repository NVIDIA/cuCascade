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

#include <future>
#include <utility>

namespace cucascade::test {

/// Get @p read on its own thread.  A datasource read's future is a deferred join, which cannot be
/// waited on with a timeout; the returned future can.
template <typename T>
std::future<T> launched(std::future<T> read)
{
  return std::async(std::launch::async, [read = std::move(read)]() mutable { return read.get(); });
}

}  // namespace cucascade::test
