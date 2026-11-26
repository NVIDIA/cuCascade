/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/io/config.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <string_view>

int main()
{
  // Storing the address through a volatile forces the linker to resolve the symbol against the
  // installed library.
  bool (*volatile supports)(std::string_view) = &cucascade::io::uring::uring_reactor::supports;
  return supports == nullptr ? 1 : 0;
}
