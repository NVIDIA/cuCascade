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

#include "io/detail/local_descriptor.hpp"

#include <atomic>

namespace cucascade::io::detail {
namespace {

// The installed seam lives in a static slot; the pointer is what readers load,
// so the hot path costs one relaxed-acquire load when no test installed one.
reopen_hook_for_testing installed_hook{};
std::atomic<reopen_hook_for_testing const*> installed{nullptr};

}  // namespace

void set_reopen_hook_for_testing(reopen_hook_for_testing hook) noexcept
{
  installed.store(nullptr, std::memory_order_release);
  if (hook.hook == nullptr) return;
  installed_hook = hook;
  installed.store(&installed_hook, std::memory_order_release);
}

reopen_hook_for_testing const* reopen_hook_for_testing_installed() noexcept
{
  return installed.load(std::memory_order_acquire);
}

}  // namespace cucascade::io::detail
