/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cucascade/memory/oom_handling_policy.hpp>

namespace cucascade {
namespace memory {

void* throw_on_oom_policy::do_handle_oom([[maybe_unused]] std::size_t bytes,
                                         [[maybe_unused]] rmm::cuda_stream_view stream,
                                         std::exception_ptr eptr,
                                         [[maybe_unused]] RetryFunc retry_function)
{
  std::rethrow_exception(eptr);
}

std::string throw_on_oom_policy::get_policy_name() const noexcept { return "rethrow"; }

std::unique_ptr<oom_handling_policy> make_default_oom_policy()
{
  return std::make_unique<throw_on_oom_policy>();
}

}  // namespace memory
}  // namespace cucascade
