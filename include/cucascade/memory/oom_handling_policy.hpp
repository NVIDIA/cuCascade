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

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <string>

namespace cucascade {
namespace memory {

struct oom_handling_policy {
  virtual ~oom_handling_policy() = default;

  using RetryFunc = std::function<void*(std::size_t, rmm::cuda_stream_view)>;

  void* handle_oom(std::size_t bytes,
                   rmm::cuda_stream_view stream,
                   std::exception_ptr eptr,
                   RetryFunc retry_function)
  {
    return do_handle_oom(bytes, stream, eptr, std::move(retry_function));
  }

  virtual std::string get_policy_name() const noexcept = 0;

 protected:
  virtual void* do_handle_oom(std::size_t bytes,
                              rmm::cuda_stream_view stream,
                              std::exception_ptr eptr,
                              RetryFunc retry_function) = 0;
};

struct throw_on_oom_policy final : public oom_handling_policy {
 protected:
  void* do_handle_oom(std::size_t bytes,
                      rmm::cuda_stream_view stream,
                      std::exception_ptr eptr,
                      RetryFunc retry_function) final;

  std::string get_policy_name() const noexcept override;
};

std::unique_ptr<oom_handling_policy> make_default_oom_policy();

}  // namespace memory
}  // namespace cucascade
