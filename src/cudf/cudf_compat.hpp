/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/null_mask.hpp>
#include <cudf/version_config.hpp>

#define CUCASCADE_CUDF_NEW_NULL_MASK \
  (CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 12))

#if CUCASCADE_CUDF_NEW_NULL_MASK
#include <rapids/cuda/buffer>
#else
#include <rmm/device_buffer.hpp>
#endif

#include <cstddef>

/**
 * @brief Alias for the buffer type cudf uses to hold a column's null mask.
 *
 * cudf 26.12 changed null masks from `rmm::device_buffer` to
 * `cuda::device_buffer<cudf::bitmask_type>`. `CUCASCADE_CUDF_NEW_NULL_MASK` is
 * set from `CUDF_VERSION_MAJOR` and `CUDF_VERSION_MINOR` and selects the
 * matching type.
 *
 * TODO(https://github.com/rapidsai/build-planning/issues/321): drop this header
 * once the minimum supported cudf is 26.12.
 */
namespace cucascade::cudf_compat {
#if CUCASCADE_CUDF_NEW_NULL_MASK
using null_mask_buffer       = ::cuda::device_buffer<cudf::bitmask_type>;
using null_mask_size_element = cudf::bitmask_type;
#else
using null_mask_buffer       = rmm::device_buffer;
using null_mask_size_element = std::byte;
#endif
/** @brief Return the null-mask allocation size in bytes across cudf buffer types. */
[[nodiscard]] inline std::size_t get_null_mask_size_bytes(null_mask_buffer const& mask) noexcept
{
  return mask.size() * sizeof(null_mask_size_element);
}
}  // namespace cucascade::cudf_compat
