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

#include "row_range_detail.hpp"

#include <stdexcept>

namespace cucascade::detail {

void copy_bits(std::uint32_t* /*dst_mask*/,
               std::size_t /*dst_bit*/,
               const std::uint32_t* /*src_mask*/,
               std::size_t /*src_bit*/,
               std::size_t /*num_bits*/,
               ::cuda::stream_ref /*stream*/)
{
  throw std::logic_error("not implemented: copy_bits");
}

void rebase_offsets(void* /*dst*/,
                    std::uint8_t /*dst_width*/,
                    const void* /*src*/,
                    std::uint8_t /*src_width*/,
                    std::size_t /*count*/,
                    std::int64_t /*dst_base*/,
                    ::cuda::stream_ref /*stream*/)
{
  throw std::logic_error("not implemented: rebase_offsets");
}

rmm::device_buffer allocate_scratch(memory::reservation* /*scratch*/,
                                    std::size_t /*bytes*/,
                                    ::cuda::stream_ref /*stream*/)
{
  throw std::logic_error("not implemented: allocate_scratch");
}

}  // namespace cucascade::detail
