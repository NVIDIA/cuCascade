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

std::size_t fit_rows(std::span<const fit_column> /*columns*/,
                     std::size_t /*row_begin*/,
                     std::size_t /*max_rows*/,
                     std::size_t /*num_rows*/)
{
  throw std::logic_error("not implemented: fit_rows");
}

void validate_request(const row_range_request& /*request*/,
                      std::span<const column_destination> /*dst*/,
                      std::span<const std::int32_t> /*source_type_ids*/,
                      std::size_t /*num_rows*/)
{
  throw std::logic_error("not implemented: validate_request");
}

}  // namespace cucascade::detail
