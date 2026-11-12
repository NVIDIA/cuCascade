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

#include <memory>
#include <span>
#include <stdexcept>

namespace cucascade::detail {

row_range_result read_rows_disk(const idata_representation& /*source*/,
                                const row_range_request& /*request*/,
                                std::span<column_destination> /*dst*/,
                                ::cuda::stream_ref /*stream*/,
                                memory::reservation* /*scratch*/,
                                std::unique_ptr<row_range_reader_state>* /*state*/,
                                std::shared_ptr<idisk_io_backend> /*io_backend*/)
{
  throw std::logic_error("not implemented: read_rows_disk");
}

}  // namespace cucascade::detail
