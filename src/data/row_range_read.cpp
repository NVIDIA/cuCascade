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

#include <cucascade/data/row_range_read.hpp>

#include <stdexcept>

namespace cucascade {

void row_range_reader_registry::register_reader_impl(std::type_index /*type*/,
                                                     row_range_read_fn /*fn*/)
{
  throw std::logic_error("not implemented: row_range_reader_registry::register_reader_impl");
}

bool row_range_reader_registry::has_reader(const idata_representation& /*source*/) const
{
  throw std::logic_error("not implemented: row_range_reader_registry::has_reader");
}

bool row_range_reader_registry::has_reader_impl(std::type_index /*type*/) const
{
  throw std::logic_error("not implemented: row_range_reader_registry::has_reader_impl");
}

row_range_result row_range_reader_registry::read(
  const idata_representation& /*source*/,
  const row_range_request& /*request*/,
  std::span<column_destination> /*dst*/,
  ::cuda::stream_ref /*stream*/,
  memory::reservation* /*scratch*/,
  std::unique_ptr<row_range_reader_state>* /*state*/) const
{
  throw std::logic_error("not implemented: row_range_reader_registry::read");
}

row_range_result read_rows_into(const row_range_reader_registry& /*registry*/,
                                const read_only_data_batch& /*batch*/,
                                const row_range_request& /*request*/,
                                std::span<column_destination> /*dst*/,
                                ::cuda::stream_ref /*stream*/,
                                memory::reservation* /*scratch*/,
                                std::unique_ptr<row_range_reader_state>* /*state*/)
{
  throw std::logic_error("not implemented: read_rows_into");
}

}  // namespace cucascade
