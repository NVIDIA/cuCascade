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

#pragma once

#include "utils/mock_test_utils.hpp"

#include <cucascade/data/data_batch.hpp>

#include <memory>

namespace cucascade {
namespace test {

inline std::shared_ptr<data_batch> make_repository_test_batch(uint64_t id)
{
  return data_batch::make(id, std::make_unique<mock_data_representation>(memory::Tier::DISK, 0));
}

}  // namespace test
}  // namespace cucascade
