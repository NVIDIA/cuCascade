/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/io/config.hpp>

#include <array>
#include <cstddef>

std::array<std::size_t, 6> cufile_config_abi_reference()
{
  using namespace cucascade::io::cufile;
  return cufile_config_library_abi_reference();
}
