/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>

namespace {

struct fail_if_loaded {
  fail_if_loaded() { std::_Exit(99); }
};

fail_if_loaded sentinel;

}  // namespace
