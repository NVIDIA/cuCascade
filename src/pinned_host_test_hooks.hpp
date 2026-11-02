/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Production expands the CUDA expression directly and discards all test-only context arguments.
#if defined(CUCASCADE_ENABLE_PINNED_HOST_TEST_HOOKS)
#include "memory/pinned_host_test_hooks.hpp"

#define CUCASCADE_PINNED_HOST_CUDA_CALL(operation, call, ...)                                     \
  ::cucascade::test::invoke_pinned_host_cuda(::cucascade::test::pinned_host_operation::operation, \
                                             [&]() noexcept { return (call); } __VA_OPT__(, )     \
                                               __VA_ARGS__)
#define CUCASCADE_PINNED_HOST_METADATA_ALLOCATION(operation) \
  ::cucascade::test::before_pinned_host_metadata(            \
    ::cucascade::test::pinned_host_operation::operation)
#else
#define CUCASCADE_PINNED_HOST_CUDA_CALL(operation, call, ...) (call)
#define CUCASCADE_PINNED_HOST_METADATA_ALLOCATION(operation)  ((void)0)
#endif
