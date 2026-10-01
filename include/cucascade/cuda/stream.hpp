/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if !defined(CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)
#define CUCASCADE_UNDEFINE_CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
#define CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
#endif

#include <cuda/stream>

// Stable RMM still includes this deprecated header. Include it while suppression is active so
// subsequent includes are protected by its include guard, even after we restore the macro below.
#if __has_include(<cuda/stream_ref>)
#include <cuda/stream_ref>
#endif

#if defined(CUCASCADE_UNDEFINE_CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)
#undef CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
#undef CUCASCADE_UNDEFINE_CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
#endif
