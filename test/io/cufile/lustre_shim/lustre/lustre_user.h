/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Minimal external UAPI shim: compile/exercise the optional production ioctl adapter
// even on hosts without Lustre development headers. Not a substitute for client qualification.
#include <cstddef>
#include <cstdint>

#define LOV_USER_MAGIC_V1    0x0BD10BD0U
#define LOV_USER_MAGIC_V3    0x0BD30BD0U
#define LOV_MAX_STRIPE_COUNT 2000
#define LL_IOC_LOV_GETSTRIPE 0x8008669bUL

struct lov_user_md_v1 {
  std::uint32_t lmm_magic;
  std::uint32_t lmm_pattern;
  std::uint64_t lmm_object_id;
  std::uint64_t lmm_object_seq;
  std::uint32_t lmm_stripe_size;
  std::uint16_t lmm_stripe_count;
  std::uint16_t lmm_stripe_offset;
};
inline std::size_t lov_user_md_size(unsigned stripes, unsigned magic)
{
  return sizeof(lov_user_md_v1) + (magic == LOV_USER_MAGIC_V3 ? 16 : 0) + stripes * 24;
}
