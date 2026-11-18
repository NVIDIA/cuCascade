/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sys/ioctl.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#if __has_include(<lustre/lustre_user.h>)
#include <lustre/lustre_user.h>
#define CUCASCADE_HAS_LUSTRE_LAYOUT 1
#elif __has_include(<linux/lustre/lustre_user.h>)
#include <linux/lustre/lustre_user.h>
#define CUCASCADE_HAS_LUSTRE_LAYOUT 1
#endif

namespace cucascade::io::cufile::detail {
#ifdef CUCASCADE_HAS_LUSTRE_LAYOUT
// Internal linkage lets the shim test compile this exact optional path independently.
template <typename Ioctl>
static std::size_t discover_lustre_stripe(int fd, std::size_t granule, Ioctl ioctl_call)
{
  auto const bytes =
    static_cast<std::size_t>(lov_user_md_size(LOV_MAX_STRIPE_COUNT, LOV_USER_MAGIC_V3));
  std::vector<std::uint64_t> buffer((bytes + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t));
  auto* layout             = reinterpret_cast<lov_user_md_v1*>(buffer.data());
  layout->lmm_magic        = LOV_USER_MAGIC_V3;
  layout->lmm_stripe_count = LOV_MAX_STRIPE_COUNT;
  if (ioctl_call(fd, LL_IOC_LOV_GETSTRIPE, buffer.data()) != 0) return 0;
  // PFL/composite/DoM layouts are not a V1/V3 stripe header.
  if (layout->lmm_magic != LOV_USER_MAGIC_V1 && layout->lmm_magic != LOV_USER_MAGIC_V3) return 0;
  auto const stripe = static_cast<std::size_t>(layout->lmm_stripe_size);
  return stripe && granule && stripe % granule == 0 ? stripe : 0;
}
#endif
}  // namespace cucascade::io::cufile::detail
