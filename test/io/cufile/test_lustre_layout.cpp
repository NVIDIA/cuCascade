/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <catch2/catch_all.hpp>
#include <io/cufile/lustre_layout.hpp>

#ifndef CUCASCADE_HAS_LUSTRE_LAYOUT
#error "The Lustre shim must compile the optional production ioctl branch"
#endif

TEST_CASE("Lustre ioctl adapter sizes output and rejects composite or invalid stripes",
          "[cufile][engine][hardening]")
{
  auto const magic  = GENERATE(LOV_USER_MAGIC_V1, LOV_USER_MAGIC_V3, 0x0BD60BD0U);
  auto const stripe = GENERATE(0U, 4097U, 4U << 20);
  auto const status = GENERATE(0, -1);
  auto result       = cucascade::io::cufile::detail::discover_lustre_stripe(
    42, 4096, [&](int fd, unsigned long request, void* buffer) {
      CHECK(fd == 42);
      CHECK(request == LL_IOC_LOV_GETSTRIPE);
      auto* layout = static_cast<lov_user_md_v1*>(buffer);
      CHECK(layout->lmm_magic == LOV_USER_MAGIC_V3);
      CHECK(layout->lmm_stripe_count == LOV_MAX_STRIPE_COUNT);
      // Touch the last byte a maximum V3 response may copy, not just its 32-byte header.
      auto bytes = lov_user_md_size(LOV_MAX_STRIPE_COUNT, LOV_USER_MAGIC_V3);
      static_cast<std::byte*>(buffer)[bytes - 1] = std::byte{0x5a};
      layout->lmm_magic                          = magic;
      layout->lmm_stripe_size                    = stripe;
      return status;
    });
  CHECK(result == (status == 0 && magic != 0x0BD60BD0U && stripe == 4U << 20 ? stripe : 0));
}
