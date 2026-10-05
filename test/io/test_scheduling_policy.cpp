/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <optional>

namespace {

using cucascade::io::request_class;
using cucascade::io::detail::resource_need;
using cucascade::io::detail::scheduling_config;
using cucascade::io::detail::scheduling_policy;
using cucascade::io::detail::scheduling_view;

/// A view with @p slots slots and @p ops ops, all free.
scheduling_view make_view(std::size_t slots = 64, std::size_t ops = 128)
{
  scheduling_view view;
  view.total_slots = slots;
  view.free_slots  = slots;
  view.total_ops   = ops;
  view.free_ops    = ops;
  return view;
}

/// Account @p n in-flight operations of @p cls holding @p slots_each slots.
void occupy(scheduling_view& view, request_class cls, std::size_t n, std::size_t slots_each = 1)
{
  view[cls].ops_in_flight += n;
  view[cls].slots_in_use += n * slots_each;
  view.free_ops -= n;
  view.free_slots -= n * slots_each;
}

}  // namespace

TEST_CASE("pick returns nothing when nothing is queued", "[io][policy]")
{
  scheduling_policy policy;
  CHECK_FALSE(policy.pick(make_view()).has_value());
}

TEST_CASE("pick order is latency, read, background, write", "[io][policy]")
{
  scheduling_policy policy;
  auto view                              = make_view();
  view[request_class::write].queued      = 1;
  view[request_class::background].queued = 1;
  view[request_class::read].queued       = 1;
  view[request_class::latency].queued    = 1;
  CHECK(policy.pick(view) == request_class::latency);

  view[request_class::latency].queued = 0;
  CHECK(policy.pick(view) == request_class::read);

  view[request_class::read].queued = 0;
  CHECK(policy.pick(view) == request_class::background);

  view[request_class::background].queued = 0;
  CHECK(policy.pick(view) == request_class::write);
}

TEST_CASE("the write starvation guard overtakes other classes", "[io][policy]")
{
  scheduling_config config;
  config.write_max_wait = std::chrono::milliseconds(20);
  scheduling_policy policy(config);

  auto view                             = make_view();
  view[request_class::latency].queued   = 3;
  view[request_class::read].queued      = 3;
  view[request_class::write].queued     = 1;
  view[request_class::write].oldest_age = std::chrono::milliseconds(19);
  CHECK(policy.pick(view) == request_class::latency);

  view[request_class::write].oldest_age = std::chrono::milliseconds(20);
  CHECK(policy.pick(view) == request_class::write);

  // ...but only within the write budget.
  occupy(view, request_class::write, 32);  // 50% of 64 slots
  CHECK(policy.pick(view) == request_class::latency);
}

TEST_CASE("pick respects the group limits", "[io][policy]")
{
  scheduling_config config;
  config.max_active_groups  = 2;
  config.max_latency_groups = 1;
  scheduling_policy policy(config);

  auto view                                   = make_view();
  view[request_class::read].queued            = 5;
  view[request_class::read].active_groups     = 1;
  view[request_class::read].expanding_groups  = 1;
  view[request_class::write].active_groups    = 1;
  view[request_class::write].expanding_groups = 1;
  CHECK_FALSE(policy.pick(view).has_value());

  // Latency has its own allowance on top of the bulk groups.
  view[request_class::latency].queued = 1;
  CHECK(policy.pick(view) == request_class::latency);
  view[request_class::latency].active_groups    = 1;
  view[request_class::latency].expanding_groups = 1;
  CHECK_FALSE(policy.pick(view).has_value());

  view[request_class::write].active_groups    = 0;
  view[request_class::write].expanding_groups = 0;
  CHECK(policy.pick(view) == request_class::read);
}

TEST_CASE("groups with all operations dispatched do not count against the group limits",
          "[io][policy]")
{
  scheduling_config config;
  config.max_active_groups  = 2;
  config.max_latency_groups = 1;
  scheduling_policy policy(config);

  // Many held groups whose operations are all in flight: still room to pull.
  auto view                                  = make_view();
  view[request_class::read].queued           = 5;
  view[request_class::read].active_groups    = 10;
  view[request_class::latency].active_groups = 3;
  view[request_class::latency].queued        = 1;
  CHECK(policy.pick(view) == request_class::latency);
  view[request_class::latency].queued = 0;
  CHECK(policy.pick(view) == request_class::read);

  // Two of them still expanding: the bulk limit applies again.
  view[request_class::read].expanding_groups = 2;
  CHECK_FALSE(policy.pick(view).has_value());
  view[request_class::latency].queued           = 1;
  view[request_class::latency].expanding_groups = 1;
  CHECK_FALSE(policy.pick(view).has_value());

  // Held (in-flight) latency groups still keep the latency reservation armed.
  view[request_class::latency].queued           = 0;
  view[request_class::latency].expanding_groups = 0;
  CHECK(scheduling_policy::latency_pressure(view));
}

TEST_CASE("background never exceeds its share", "[io][policy]")
{
  scheduling_policy policy;  // background share 75%
  auto view                              = make_view(64, 128);
  view[request_class::background].queued = 1;
  occupy(view, request_class::background, 48);  // 75% of 64
  view.total_ops = 0;                           // ops axis unconstrained

  // No foreground work at all: the share still applies.
  CHECK_FALSE(scheduling_policy::foreground_pressure(view));
  CHECK_FALSE(policy.pick(view).has_value());
  CHECK_FALSE(policy.may_dispatch(request_class::background, resource_need{1, 1}, view));
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));

  // One below the share: background may be pulled and dispatched.
  view[request_class::background].ops_in_flight = 47;
  view[request_class::background].slots_in_use  = 47;
  view.free_slots                               = 64 - 47;
  CHECK(policy.pick(view) == request_class::background);
  CHECK(policy.may_dispatch(request_class::background, resource_need{1, 1}, view));
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));

  // Under foreground pressure the share is unchanged.
  view[request_class::read].active_groups = 1;
  CHECK(policy.may_dispatch(request_class::background, resource_need{1, 1}, view));
  occupy(view, request_class::background, 1);  // 48 of 64
  CHECK_FALSE(policy.may_dispatch(request_class::background, resource_need{1, 1}, view));
}

TEST_CASE("background groups are capped by max_background_groups", "[io][policy]")
{
  scheduling_policy policy;  // 4 active groups, 2 of them background
  auto view                                        = make_view();
  view[request_class::background].queued           = 1;
  view[request_class::background].active_groups    = 2;
  view[request_class::background].expanding_groups = 2;  // bulk room left: 2 < 4
  CHECK_FALSE(policy.pick(view).has_value());

  // The sub-limit counts inside max_active_groups: a read can still be pulled.
  view[request_class::read].queued = 1;
  CHECK(policy.pick(view) == request_class::read);

  view[request_class::read].queued = 0;
  scheduling_config config;
  config.max_background_groups = 3;
  CHECK(scheduling_policy(config).pick(view) == request_class::background);

  // ...and never above it.
  view[request_class::read].active_groups    = 2;
  view[request_class::read].expanding_groups = 2;
  CHECK_FALSE(scheduling_policy(config).pick(view).has_value());
}

TEST_CASE("max_background_groups above max_active_groups behaves as max_active_groups",
          "[io][policy]")
{
  scheduling_config config;
  config.max_active_groups     = 1;
  config.max_background_groups = 2;  // effective limit: 1
  scheduling_policy policy(config);

  // No background group expanding and bulk room left: background is pickable.
  auto view                              = make_view();
  view[request_class::background].queued = 1;
  CHECK(policy.pick(view) == request_class::background);

  // One expanding: the clamped sub-limit (and the bulk limit) is reached.
  view[request_class::background].active_groups    = 1;
  view[request_class::background].expanding_groups = 1;
  CHECK_FALSE(policy.pick(view).has_value());
}

TEST_CASE("reads keep a floor of slots free while background work is expanding", "[io][policy]")
{
  scheduling_policy policy;  // reserve min(8, 64 / 4, 48) = 8
  auto view                                        = make_view(64, 0);
  view[request_class::background].active_groups    = 1;
  view[request_class::background].expanding_groups = 1;
  view[request_class::background].ops_in_flight    = 10;
  view[request_class::background].slots_in_use     = 10;
  view.free_slots                                  = 8;
  REQUIRE(policy.background_pressure(view));
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));
  CHECK_FALSE(policy.may_dispatch(request_class::write, resource_need{1, 1}, view));
  CHECK(policy.may_dispatch(request_class::latency, resource_need{1, 1}, view));
  CHECK(policy.may_dispatch(request_class::background, resource_need{1, 1}, view));

  view.free_slots = 9;
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{2, 1}, view));

  // Background at its share cannot use the floor: no reservation.
  view.free_slots                               = 8;
  view[request_class::background].ops_in_flight = 48;
  view[request_class::background].slots_in_use  = 48;
  CHECK_FALSE(policy.background_pressure(view));
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));

  // Background with nothing left to dispatch needs no reservation either.
  view[request_class::background].ops_in_flight    = 10;
  view[request_class::background].slots_in_use     = 10;
  view[request_class::background].expanding_groups = 0;
  CHECK_FALSE(policy.background_pressure(view));
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));

  // A zero reservation disables the floor.
  view[request_class::background].expanding_groups = 1;
  scheduling_config config;
  config.reserved_background_slots = 0;
  CHECK(scheduling_policy(config).may_dispatch(request_class::read, resource_need{1, 1}, view));
}

TEST_CASE("only refusals caused by the background floor are reported as such", "[io][policy]")
{
  scheduling_policy policy;  // floor min(8, 64 / 4, 48) = 8, write share 32 slots
  auto view = make_view(64, 0);
  occupy(view, request_class::background, 10);  // below its share: pressure
  view[request_class::background].active_groups    = 1;
  view[request_class::background].expanding_groups = 1;
  view.free_slots                                  = 20;
  REQUIRE(policy.background_pressure(view));

  // A 16-slot read fits (20 free) but would leave 4 < 8 free: the floor.
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{16, 1}, view));
  CHECK(policy.refused_by_background_floor(request_class::read, resource_need{16, 1}, view));
  // Allowed, physical misfit, or a class the floor does not apply to: not reported.
  CHECK_FALSE(policy.refused_by_background_floor(request_class::read, resource_need{12, 1}, view));
  CHECK_FALSE(policy.refused_by_background_floor(request_class::read, resource_need{24, 1}, view));
  CHECK_FALSE(
    policy.refused_by_background_floor(request_class::background, resource_need{16, 1}, view));
  CHECK_FALSE(
    policy.refused_by_background_floor(request_class::latency, resource_need{16, 1}, view));

  // A write at its share is refused by the share, floor or not.
  auto writes = view;
  occupy(writes, request_class::write, 2, 16);  // 32 slots = the write share
  writes.free_slots = 20;
  CHECK_FALSE(policy.may_dispatch(request_class::write, resource_need{1, 1}, writes));
  CHECK_FALSE(
    policy.refused_by_background_floor(request_class::write, resource_need{1, 1}, writes));
  CHECK_FALSE(
    policy.refused_by_background_floor(request_class::write, resource_need{16, 1}, writes));
  // ...while a read in the same state is held back by the floor alone.
  CHECK(policy.refused_by_background_floor(request_class::read, resource_need{16, 1}, writes));

  // Refused by the latency reservation: not reported, with or without the floor.
  auto latency                           = view;
  latency[request_class::latency].queued = 1;
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{19, 1}, latency));
  CHECK_FALSE(
    policy.refused_by_background_floor(request_class::read, resource_need{19, 1}, latency));
  latency[request_class::background].expanding_groups = 0;  // no floor
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{19, 1}, latency));
  CHECK_FALSE(
    policy.refused_by_background_floor(request_class::read, resource_need{19, 1}, latency));
}

TEST_CASE("the background floor is clamped to a quarter of the axis and to the share",
          "[io][policy]")
{
  // 16 slots: min(8, 16 / 4, share_of(16, 0.75) = 12) = 4.
  scheduling_policy policy;
  CHECK(policy.background_reserve(16) == 4);
  CHECK(policy.background_reserve(64) == 8);
  CHECK(policy.background_reserve(0) == 0);
  auto view                                        = make_view(16, 0);
  view[request_class::background].active_groups    = 1;
  view[request_class::background].expanding_groups = 1;
  view[request_class::background].ops_in_flight    = 2;
  view[request_class::background].slots_in_use     = 2;
  view.free_slots                                  = 4;
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));
  view.free_slots = 5;
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));

  // 64 slots, share 10%: min(8, 16, share_of(64, 0.1) = 6) = 6.
  scheduling_config config;
  config.background_slot_fraction = 0.1;
  scheduling_policy narrow(config);
  CHECK(narrow.background_reserve(64) == 6);
  auto wide                                        = make_view(64, 0);
  wide[request_class::background].active_groups    = 1;
  wide[request_class::background].expanding_groups = 1;
  wide[request_class::background].ops_in_flight    = 2;
  wide[request_class::background].slots_in_use     = 2;
  wide.free_slots                                  = 6;
  REQUIRE(narrow.background_pressure(wide));
  CHECK_FALSE(narrow.may_dispatch(request_class::read, resource_need{1, 1}, wide));
  wide.free_slots = 7;
  CHECK(narrow.may_dispatch(request_class::read, resource_need{1, 1}, wide));

  // The ops axis is reserved the same way: min(8, 32 / 4, share_of(32, 0.75) = 24) = 8.
  auto ops_view                                        = make_view(0, 32);
  ops_view[request_class::background].active_groups    = 1;
  ops_view[request_class::background].expanding_groups = 1;
  ops_view[request_class::background].ops_in_flight    = 2;
  ops_view.free_ops                                    = 8;
  CHECK_FALSE(policy.may_dispatch(request_class::write, resource_need{0, 1}, ops_view));
  ops_view.free_ops = 9;
  CHECK(policy.may_dispatch(request_class::write, resource_need{0, 1}, ops_view));
}

TEST_CASE("writes never exceed their share of slots and ops", "[io][policy]")
{
  scheduling_policy policy;  // 50% / 50%
  auto view = make_view(64, 128);

  CHECK(policy.may_dispatch(request_class::write, resource_need{16, 1}, view));
  occupy(view, request_class::write, 2, 16);  // 32 of 64 slots
  CHECK_FALSE(policy.may_dispatch(request_class::write, resource_need{1, 1}, view));
  CHECK(policy.may_dispatch(request_class::read, resource_need{16, 1}, view));

  // Ops axis: host-direct writes need no slots but still count against the ring share.
  auto ops_view = make_view(64, 8);
  occupy(ops_view, request_class::write, 4, 0);
  CHECK_FALSE(policy.may_dispatch(request_class::write, resource_need{0, 1}, ops_view));
  CHECK(policy.may_dispatch(request_class::read, resource_need{0, 1}, ops_view));
}

TEST_CASE("an operation larger than a share is admitted when its class is idle", "[io][policy]")
{
  scheduling_policy policy;
  auto view = make_view(8, 16);
  CHECK(policy.may_dispatch(request_class::write, resource_need{6, 1}, view));  // > 50% of 8
  occupy(view, request_class::write, 1, 6);
  CHECK_FALSE(policy.may_dispatch(request_class::write, resource_need{1, 1}, view));
}

TEST_CASE("physical capacity is always enforced", "[io][policy]")
{
  scheduling_policy policy;
  auto view = make_view(4, 4);
  occupy(view, request_class::read, 3, 1);
  CHECK_FALSE(policy.may_dispatch(request_class::latency, resource_need{2, 1}, view));
  CHECK(policy.may_dispatch(request_class::latency, resource_need{1, 1}, view));

  // An unconstrained axis (total 0) never blocks.
  scheduling_view open;
  CHECK(policy.may_dispatch(request_class::write, resource_need{1000, 1000}, open));
}

TEST_CASE("slots are reserved for latency work while it is pending", "[io][policy]")
{
  scheduling_config config;
  config.reserved_latency_slots = 2;
  scheduling_policy policy(config);

  auto view = make_view(16, 128);
  occupy(view, request_class::read, 13, 1);  // 3 free
  view.total_ops = 0;
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));

  view[request_class::latency].queued = 1;
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));  // leaves 2
  occupy(view, request_class::read, 1, 1);                                     // 2 free
  CHECK_FALSE(policy.may_dispatch(request_class::read, resource_need{1, 1}, view));
  CHECK_FALSE(policy.may_dispatch(request_class::background, resource_need{1, 1}, view));
  CHECK(policy.may_dispatch(request_class::latency, resource_need{1, 1}, view));

  // The reservation is clamped to half the capacity.
  auto tiny                           = make_view(2, 0);
  tiny.total_ops                      = 0;
  tiny[request_class::latency].queued = 1;
  CHECK(policy.may_dispatch(request_class::read, resource_need{1, 1}, tiny));
}

TEST_CASE("share_of floors and keeps at least one", "[io][policy]")
{
  CHECK(scheduling_policy::share_of(64, 0.5) == 32);
  CHECK(scheduling_policy::share_of(3, 0.5) == 1);
  CHECK(scheduling_policy::share_of(1, 0.1) == 1);
  CHECK(scheduling_policy::share_of(0, 0.5) == 0);
  CHECK(scheduling_policy::share_of(10, 2.0) == 10);
}
