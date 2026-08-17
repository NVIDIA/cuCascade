/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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

#pragma once

#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/s3/list_parser.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::io::rest {

// ---------------------------------------------------------------------------
// rest_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief RESTful object-store (s3://) ioctx. Specialisation of
 *        @c templated_ioctx<rest_reactor>.
 *
 * Owns a pool of @c rest_reactor workers (round-robined by the base) that share
 * one @p authorizer.  Overrides @c create_io_object to resolve an object's size
 * via a blocking HEAD before constructing the @c rest_io_object — the static
 * reactor factory cannot do this since it needs the authorizer + a round-trip.
 */
class rest_ioctx : public templated_ioctx<rest_reactor> {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx (one context per
  /// pool: it carries the per-reactor @c config, the presigning authorizer, and
  /// the pinned bounce-staging resource — all of which must outlive this ioctx).
  /// The ioctx config is sourced from the reactors themselves — see
  /// @c templated_ioctx.
  rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::restful; }

  /// Pool-aggregated perf counters: per-reactor snapshots with totals and
  /// counts summed, maxes maxed, and ttfb the smallest non-zero reactor value.
  /// Lock-free; safe to call while the pool is running.
  [[nodiscard]] rest_perf_snapshot perf_snapshot() const noexcept;

  /// Stream a bucket's ListObjectsV2 pages under @p prefix to @p sink, one call
  /// per page (a page holds at most 1000 entries, so peak memory is one page
  /// regardless of bucket population).  @p sink returns false to stop early —
  /// no further LIST requests are issued.  @p page_size is clamped to [1,1000]
  /// (0 and >1000 mean 1000).  Throws (never truncates) on a truncated page
  /// without a continuation token, and once more than @p max_scanned entries
  /// have been scanned across pages (bounds time / request count on a prefix
  /// whose population dwarfs the caller's matches).
  void list_objects_paged(std::string_view bucket,
                          std::string_view prefix,
                          std::size_t page_size,
                          std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                          std::optional<std::size_t> max_scanned = std::nullopt);

  /// Whole-listing convenience over @c list_objects_paged: every object under
  /// @p prefix, in document order, with sizes.  Throws (never truncates) when
  /// the accumulated entries would exceed @p max_keys — a partial key set would
  /// resolve a glob to a silently incomplete table.
  [[nodiscard]] std::vector<s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt);

  /// The configured matched cap (@c config.list_max_matches) — exposed so a
  /// glob layer one level up can bound its match set without a reactor handle.
  /// Falls back to the built-in default when the pool is empty (never in
  /// practice).
  [[nodiscard]] std::size_t list_max_matches() const;

 protected:
  /// Backend hook invoked by @c ioctx::open_io_object: parse @p path
  /// (s3://bucket/key), HEAD it for the size, and build a @c rest_io_object.
  /// Throws on a non-s3 scheme or a failed HEAD.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

  /// Hinted open: @c open_hint::parquet_footer_probe resolves the size AND stashes
  /// the object's trailing bytes in one suffix-range GET, stashed on the
  /// returned io_object; every other hint falls back to the plain HEAD path above.
  std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint) override;

  /// Known-size open: the caller already learned the object's size (e.g. from a
  /// ListObjectsV2 response), so the io_object is built with ZERO network — no
  /// HEAD, no probe.
  std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size) override;

 private:
  /// Resolve @p path with a single suffix-range GET: it discovers the size and
  /// stashes the object's trailing bytes on the returned io_object so cuDF's
  /// footer reads are served locally.  Falls back to a plain HEAD (no stash)
  /// when the response is unusable.
  std::shared_ptr<io_object> create_footer_probe_object(std::string path);
};

}  // namespace cucascade::io::rest
