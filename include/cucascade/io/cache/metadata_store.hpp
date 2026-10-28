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

#include <cucascade/io/types.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cucascade::io::cache {

namespace detail {

/// Transparent hasher so the store can be looked up by @c std::string_view (or
/// @c const char*) without materialising a @c std::string.  Paired with
/// @c std::equal_to<> below, this enables C++20 heterogeneous lookup on the
/// underlying @c unordered_map — without both, a string_view-taking getter
/// would just construct a temporary key on every call and be strictly worse
/// than taking @c std::string const&.
struct string_hash {
  using is_transparent = void;
  [[nodiscard]] std::size_t operator()(std::string_view sv) const noexcept
  {
    return std::hash<std::string_view>{}(sv);
  }
};

}  // namespace detail

/**
 * @brief Thread-safe per-ioctx metadata cache: one entry per object path and
 *        profile, reusable only by an open with the same object identity.
 *
 * Owned by @c ioctx and always present, independent of the @c fs_cache.
 * Callers that have parsed file metadata (e.g. a parquet footer) park it here
 * so a later scan of the same object can skip the parse.  A profile is an
 * independent cache dimension for the same object (for example metadata
 * derived for one access pattern); the empty profile is the canonical one.
 * Path and profile are encoded unambiguously, so no (path, profile) pair can
 * alias another.
 *
 * An entry records the @c io_object::identity() of the open that published it
 * and is returned only to an open with an equal identity: a strong ETag or a
 * local file's stat tuple can be shared by separate opens, an open-local
 * (@c opened_handle) identity only by the publishing object itself.  A path
 * lookup (@ref has_path) is a candidate hint, never permission to consume
 * metadata.
 *
 * The store remembers the newest identity observed for each path
 * (@ref open_observed, by open generation).  Observing a newer identity retires
 * the path's entries (every profile), and a publication from an open of any
 * other identity is dropped, so a slow parse of a replaced version cannot
 * overwrite the current one.
 *
 * @ref get_or_load is single-flight: concurrent misses for one identity share
 * one load; a failed load is not cached and the next miss retries; a waiter can
 * cancel its wait without cancelling the shared load.  Same-identity
 * re-registration keeps the first published value.  Payloads that leave the
 * store are released outside its lock.
 */
class metadata_store {
 public:
  /// Cumulative counters (@ref statistics); @c entries is the current size.
  struct counters {
    std::size_t entries = 0;
    std::uint64_t hits = 0, misses = 0;
    std::uint64_t replacements = 0, bypasses = 0;
    std::uint64_t joined = 0, builds = 0, failures = 0, cancelled_waiters = 0, stale = 0;
  };

  using loader_type = std::function<std::shared_ptr<io_object_metadata>()>;

  metadata_store() = default;
  ~metadata_store();
  metadata_store(metadata_store const&)            = delete;
  metadata_store& operator=(metadata_store const&) = delete;
  metadata_store(metadata_store&&)                 = delete;
  metadata_store& operator=(metadata_store&&)      = delete;

  /// Record @p obj's identity as observed for its path.  A newer open (by
  /// @c io_object::open_generation) with a different identity retires the
  /// path's entries of every profile; an older one changes nothing.
  /// @c opened_handle identities are not evidence and are ignored.  The
  /// datasource calls this when it is constructed.
  void open_observed(io_object const& obj);

  /// The metadata registered for @p obj's identity under @p profile or, on a
  /// miss, the result of @p loader, registered for it (see
  /// @ref register_metadata).  Concurrent misses for the same identity and
  /// profile share one call of @p loader; a loader failure is rethrown to
  /// every caller sharing it and is not cached.  @p cache_hit, when not null,
  /// is set to whether the value came from the store.
  /// @throws std::system_error (@c operation_canceled) when @p stop is
  ///         requested before the call or while it waits for another caller's
  ///         load; that load continues.
  std::shared_ptr<io_object_metadata> get_or_load(io_object const& obj,
                                                  loader_type const& loader,
                                                  bool* cache_hit          = nullptr,
                                                  std::stop_token stop     = {},
                                                  std::string_view profile = {});

  /// Record @p metadata for @p obj's path, @p profile and identity, replacing
  /// the entry of another identity when @p obj is the newer open.  A
  /// registration for an identity that already has an entry keeps the existing
  /// value.  Ignored (silently) when @p metadata is null, and when @p obj is
  /// older than the path's entry or its observed identity differs.
  void register_metadata(io_object const& obj,
                         std::shared_ptr<io_object_metadata> metadata,
                         std::string_view profile = {});

  /// The metadata registered for @p obj's path and @p profile if it was
  /// published for an equal identity; nullptr otherwise.
  [[nodiscard]] std::shared_ptr<io_object_metadata> get_metadata(
    io_object const& obj, std::string_view profile = {}) const;

  /// True when @p object_path has a canonical-profile entry that a later open
  /// can reuse: its identity is not @c opened_handle.  A hint for choosing how
  /// to open the object (a known footer needs no probe), never a source of
  /// metadata: only @ref get_metadata / @ref get_or_load with the opened
  /// object are.
  [[nodiscard]] bool has_path(std::string_view object_path) const;

  /// Retire the entry of exactly @p path and @p profile, leaving the path's
  /// other profiles and its observed identity alone (for example to rebuild
  /// metadata found inconsistent with the opened object).
  /// Holders of the retired value keep it until they release it.  Returns
  /// whether there was such an entry.
  bool invalidate(std::string_view path, std::string_view profile = {});

  /// Drop all cached metadata (for example after an opened-object identity
  /// mismatch).  Observed identities are kept.
  void clear();

  [[nodiscard]] counters statistics() const;

  /// One summary line of the counters, then one line per entry.  Bytes of a key
  /// or identity outside printable ASCII (a profile may hold any) and the
  /// backslash are escaped (@c \\xNN, @c \\\\), so an entry stays on its line.
  [[nodiscard]] std::string describe_metadata_store() const;

 private:
  template <typename V>
  using string_map = std::unordered_map<std::string, V, detail::string_hash, std::equal_to<>>;

  struct entry {
    std::string key;
    std::string path;
    std::uint64_t generation;
    object_identity identity;
    /// Last, so a member that throws while an entry is built leaves the
    /// payload with the caller (see @ref put_identity).
    std::shared_ptr<io_object_metadata> value;
  };
  using entries  = std::list<entry>;
  using iterator = entries::iterator;

  struct observation {
    object_identity identity;
    std::uint64_t generation;
  };

  struct flight {
    std::condition_variable_any ready;
    bool done = false;
    std::shared_ptr<io_object_metadata> value;
    std::exception_ptr error;
  };

  /// The entry key of (@p path, @p profile): the decimal length of @p path
  /// frames its exact bytes, the rest is the profile.
  static std::string cache_key(std::string_view path, std::string_view profile = {});

  /// Observe @p identity, computed by the caller from @p obj before it took
  /// @c _mutex (a backend's @c identity() runs outside the store lock).
  void observe_locked(io_object const& obj, object_identity const& identity, entries& retired);
  void retire(iterator it, entries& retired) noexcept;
  /// Retire every entry of @p path, of any profile; returns how many.
  std::size_t retire_path(std::string_view path, entries& retired) noexcept;
  /// Publish @p value for (@p path, @p profile).  @p value is moved into the
  /// store only when it is published; when it is not, or when publishing
  /// throws, it stays with the caller, which releases it after @c _mutex.
  bool put_identity(std::string const& path,
                    object_identity identity,
                    std::uint64_t generation,
                    std::shared_ptr<io_object_metadata>& value,
                    std::string_view profile,
                    entries& retired);

  mutable std::mutex _mutex;
  entries _entries;
  string_map<iterator> _index;
  /// How many entries each path has, over all profiles: lets a path without
  /// entries skip the scan of @c _entries when a newer identity retires them.
  string_map<std::size_t> _entries_per_path;
  mutable counters _stats;
  string_map<observation> _observed;
  std::unordered_map<std::string, std::shared_ptr<flight>> _flights;
};

}  // namespace cucascade::io::cache
