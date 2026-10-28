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

#include <cucascade/io/cache/metadata_store.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace cucascade::io::cache {

namespace {

// Write `text` to `out` with the backslash and every byte outside printable
// ASCII escaped (as \\ and \xNN): an entry key may hold any bytes (the profile
// is free-form), and one entry must stay on one line of a log.
void write_escaped(std::ostream& out, std::string_view text)
{
  constexpr std::string_view digits = "0123456789abcdef";
  for (char const c : text) {
    auto const byte = static_cast<unsigned char>(c);
    if (byte == '\\') {
      out << "\\\\";
    } else if (byte >= 0x20 && byte < 0x7f) {
      out << c;
    } else {
      out << "\\x" << digits[byte >> 4U] << digits[byte & 0xfU];
    }
  }
}

}  // namespace

metadata_store::~metadata_store() { clear(); }

// ---------------------------------------------------------------------------
// Targeted invalidation
// ---------------------------------------------------------------------------

bool metadata_store::invalidate(std::string_view path, std::string_view profile)
{
  auto const lookup = cache_key(path, profile);
  entries retired;
  std::lock_guard lock{_mutex};
  auto const found = _index.find(lookup);
  if (found == _index.end()) return false;
  retire(found->second, retired);
  return true;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

void metadata_store::open_observed(io_object const& obj)
{
  auto const identity = obj.identity();
  entries retired;
  std::lock_guard lock{_mutex};
  observe_locked(obj, identity, retired);
}

void metadata_store::observe_locked(io_object const& obj,
                                    object_identity const& identity,
                                    entries& retired)
{
  if (identity.kind == identity_kind::opened_handle) return;
  auto [it, inserted] =
    _observed.try_emplace(obj.object_path(), observation{identity, obj.open_generation()});
  if (inserted) return;
  if (it->second.identity == identity) {
    it->second.generation = std::max(it->second.generation, obj.open_generation());
    return;
  }
  if (obj.open_generation() <= it->second.generation) return;
  it->second = observation{identity, obj.open_generation()};
  _stats.replacements += retire_path(obj.object_path(), retired);
}

// ---------------------------------------------------------------------------
// Lookup and publication
// ---------------------------------------------------------------------------

std::shared_ptr<io_object_metadata> metadata_store::get_or_load(io_object const& obj,
                                                                loader_type const& loader,
                                                                bool* cache_hit,
                                                                std::stop_token stop,
                                                                std::string_view profile)
{
  if (stop.stop_requested()) {
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  }
  open_observed(obj);
  auto const lookup   = cache_key(obj.object_path(), profile);
  auto const identity = obj.identity();
  // A local stat identity may be shared by separate opens while the stat tuple
  // is unchanged, so two opens of one file share a load.  The (identity, entry)
  // tuple is framed like an entry key so that no two tuples alias.
  auto const flight_key = cache_key(identity.cache_key(obj.raw_file_cache_id()), lookup);
  std::shared_ptr<flight> work;
  {
    std::unique_lock lock{_mutex};
    auto found = _index.find(lookup);
    if (found != _index.end() && found->second->identity == identity) {
      ++_stats.hits;
      if (cache_hit) *cache_hit = true;
      return found->second->value;
    }
    ++_stats.misses;
    if (cache_hit) *cache_hit = false;
    auto active = _flights.find(flight_key);
    if (active != _flights.end()) {
      work = active->second;
      ++_stats.joined;
      if (!work->ready.wait(lock, stop, [&] { return work->done; })) {
        ++_stats.cancelled_waiters;
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      }
      if (work->error) std::rethrow_exception(work->error);
      return work->value;
    }
    work = std::make_shared<flight>();
    _flights.emplace(flight_key, work);
    ++_stats.builds;
  }
  // No store lock during I/O, parsing, or publication.  Waiters hold their own
  // reference to this flight, including when it is removed after an error.
  std::shared_ptr<io_object_metadata> value;
  std::exception_ptr error;
  try {
    value = loader();
    register_metadata(obj, value, profile);
  } catch (...) {
    error = std::current_exception();
  }
  {
    std::lock_guard lock{_mutex};
    work->value = value;
    work->error = error;
    work->done  = true;
    if (error) ++_stats.failures;
    _flights.erase(flight_key);
  }
  work->ready.notify_all();
  if (error) std::rethrow_exception(error);
  return value;
}

std::string metadata_store::cache_key(std::string_view path, std::string_view profile)
{
  // Frame empty profiles too, so a raw path can never impersonate an encoded tuple.
  auto key = std::to_string(path.size());
  key.reserve(key.size() + 1 + path.size() + profile.size());
  key.push_back(':');
  key.append(path);
  key.append(profile);
  return key;
}

void metadata_store::retire(iterator it, entries& retired) noexcept
{
  --_stats.entries;
  _index.erase(it->key);
  if (auto const counted = _entries_per_path.find(it->path);
      counted != _entries_per_path.end() && --counted->second == 0) {
    _entries_per_path.erase(counted);
  }
  retired.splice(retired.end(), _entries, it);
}

std::size_t metadata_store::retire_path(std::string_view path, entries& retired) noexcept
{
  auto const counted = _entries_per_path.find(path);
  if (counted == _entries_per_path.end()) return 0;
  auto const count = counted->second;
  std::size_t done = 0;
  for (auto it = _entries.begin(); it != _entries.end() && done < count;) {
    auto current = it++;
    if (current->path != path) continue;
    retire(current, retired);
    ++done;
  }
  return done;
}

void metadata_store::clear()
{
  entries retired;
  std::lock_guard lock{_mutex};
  for (auto it = _entries.begin(); it != _entries.end();) {
    retire(it++, retired);
  }
}

bool metadata_store::put_identity(std::string const& path,
                                  object_identity identity,
                                  std::uint64_t generation,
                                  std::shared_ptr<io_object_metadata>& value,
                                  std::string_view profile,
                                  entries& retired)
{
  // A payload that is not published stays in `value`: the caller releases it
  // after _mutex, because metadata destructors may inspect their owning ioctx.
  if (!value) return false;
  if (identity.kind != identity_kind::opened_handle) {
    auto current = _observed.find(path);
    if (current != _observed.end() && current->second.identity != identity) {
      ++_stats.bypasses;
      ++_stats.stale;
      return false;
    }
  }
  auto lookup = cache_key(path, profile);
  auto found  = _index.find(lookup);
  if (found != _index.end()) {
    auto it = found->second;
    if (generation < it->generation) {
      ++_stats.bypasses;
      return false;
    }
    if (it->identity == identity) {
      // A duplicate publication keeps the first value but still advances the
      // ordering fence; otherwise an older, different-version fill could
      // replace this record.
      it->generation = std::max(it->generation, generation);
      return true;
    }
    ++_stats.replacements;
    retire(it, retired);
  }
  auto const [counted, inserted] = _entries_per_path.try_emplace(path, 0);
  try {
    // The node is allocated before anything is moved into it, and `value` is
    // the entry's last member: if building the entry throws, the payload is
    // still the caller's and is not released under _mutex.
    entries staged;
    staged.emplace_front(
      std::move(lookup), path, generation, std::move(identity), std::move(value));
    _entries.splice(_entries.begin(), staged);
    try {
      _index.emplace(_entries.front().key, _entries.begin());
    } catch (...) {
      retired.splice(retired.end(), _entries, _entries.begin());
      throw;
    }
  } catch (...) {
    if (counted->second == 0) { _entries_per_path.erase(counted); }
    throw;
  }
  ++counted->second;
  ++_stats.entries;
  return true;
}

void metadata_store::register_metadata(io_object const& obj,
                                       std::shared_ptr<io_object_metadata> metadata,
                                       std::string_view profile)
{
  if (!metadata) return;
  auto identity = obj.identity();
  // Both outlive the lock below: payloads leaving the store (retired entries,
  // and this one when it is dropped or publishing it throws) are released
  // after _mutex is unlocked.
  entries retired;
  auto payload = std::move(metadata);
  std::lock_guard lock{_mutex};
  observe_locked(obj, identity, retired);
  put_identity(
    obj.object_path(), std::move(identity), obj.open_generation(), payload, profile, retired);
}

std::shared_ptr<io_object_metadata> metadata_store::get_metadata(io_object const& obj,
                                                                 std::string_view profile) const
{
  auto const lookup   = cache_key(obj.object_path(), profile);
  auto const identity = obj.identity();
  std::lock_guard lock{_mutex};
  auto const found = _index.find(lookup);
  if (found != _index.end() && found->second->identity == identity) {
    ++_stats.hits;
    return found->second->value;
  }
  ++_stats.misses;
  return {};
}

bool metadata_store::has_path(std::string_view object_path) const
{
  auto const lookup = cache_key(object_path);
  std::lock_guard lock{_mutex};
  auto const it = _index.find(lookup);
  if (it == _index.end()) return false;
  // An opened-handle identity is scoped to one datasource and cannot justify
  // replacing a footer probe for a later open.  Strong object tags and local
  // stat identities are reusable candidates; the full identity check still
  // happens after the datasource is opened.
  return it->second->identity.kind != identity_kind::opened_handle;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

metadata_store::counters metadata_store::statistics() const
{
  std::lock_guard lock{_mutex};
  return _stats;
}

std::string metadata_store::describe_metadata_store() const
{
  std::lock_guard lock{_mutex};
  std::ostringstream out;
  out << "entries=" << _stats.entries << " hits=" << _stats.hits << " misses=" << _stats.misses
      << " joined=" << _stats.joined << " builds=" << _stats.builds
      << " failures=" << _stats.failures << " cancelled_waiters=" << _stats.cancelled_waiters
      << " stale=" << _stats.stale << " in_flight=" << _flights.size() << '\n';
  for (auto const& entry : _entries) {
    out << "key=";
    write_escaped(out, entry.key);
    out << " identity=";
    write_escaped(out, entry.identity.cache_key(""));
    out << " generation=" << entry.generation << " holders=" << entry.value.use_count() << '\n';
  }
  return out.str();
}

}  // namespace cucascade::io::cache
