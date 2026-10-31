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

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace cucascade::io;
using cucascade::io::cache::metadata_store;

namespace {

struct value : io_object_metadata {
  explicit value(std::size_t bytes) : bytes(bytes) {}
  std::size_t bytes;
};

/// An io_object with a fixed identity; the store reads only its path, locator,
/// identity and open generation.
class fake_object final : public io_object {
 public:
  fake_object(std::string path, std::string key, object_identity identity)
    : path_(std::move(path)), key_(std::move(key)), identity_(std::move(identity))
  {
  }
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return key_; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return path_; }
  [[nodiscard]] size_t size() const noexcept override { return 1; }
  [[nodiscard]] object_identity identity() const override { return identity_; }

 private:
  std::string path_;
  std::string key_;
  object_identity identity_;
};

/// An object with the base identity, a strong validation tag, and a writable flag.
class tagged_object final : public io_object {
 public:
  explicit tagged_object(std::string path, std::string tag = {}, bool writable = false)
    : path_(std::move(path)), tag_(std::move(tag)), writable_(writable)
  {
  }
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return path_; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return path_; }
  [[nodiscard]] size_t size() const noexcept override { return 64; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return tag_; }
  [[nodiscard]] bool validation_tag_is_strong() const noexcept override { return true; }
  [[nodiscard]] bool writable() const noexcept override { return writable_; }

 private:
  std::string path_;
  std::string tag_;
  bool writable_;
};

object_identity tag(std::string version = "v1")
{
  return {identity_kind::object_tag, 32, std::move(version)};
}

object_identity local(std::uint64_t inode = 2)
{
  object_identity identity{identity_kind::local_stat, 256};
  identity.device   = 1;
  identity.inode    = inode;
  identity.ctime_ns = 3;
  return identity;
}

bool wait_for_joined(metadata_store const& store, std::uint64_t count)
{
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (store.statistics().joined < count && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return store.statistics().joined >= count;
}

std::size_t bytes_of(std::shared_ptr<io_object_metadata> const& metadata)
{
  REQUIRE(metadata);
  return std::static_pointer_cast<value>(metadata)->bytes;
}

/// A path that shares @p path's write epoch (the store keeps its epochs in a
/// fixed table indexed by path hash), found by trying candidates until a write
/// of one makes a registration for @p path stale.  A write of it advances
/// @p path's epoch but leaves @p path's entries in place: the state between
/// invalidate_written's epoch bump and its erase.
std::string epoch_neighbour(std::string const& path)
{
  for (int i = 0; i < (1 << 16); ++i) {
    auto candidate = path + ".neighbour" + std::to_string(i);
    metadata_store probe;
    fake_object object(path, path, local());
    probe.stamp(object);
    probe.invalidate_written(candidate);
    probe.register_metadata(object, std::make_shared<value>(0));
    if (probe.statistics().stale == 1) { return candidate; }
  }
  FAIL("no candidate path shares a write epoch with " << path);
  return {};
}

}  // namespace

// ===========================================================================
// Registration and lookup
// ===========================================================================

TEST_CASE("metadata_store returns a registration to opens of the same identity",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_object const object("s3://bucket/key.parquet", "locator", tag());
  auto const metadata = std::make_shared<value>(7);
  store.register_metadata(object, metadata);
  CHECK(store.get_metadata(object) == metadata);

  // A separate open with an equal identity reuses it; one of another identity
  // or another path does not.
  CHECK(store.get_metadata(fake_object("s3://bucket/key.parquet", "locator", tag())) == metadata);
  CHECK_FALSE(store.get_metadata(fake_object("s3://bucket/key.parquet", "locator", tag("v2"))));
  CHECK_FALSE(store.get_metadata(fake_object("s3://bucket/other.parquet", "locator", tag())));
  CHECK_FALSE(metadata_store{}.get_metadata(object));
}

TEST_CASE("metadata_store keeps the first registration of an identity and ignores null metadata",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_object const object("k", "k", tag());
  auto const first = std::make_shared<value>(1);
  store.register_metadata(object, first);
  store.register_metadata(object, std::make_shared<value>(2));
  store.register_metadata(fake_object("k", "k", tag()), std::make_shared<value>(3));
  CHECK(store.get_metadata(object) == first);

  // A null registration is silently ignored: it must not erase the entry.
  store.register_metadata(object, nullptr);
  CHECK(store.get_metadata(object) == first);
  CHECK(store.statistics().entries == 1);
}

TEST_CASE("metadata_store replaces the previous version of a path on register",
          "[cache][metadata_store][cache_identity]")
{
  metadata_store store;
  tagged_object const first{"s3://bucket/t.parquet", "\"v1\""};
  tagged_object const second{"s3://bucket/t.parquet", "\"v2\""};
  REQUIRE(first.identity_cache_key() != second.identity_cache_key());

  store.register_metadata(first, std::make_shared<value>(1));
  REQUIRE(store.get_metadata(first));
  store.register_metadata(second, std::make_shared<value>(2));

  // The first version's entry is gone; only the second is reachable.
  CHECK(store.get_metadata(first) == nullptr);
  CHECK(bytes_of(store.get_metadata(second)) == 2);
  CHECK(store.statistics().entries == 1);
  CHECK(store.statistics().replacements == 1);
}

TEST_CASE("metadata_store keeps only the newest of several versions",
          "[cache][metadata_store][cache_identity]")
{
  metadata_store store;
  std::vector<std::unique_ptr<tagged_object>> objects;
  for (int i = 0; i < 5; ++i) {
    objects.push_back(
      std::make_unique<tagged_object>("s3://bucket/t.parquet", "\"v" + std::to_string(i) + "\""));
    store.register_metadata(*objects.back(), std::make_shared<value>(static_cast<std::size_t>(i)));
  }
  for (std::size_t i = 0; i + 1 < objects.size(); ++i) {
    CHECK(store.get_metadata(*objects[i]) == nullptr);
  }
  CHECK(bytes_of(store.get_metadata(*objects.back())) == 4);
  CHECK(store.statistics().entries == 1);
}

TEST_CASE("metadata_store versions of different paths do not interact",
          "[cache][metadata_store][cache_identity]")
{
  metadata_store store;
  tagged_object const a{"s3://bucket/a.parquet", "\"v1\""};
  tagged_object const b{"s3://bucket/b.parquet", "\"v1\""};
  store.register_metadata(a, std::make_shared<value>(1));
  store.register_metadata(b, std::make_shared<value>(2));

  tagged_object const a2{"s3://bucket/a.parquet", "\"v2\""};
  store.register_metadata(a2, std::make_shared<value>(3));

  CHECK(store.get_metadata(a) == nullptr);
  CHECK(bytes_of(store.get_metadata(a2)) == 3);
  CHECK(bytes_of(store.get_metadata(b)) == 2);
}

TEST_CASE("metadata_store has_path is a reusable-identity hint for an exact path",
          "[cache][metadata_store][cache_identity]")
{
  metadata_store store;
  CHECK_FALSE(store.has_path("/data/lineitem.parquet"));
  store.register_metadata(fake_object("/data/lineitem.parquet", "/data/lineitem.parquet", local()),
                          std::make_shared<value>(1));

  // Every spelling of the path reaches the entry: a string_view, a string
  // literal, an owning std::string, and a view that is a slice of a longer,
  // not NUL-terminated buffer.
  CHECK(store.has_path("/data/lineitem.parquet"));
  CHECK(store.has_path(std::string_view{"/data/lineitem.parquet"}));
  CHECK(store.has_path(std::string{"/data/lineitem.parquet"}));
  std::string const haystack = "/data/lineitem.parquet.bak";
  CHECK(store.has_path(std::string_view{haystack}.substr(0, 22)));
  // Exact path only.
  CHECK_FALSE(store.has_path(haystack));
  CHECK_FALSE(store.has_path("/data/lineitem"));

  // A null registration records nothing for a new path.
  store.register_metadata(tagged_object{"s3://bucket/null.parquet", "\"v1\""}, nullptr);
  CHECK_FALSE(store.has_path("s3://bucket/null.parquet"));

  // Replacing the version keeps the path known.
  tagged_object const v1{"s3://bucket/t.parquet", "\"v1\""};
  tagged_object const v2{"s3://bucket/t.parquet", "\"v2\""};
  store.register_metadata(v1, std::make_shared<value>(1));
  store.register_metadata(v2, std::make_shared<value>(2));
  CHECK(store.has_path("s3://bucket/t.parquet"));
}

TEST_CASE("Metadata entries are scoped to the backend locator",
          "[cache][metadata_store][object_identity]")
{
  metadata_store store;
  fake_object a("shared", "locator-A", tag("\"same-tag\""));
  fake_object b("shared", "locator-B", tag("\"same-tag\""));
  auto pa = std::make_shared<value>(1);
  auto pb = std::make_shared<value>(2);
  store.register_metadata(a, pa);
  CHECK_FALSE(store.get_metadata(b));
  CHECK(store.get_metadata(a) == pa);

  bool hit    = true;
  int loads   = 0;
  auto loaded = store.get_or_load(
    b,
    [&] {
      ++loads;
      return pb;
    },
    &hit);
  CHECK_FALSE(hit);
  CHECK(loads == 1);
  CHECK(loaded == pb);
  CHECK(store.get_metadata(b) == pb);
  CHECK_FALSE(store.get_metadata(a));

  auto const stale = store.statistics().stale;
  store.register_metadata(a, std::make_shared<value>(3));
  CHECK(store.get_metadata(b) == pb);
  CHECK(store.statistics().stale == stale + 1);
}

TEST_CASE("metadata_store writable objects never advertise or share metadata",
          "[cache][metadata_store]")
{
  metadata_store store;
  tagged_object const writer{"/data/out.parquet", "\"v1\"", /*writable=*/true};
  REQUIRE(writer.identity().kind == identity_kind::opened_handle);
  auto const metadata = std::make_shared<value>(1);
  store.register_metadata(writer, metadata);
  CHECK(store.get_metadata(writer) == metadata);
  CHECK_FALSE(store.has_path("/data/out.parquet"));
  CHECK_FALSE(store.get_metadata(tagged_object{"/data/out.parquet", "\"v1\"", true}));
  CHECK_FALSE(store.get_metadata(tagged_object{"/data/out.parquet", "\"v1\""}));
}

TEST_CASE("Metadata store validates reopened objects and isolates access namespaces",
          "[cache][metadata_store][object_identity]")
{
  struct object : io_object {
    std::string path = "same-path", raw = "backend-locator", etag;
    const std::string& object_path() const noexcept override { return path; }
    const std::string& raw_file_cache_id() const noexcept override { return raw; }
    size_t size() const noexcept override { return 64; }
    std::string_view validation_tag() const noexcept override { return etag; }
    bool validation_tag_is_strong() const noexcept override { return true; }
  };
  object first, reopened;
  metadata_store store, other_scope;
  auto metadata = std::make_shared<value>(64);
  store.register_metadata(first, metadata);
  CHECK(store.get_metadata(first) == metadata);
  CHECK_FALSE(store.get_metadata(reopened));
  CHECK(first.identity_cache_key() != reopened.identity_cache_key());
  first.etag = reopened.etag = "opaque-multipart-tag-42";
  store.register_metadata(first, metadata);
  CHECK(store.get_metadata(reopened) == metadata);
  CHECK_FALSE(other_scope.get_metadata(first));
  CHECK(first.identity_cache_key() == reopened.identity_cache_key());
  reopened.etag = "new-tag";
  CHECK_FALSE(store.get_metadata(reopened));
  CHECK(first.identity_cache_key() != reopened.identity_cache_key());
  reopened.etag = first.etag;
  reopened.raw  = "different-backend-locator";
  CHECK_FALSE(store.get_metadata(reopened));
  CHECK(first.identity_cache_key() != reopened.identity_cache_key());
}

// ===========================================================================
// Single-flight loads and identity fences
// ===========================================================================

TEST_CASE("Metadata concurrent misses share success or failure and permit retry",
          "[cache][metadata_store]")
{
  for (bool fail : {false, true}) {
    metadata_store store;
    fake_object object("file", "file:v1", tag());
    std::promise<void> started, release;
    auto gate     = release.get_future().share();
    auto payload  = std::make_shared<value>(32);
    auto producer = std::async(std::launch::async, [&] {
      return store.get_or_load(object, [&]() -> std::shared_ptr<io_object_metadata> {
        started.set_value();
        gate.wait();
        if (fail) throw std::runtime_error("parse failed");
        return payload;
      });
    });
    started.get_future().wait();
    std::atomic<int> extra_loads{0};
    auto waiter = std::async(std::launch::async, [&] {
      return store.get_or_load(object, [&] {
        ++extra_loads;
        return payload;
      });
    });
    bool joined = wait_for_joined(store, 1);
    release.set_value();
    CHECK(joined);
    if (fail) {
      CHECK_THROWS_WITH(producer.get(), "parse failed");
      CHECK_THROWS_WITH(waiter.get(), "parse failed");
      CHECK_FALSE(store.get_metadata(object));
      CHECK(store.get_or_load(object, [&] { return payload; }) == payload);
      CHECK(store.statistics().failures == 1);
      CHECK(store.statistics().builds == 2);
    } else {
      CHECK(producer.get() == payload);
      CHECK(waiter.get() == payload);
      CHECK(store.statistics().builds == 1);
    }
    CHECK(extra_loads == 0);
    CHECK(store.describe_metadata_store().find("in_flight=0") != std::string::npos);
  }
}

TEST_CASE("Cancelling a metadata waiter leaves the shared producer alive",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_object object("file", "file:v1", tag());
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto payload  = std::make_shared<value>(32);
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(object, [&] {
      started.set_value();
      gate.wait();
      return payload;
    });
  });
  started.get_future().wait();
  std::stop_source stop;
  auto waiter = std::async(std::launch::async, [&] {
    return store.get_or_load(object, [&] { return payload; }, nullptr, stop.get_token());
  });
  bool joined = wait_for_joined(store, 1);
  stop.request_stop();
  auto status = waiter.wait_for(std::chrono::seconds(5));
  release.set_value();
  CHECK(joined);
  CHECK(status == std::future_status::ready);
  CHECK_THROWS_AS(waiter.get(), std::system_error);
  CHECK(producer.get() == payload);
  CHECK(store.get_metadata(object) == payload);
  CHECK(store.statistics().cancelled_waiters == 1);

  // A stop requested before the call cancels it without loading.
  bool loaded = false;
  CHECK_THROWS_AS(store.get_or_load(
                    object,
                    [&] {
                      loaded = true;
                      return payload;
                    },
                    nullptr,
                    stop.get_token()),
                  std::system_error);
  CHECK_FALSE(loaded);
}

TEST_CASE("Observing a newer identity fences an older in-flight publication",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_object old("file", "file:old", tag("old"));
  fake_object newer("file", "file:new", tag("new"));
  auto payload = std::make_shared<value>(32);
  auto result  = store.get_or_load(old, [&] {
    store.open_observed(newer);
    return payload;
  });
  CHECK(result == payload);
  CHECK_FALSE(store.get_metadata(old));
  CHECK_FALSE(store.has_path("file"));
  CHECK(store.statistics().stale == 1);
  auto fresh = std::make_shared<value>(64);
  store.register_metadata(newer, fresh);
  CHECK(store.get_metadata(newer) == fresh);
  CHECK(result == payload);
}

TEST_CASE("Metadata flights do not join a different identity", "[cache][metadata_store]")
{
  metadata_store store;
  fake_object old("file", "file:old", tag("old"));
  fake_object newer("file", "file:new", tag("new"));
  auto payload = std::make_shared<value>(32);
  auto fresh   = std::make_shared<value>(64);
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(old, [&] {
      started.set_value();
      gate.wait();
      return payload;
    });
  });
  started.get_future().wait();
  auto independent = std::async(std::launch::async, [&] {
    auto other_identity = store.get_or_load(newer, [&] { return fresh; });
    return other_identity == fresh;
  });
  auto status      = independent.wait_for(std::chrono::seconds(5));
  release.set_value();
  CHECK(status == std::future_status::ready);
  CHECK(independent.get());
  CHECK(producer.get() == payload);
  CHECK(store.get_metadata(newer) == fresh);
  CHECK(store.statistics().joined == 0);
  CHECK(store.statistics().builds == 2);
}

TEST_CASE("Metadata profiles are independent cache identities", "[cache][metadata_store]")
{
  metadata_store store;
  fake_object object("file", "file:v1", tag());
  auto first  = std::make_shared<value>(32);
  auto second = std::make_shared<value>(64);
  CHECK(store.get_or_load(object, [&] { return first; }, nullptr, {}, "profile-a") == first);
  CHECK(store.get_or_load(object, [&] { return second; }, nullptr, {}, "profile-b") == second);
  CHECK(store.get_metadata(object, "profile-a") == first);
  CHECK(store.get_metadata(object, "profile-b") == second);
  CHECK(store.statistics().builds == 2);
}

TEST_CASE("Targeted metadata invalidation preserves other profiles", "[cache][metadata_store]")
{
  metadata_store store;
  fake_object object("file", "file:v1", tag());
  auto canonical = std::make_shared<value>(32);
  auto profile   = std::make_shared<value>(64);
  store.register_metadata(object, canonical, "");
  store.register_metadata(object, profile, "iceberg_flat_v1");

  CHECK(store.invalidate("file", "iceberg_flat_v1"));
  CHECK(store.get_metadata(object, "") == canonical);
  CHECK_FALSE(store.get_metadata(object, "iceberg_flat_v1"));
  CHECK_FALSE(store.invalidate("file", "iceberg_flat_v1"));
}

TEST_CASE("Same-key opens preserve warm metadata and advance the observation fence",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_object first("file", "file:a", tag("a"));
  fake_object delayed("file", "file:b", tag("b"));
  fake_object latest("file", "file:a", tag("a"));
  auto payload = std::make_shared<value>(32);
  store.register_metadata(first, payload);
  store.open_observed(latest);
  store.open_observed(delayed);
  store.register_metadata(delayed, std::make_shared<value>(64));
  CHECK(store.get_metadata(latest) == payload);
  CHECK(store.statistics().replacements == 0);
  CHECK(store.statistics().stale == 1);
}

TEST_CASE("Metadata store isolates identity and ioctx", "[cache][metadata_store]")
{
  metadata_store store, other;
  auto v = std::make_shared<value>(100);
  fake_object first("file", "file:v1", tag());
  store.register_metadata(first, v);
  CHECK(store.get_metadata(first) == v);
  CHECK_FALSE(other.get_metadata(first));
  CHECK(store.get_metadata(first) == v);

  auto next = std::make_shared<value>(200);
  fake_object replacement("file", "file:v2", tag("v2"));
  store.register_metadata(replacement, next);
  CHECK(store.statistics().entries == 1);
  store.register_metadata(first, v);
  CHECK(store.get_metadata(replacement) == next);
  CHECK(store.statistics().bypasses == 1);
  store.register_metadata(replacement, next);
  CHECK(store.statistics().entries == 1);
  CHECK(store.statistics().bypasses == 1);
  CHECK(v->bytes == 100);
}

TEST_CASE("Identity replacement invalidates entries for a path", "[cache][metadata_store]")
{
  metadata_store store;
  fake_object first("file", "file:v1", tag("v1"));
  store.register_metadata(first, std::make_shared<value>(100));
  CHECK(store.statistics().entries == 1);

  fake_object replacement("file", "file:v2", tag("v2"));
  store.open_observed(replacement);
  CHECK_FALSE(store.get_metadata(first));
  CHECK_FALSE(store.has_path("file"));
  CHECK(store.statistics().entries == 0);
}

TEST_CASE("Identity replacement retires every metadata profile for a path",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_object first("file", "file:v1", tag("v1"));
  auto canonical = std::make_shared<value>(100);
  auto iceberg   = std::make_shared<value>(200);
  auto extra     = std::make_shared<value>(300);
  store.register_metadata(first, canonical, "");
  store.register_metadata(first, iceberg, "iceberg_flat_v1");
  store.register_metadata(first, extra, "other");
  CHECK(store.statistics().entries == 3);

  fake_object replacement("file", "file:v2", tag("v2"));
  store.open_observed(replacement);

  CHECK_FALSE(store.get_metadata(first, ""));
  CHECK_FALSE(store.get_metadata(first, "iceberg_flat_v1"));
  CHECK_FALSE(store.get_metadata(first, "other"));
  CHECK(store.statistics().entries == 0);
}

TEST_CASE("Metadata store supports concurrent publication", "[cache][metadata_store]")
{
  metadata_store store;
  std::atomic<bool> all_hits{true};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([&, i] {
      for (int j = 0; j < 100; ++j) {
        auto object = fake_object("file-" + std::to_string(i) + "-" + std::to_string(j),
                                  "key-" + std::to_string(i) + "-" + std::to_string(j),
                                  tag());
        store.register_metadata(object, std::make_shared<value>(512));
        auto held = store.get_metadata(object);
        if (!held) all_hits.store(false, std::memory_order_relaxed);
      }
    });
  for (auto& thread : threads)
    thread.join();
  CHECK(all_hits.load(std::memory_order_relaxed));
  CHECK(store.statistics().entries == 400);
}

TEST_CASE("Open-local entries do not advertise reusable path hints", "[cache][metadata_store]")
{
  metadata_store store;
  fake_object object("file", "file:open", {identity_kind::opened_handle, 42, "1"});
  auto v = std::make_shared<value>(256);
  store.register_metadata(object, v);
  CHECK_FALSE(store.has_path("file"));
  CHECK(store.get_metadata(object) == v);
  fake_object reopened("file", "file:open2", {identity_kind::opened_handle, 42, "2"});
  CHECK_FALSE(store.get_metadata(reopened));
}

TEST_CASE("Local metadata reuse requires the observed identity", "[cache][metadata_store]")
{
  object_identity original{identity_kind::local_stat, 256, "3"};
  original.device   = 1;
  original.inode    = 2;
  original.ctime_ns = 3;
  fake_object object("file", "file:local", original);
  metadata_store store;
  auto stable = std::make_shared<value>(256);
  store.register_metadata(object, stable);
  CHECK(store.get_metadata(object) == stable);

  auto replacement  = original;
  replacement.inode = 4;
  fake_object replaced("file", "file:local", replacement);
  CHECK_FALSE(store.get_metadata(replaced));
  auto fresh = std::make_shared<value>(512);
  store.register_metadata(replaced, fresh);
  CHECK(store.get_metadata(replaced) == fresh);
}

TEST_CASE("Local stat identity reuses metadata across opens", "[cache][metadata_store]")
{
  object_identity local_identity{identity_kind::local_stat, 256, "same-ctime"};
  local_identity.device   = 1;
  local_identity.inode    = 2;
  local_identity.ctime_ns = 3;
  fake_object first("file", "file:local", local_identity);
  fake_object reopened("file", "file:local", local_identity);
  metadata_store store;
  auto old_value = std::make_shared<value>(256);
  store.register_metadata(first, old_value);
  CHECK(store.get_metadata(reopened) == old_value);
  bool loaded    = false;
  bool hit       = false;
  auto new_value = store.get_or_load(
    reopened,
    [&] {
      loaded = true;
      return std::make_shared<value>(512);
    },
    &hit);
  CHECK_FALSE(loaded);
  CHECK(hit);
  CHECK(new_value == old_value);
  CHECK(store.get_metadata(reopened) == old_value);
}

TEST_CASE("Metadata replacement releases payloads outside the store lock",
          "[cache][metadata_store]")
{
  metadata_store store;
  bool released = false;
  struct reentrant_value : value {
    std::function<void()> on_release;
    explicit reentrant_value(std::function<void()> callback)
      : value(128), on_release(std::move(callback))
    {
    }
    ~reentrant_value() override { on_release(); }
  };
  fake_object first("file", "file:v1", tag("v1"));
  store.register_metadata(
    first, std::make_shared<reentrant_value>([&] { released = store.statistics().entries == 1; }));
  fake_object second("file", "file:v2", tag("v2"));
  store.register_metadata(second, std::make_shared<value>(128));
  CHECK(released);
}

TEST_CASE("Metadata registrations the store drops are released outside its lock",
          "[cache][metadata_store]")
{
  metadata_store store;
  struct reentrant_value : value {
    metadata_store& store;
    bool& released;
    reentrant_value(metadata_store& owner, bool& flag) : value(16), store(owner), released(flag) {}
    // Takes the store lock: released under it, this would deadlock.
    ~reentrant_value() override { released = store.statistics().entries <= 1; }
  };
  bool released = false;
  fake_object first("file", "file", tag());
  store.register_metadata(first, std::make_shared<value>(1));

  SECTION("a duplicate of a published identity")
  {
    store.register_metadata(fake_object("file", "file", tag()),
                            std::make_shared<reentrant_value>(store, released));
  }
  SECTION("a registration from an object opened before a write")
  {
    fake_object before("file", "file", tag());
    store.stamp(before);
    store.invalidate_written("file");
    store.register_metadata(before, std::make_shared<reentrant_value>(store, released));
    CHECK(store.statistics().stale == 1);
  }
  SECTION("a registration of an identity older than the observed one")
  {
    fake_object older("file", "file", tag("v0"));
    fake_object newer("file", "file", tag("v2"));
    store.register_metadata(newer, std::make_shared<value>(2));
    store.register_metadata(older, std::make_shared<reentrant_value>(store, released));
    CHECK(store.statistics().stale == 1);
  }
  CHECK(released);
}

TEST_CASE("Metadata path and profile delimiters cannot alias another tuple",
          "[cache][metadata_store][key_encoding]")
{
  auto const reverse = GENERATE(false, true);
  metadata_store store;
  fake_object first("p", "same-key", tag());
  fake_object second("p\nprofile=q", "same-key", tag());
  auto a = std::make_shared<value>(32);
  auto b = std::make_shared<value>(64);
  CHECK(store.get_or_load(first, [&] { return a; }, nullptr, {}, "q") == a);
  CHECK(store.get_or_load(second, [&] { return b; }) == b);
  CHECK(store.get_metadata(first, "q") == a);
  CHECK(store.get_metadata(second) == b);
  CHECK(store.statistics().entries == 2);
  CHECK(store.statistics().builds == 2);
  CHECK(store.has_path(second.object_path()));
  // has_path is a hint for the canonical profile only.
  CHECK_FALSE(store.has_path(first.object_path()));
  if (reverse) {
    CHECK(store.invalidate(second.object_path()));
    CHECK_FALSE(store.get_metadata(second));
    CHECK(store.get_metadata(first, "q") == a);
  } else {
    CHECK(store.invalidate(first.object_path(), "q"));
    CHECK_FALSE(store.get_metadata(first, "q"));
    CHECK(store.get_metadata(second) == b);
  }
  CHECK(store.statistics().entries == 1);
  CHECK_FALSE(
    store.invalidate(reverse ? second.object_path() : first.object_path(), reverse ? "" : "q"));
  CHECK(store.statistics().entries == 1);
}

TEST_CASE("Metadata identity retirement follows the raw path across encoded profiles",
          "[cache][metadata_store][key_encoding]")
{
  metadata_store store;
  fake_object first("p", "same-key", tag());
  fake_object second("p\nprofile=q", "same-key", tag());
  auto a = std::make_shared<value>(32);
  auto b = std::make_shared<value>(64);
  for (auto const& profile :
       std::vector<std::string>{"", "q", "\nprofile=q", "1:p:q", std::string("a\0b", 3)})
    store.register_metadata(first, a, profile);
  store.register_metadata(second, b);
  CHECK(store.statistics().entries == 6);
  fake_object newer(first.object_path(), "new", tag("v2"));
  store.open_observed(newer);
  CHECK(store.statistics().entries == 1);
  CHECK(store.statistics().replacements == 5);
  CHECK_FALSE(store.has_path(first.object_path()));
  CHECK(store.has_path(second.object_path()));
  CHECK(store.get_metadata(second) == b);
  store.register_metadata(newer, a, "q");
  CHECK(store.get_metadata(newer, "q") == a);
  CHECK(store.get_metadata(second) == b);
}

TEST_CASE("Metadata colliding path profile spellings keep independent flights",
          "[cache][metadata_store][key_encoding]")
{
  metadata_store store;
  fake_object first("p", "same-key", tag());
  fake_object second("p\nprofile=q", "same-key", tag());
  auto a = std::make_shared<value>(32);
  auto b = std::make_shared<value>(64);
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(
      first,
      [&] {
        started.set_value();
        gate.wait();
        return a;
      },
      nullptr,
      {},
      "q");
  });
  started.get_future().wait();
  auto independent =
    std::async(std::launch::async, [&] { return store.get_or_load(second, [&] { return b; }); });
  auto status = independent.wait_for(std::chrono::seconds(2));
  release.set_value();
  CHECK(status == std::future_status::ready);
  CHECK(independent.get() == b);
  CHECK(producer.get() == a);
  CHECK(store.statistics().joined == 0);
  CHECK(store.statistics().builds == 2);
  CHECK(store.get_metadata(first, "q") == a);
  CHECK(store.get_metadata(second) == b);
}

TEST_CASE("Metadata tuple encoding accepts empty paths and embedded delimiters",
          "[cache][metadata_store][key_encoding]")
{
  metadata_store store;
  std::vector<std::pair<std::string, std::string>> tuples{{"", ""},
                                                          {"", "1:p"},
                                                          {"1:", "p"},
                                                          {"p:", "q"},
                                                          {"p", ":q"},
                                                          {"p\nprofile=", "q"},
                                                          {"p", "\nprofile=q"},
                                                          {std::string("p\0q", 3), ""},
                                                          {"p", std::string("\0q", 2)}};
  std::vector<std::unique_ptr<fake_object>> objects;
  std::vector<std::shared_ptr<value>> payloads;
  for (auto const& [path, profile] : tuples) {
    objects.push_back(std::make_unique<fake_object>(path, "same-key", tag()));
    payloads.push_back(std::make_shared<value>(payloads.size()));
    store.register_metadata(*objects.back(), payloads.back(), profile);
  }
  CHECK(store.statistics().entries == tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    CHECK(store.get_metadata(*objects[i], tuples[i].second) == payloads[i]);
    CHECK(store.invalidate(tuples[i].first, tuples[i].second));
    CHECK(store.statistics().entries == tuples.size() - i - 1);
  }
}

// ===========================================================================
// Write invalidation
// ===========================================================================

TEST_CASE("metadata_store drops a registration from an object opened before a write",
          "[cache][metadata_store][write]")
{
  metadata_store store;
  // An in-place write can leave the stat identity unchanged (coarse ctime):
  // only the write epoch tells the two opens apart.
  fake_object before("/data/t.parquet", "/data/t.parquet", local());
  store.stamp(before);
  store.register_metadata(before, std::make_shared<value>(1));
  REQUIRE(store.has_path("/data/t.parquet"));

  store.invalidate_written("/data/t.parquet");
  CHECK_FALSE(store.has_path("/data/t.parquet"));
  CHECK_FALSE(store.get_metadata(before));
  CHECK(store.statistics().entries == 0);

  // A parse through the pre-write object that finishes now is dropped; one
  // through an object opened afterwards registers again.
  store.register_metadata(before, std::make_shared<value>(2));
  CHECK_FALSE(store.has_path("/data/t.parquet"));
  CHECK(store.statistics().stale == 1);
  fake_object after("/data/t.parquet", "/data/t.parquet", local());
  store.stamp(after);
  store.register_metadata(after, std::make_shared<value>(3));
  CHECK(store.has_path("/data/t.parquet"));
  CHECK(bytes_of(store.get_metadata(after)) == 3);

  // Other paths are untouched by the write.
  fake_object other("/data/u.parquet", "/data/u.parquet", local(7));
  store.stamp(other);
  store.register_metadata(other, std::make_shared<value>(4));
  store.invalidate_written("/data/t.parquet");
  CHECK(bytes_of(store.get_metadata(other)) == 4);
}

TEST_CASE("metadata_store restarts the identity history of a written path",
          "[cache][metadata_store][write]")
{
  metadata_store store;
  // `reader` is opened, then `newest` (a later open generation) is observed
  // with another identity before the write.
  fake_object reader("file", "file", local(2));
  fake_object newest("file", "file", local(3));
  REQUIRE(reader.open_generation() < newest.open_generation());
  store.open_observed(newest);

  // Once written, the old observation no longer fences the path: an object
  // stamped after the write registers whatever its open generation.
  store.invalidate_written("file");
  store.stamp(reader);
  store.register_metadata(reader, std::make_shared<value>(1));
  CHECK(bytes_of(store.get_metadata(reader)) == 1);
  // An object stamped before the write cannot observe either.
  store.open_observed(newest);
  CHECK(bytes_of(store.get_metadata(reader)) == 1);
}

TEST_CASE("metadata_store loads do not join across a write", "[cache][metadata_store][write]")
{
  metadata_store store;
  fake_object before("file", "file", local());
  store.stamp(before);
  auto stale = std::make_shared<value>(1);
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(before, [&] {
      started.set_value();
      gate.wait();
      return stale;
    });
  });
  started.get_future().wait();

  // Same identity, opened after the write: it must not receive the pre-write parse.
  store.invalidate_written("file");
  fake_object after("file", "file", local());
  store.stamp(after);
  auto fresh = std::make_shared<value>(2);
  auto independent =
    std::async(std::launch::async, [&] { return store.get_or_load(after, [&] { return fresh; }); });
  auto const status = independent.wait_for(std::chrono::seconds(5));
  release.set_value();
  CHECK(status == std::future_status::ready);
  CHECK(independent.get() == fresh);
  // The pre-write load still answers its own caller, but publishes nothing.
  CHECK(producer.get() == stale);
  CHECK(store.get_metadata(after) == fresh);
  CHECK(store.statistics().joined == 0);
  CHECK(store.statistics().builds == 2);
}

TEST_CASE("metadata_store write invalidation drops every profile of the path only",
          "[cache][metadata_store][write]")
{
  metadata_store store;
  fake_object before("/data/t.parquet", "/data/t.parquet", local());
  fake_object other("/data/u.parquet", "/data/u.parquet", local(7));
  store.stamp(before);
  store.stamp(other);
  for (auto const* profile : {"", "a", "b"}) {
    store.register_metadata(before, std::make_shared<value>(1), profile);
    store.register_metadata(other, std::make_shared<value>(2), profile);
  }
  REQUIRE(store.statistics().entries == 6);
  store.invalidate_written("/data/t.parquet");
  CHECK(store.statistics().entries == 3);
  for (auto const* profile : {"", "a", "b"}) {
    CHECK_FALSE(store.get_metadata(before, profile));
    CHECK(store.get_metadata(other, profile));
  }
  // A path without entries is a no-op beyond its epoch.
  store.invalidate_written("/data/never-registered.parquet");
  CHECK(store.statistics().entries == 3);
}

TEST_CASE("metadata_store returns an entry only to opens of the epoch it was published under",
          "[cache][metadata_store][write]")
{
  // A parse through an object opened while a write was in progress can land
  // after the write advanced the path's epoch and before the write erased the
  // path's entries.  A write of a path sharing the epoch reproduces that state.
  std::string const path = "/data/t.parquet";
  auto const neighbour   = epoch_neighbour(path);
  metadata_store store;
  // Opened (constructed) before `torn`, stamped only after the write: an older
  // open generation must not keep it from replacing the older-epoch entry.
  fake_object after(path, path, local());
  fake_object torn(path, path, local());
  REQUIRE(after.open_generation() < torn.open_generation());
  store.stamp(torn);
  store.register_metadata(torn, std::make_shared<value>(1));
  REQUIRE(store.has_path(path));

  store.invalidate_written(neighbour);
  REQUIRE(store.statistics().entries == 1);
  // The entry is still there for the open it was published by ...
  CHECK(bytes_of(store.get_metadata(torn)) == 1);
  // ... but an open stamped after the write, with an equal stat identity, gets
  // neither the entry nor a path hint.
  store.stamp(after);
  CHECK_FALSE(store.has_path(path));
  CHECK_FALSE(store.get_metadata(after));
  bool hit       = true;
  auto const own = std::make_shared<value>(2);
  CHECK(store.get_or_load(after, [&] { return own; }, &hit) == own);
  CHECK_FALSE(hit);

  // Its publication replaced the older-epoch entry.
  CHECK(store.get_metadata(after) == own);
  CHECK(store.has_path(path));
  CHECK(store.statistics().entries == 1);
  CHECK(store.statistics().replacements == 1);
  CHECK_FALSE(store.get_metadata(torn));
}

TEST_CASE("metadata_store dump keeps every entry on one escaped line", "[cache][metadata_store]")
{
  metadata_store store;
  fake_object object("dir\\file", "dir\\file", tag());
  store.register_metadata(object, std::make_shared<value>(1), std::string("a\nb\0c", 5));
  auto const dump = store.describe_metadata_store();
  // The summary line and one line for the entry.
  CHECK(std::count(dump.begin(), dump.end(), '\n') == 2);
  CHECK(dump.find('\0') == std::string::npos);
  CHECK(dump.find("key=8:dir\\\\filea\\x0ab\\x00c ") != std::string::npos);
}

TEST_CASE("metadata_store targeted invalidation leaves the write epoch alone",
          "[cache][metadata_store][write]")
{
  // Unlike a write, a targeted retire does not fence the object that is open:
  // the same open may rebuild and republish the entry it retired.
  metadata_store store;
  fake_object object("/data/t.parquet", "/data/t.parquet", local());
  store.stamp(object);
  store.register_metadata(object, std::make_shared<value>(1));
  CHECK(store.invalidate("/data/t.parquet"));
  store.register_metadata(object, std::make_shared<value>(2));
  CHECK(bytes_of(store.get_metadata(object)) == 2);
  CHECK(store.has_path("/data/t.parquet"));
  CHECK(store.statistics().stale == 0);
}
