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

/// An object with the base io_object identity: a backend that declares its
/// tag strong.
class tagged_object final : public io_object {
 public:
  explicit tagged_object(std::string path, std::string tag = {})
    : path_(std::move(path)), tag_(std::move(tag))
  {
  }
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return path_; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return path_; }
  [[nodiscard]] size_t size() const noexcept override { return 64; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return tag_; }
  [[nodiscard]] bool validation_tag_is_strong() const noexcept override { return true; }

 private:
  std::string path_;
  std::string tag_;
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

TEST_CASE("metadata_store targeted invalidation lets the open object republish",
          "[cache][metadata_store]")
{
  // A targeted retire does not fence the object that is open: the same open
  // may rebuild and republish the entry it retired.
  metadata_store store;
  fake_object object("/data/t.parquet", "/data/t.parquet", local());
  store.register_metadata(object, std::make_shared<value>(1));
  CHECK(store.invalidate("/data/t.parquet"));
  store.register_metadata(object, std::make_shared<value>(2));
  CHECK(bytes_of(store.get_metadata(object)) == 2);
  CHECK(store.has_path("/data/t.parquet"));
  CHECK(store.statistics().stale == 0);
}
