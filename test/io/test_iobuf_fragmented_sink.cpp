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

#include <cucascade/io/iobuf_fragmented_sink.hpp>

#include <catch2/catch_all.hpp>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using cucascade::io::IoBufFragmentedSink;
using Fill    = IoBufFragmentedSink::Fill;
using Segment = IoBufFragmentedSink::Segment;

namespace {

/// Deterministic body pattern: the byte at logical body offset `i`.
std::byte body_byte(std::size_t i) noexcept
{
  auto const x = static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
  return static_cast<std::byte>(x >> 56);
}

/// Injects a small hole scratch for the scope of a test and restores the default afterwards.
class scratch_guard {
 public:
  explicit scratch_guard(std::size_t len) : _block(len, std::byte{0xEE})
  {
    IoBufFragmentedSink::setHoleScratch(_block.data(), _block.size());
  }
  ~scratch_guard() { IoBufFragmentedSink::setHoleScratch(nullptr, 0); }
  scratch_guard(scratch_guard const&)            = delete;
  scratch_guard& operator=(scratch_guard const&) = delete;
  scratch_guard(scratch_guard&&)                 = delete;
  scratch_guard& operator=(scratch_guard&&)      = delete;

  [[nodiscard]] std::byte* data() noexcept { return _block.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return _block.size(); }

 private:
  std::vector<std::byte> _block;
};

std::byte* ptr(iovec const& v) noexcept { return static_cast<std::byte*>(v.iov_base); }

std::size_t sum_lengths(iovec const* v, std::size_t n) noexcept
{
  std::size_t total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    total += v[i].iov_len;
  }
  return total;
}

/// Emulate a receive of `n` body bytes into the iovecs of a fill (as recvmsg would).
void deliver(iovec const* v, std::size_t iov_count, std::size_t body_offset, std::size_t n)
{
  for (std::size_t i = 0; i < iov_count && n > 0; ++i) {
    std::size_t const take = std::min(n, v[i].iov_len);
    for (std::size_t j = 0; j < take; ++j) {
      ptr(v[i])[j] = body_byte(body_offset + j);
    }
    body_offset += take;
    n -= take;
  }
}

}  // namespace

// -------------------------------------------------------------------------------------------------
// Construction / introspection
// -------------------------------------------------------------------------------------------------

TEST_CASE("IoBufFragmentedSink: create() is an empty, finished chain", "[io][iobuf_sink]")
{
  auto sink = IoBufFragmentedSink::create();
  CHECK(sink.empty());
  CHECK_FALSE(sink.isChained());
  CHECK(sink.countChainElements() == 0);
  CHECK(sink.computeChainDataLength() == 0);
  CHECK(sink.computeChainDataLengthWithoutHoles() == 0);
  CHECK(sink.length() == 0);
  CHECK(sink.segments().empty());
  CHECK(sink.done());
  CHECK(sink.remaining() == 0);

  std::array<iovec, 4> iov{};
  auto const fill = sink.fillIovecs(iov.data(), iov.size(), 1024);
  CHECK(fill.iov_count == 0);
  CHECK(fill.bytes == 0);
}

TEST_CASE("IoBufFragmentedSink: documented footprint", "[io][iobuf_sink]")
{
  // Keep the header's "Size" section honest (LP64).
  STATIC_REQUIRE(sizeof(void*) != 8 || sizeof(IoBufFragmentedSink) == 168);
  STATIC_REQUIRE(std::is_nothrow_move_constructible_v<IoBufFragmentedSink>);
  STATIC_REQUIRE(IoBufFragmentedSink::kInlineSegments == 4);
}

TEST_CASE("IoBufFragmentedSink: wrapBuffer is a single unchained data segment", "[io][iobuf_sink]")
{
  std::array<std::byte, 100> buf{};
  auto sink = IoBufFragmentedSink::wrapBuffer(buf.data(), buf.size());
  CHECK_FALSE(sink.empty());
  CHECK_FALSE(sink.isChained());
  CHECK(sink.countChainElements() == 1);
  CHECK(sink.length() == 100);
  CHECK(sink.computeChainDataLength() == 100);
  CHECK(sink.computeChainDataLengthWithoutHoles() == 100);
  CHECK(sink.front().data == buf.data());
  CHECK(sink.front().length == 100);
  CHECK_FALSE(sink.front().hole);
  CHECK(&sink.front() == &sink.back());
  CHECK_FALSE(sink.done());
  CHECK(sink.remaining() == 100);
}

TEST_CASE("IoBufFragmentedSink: appends, prepends and IOBuf-like introspection", "[io][iobuf_sink]")
{
  std::array<std::byte, 10> a{};
  std::array<std::byte, 20> b{};
  std::array<std::byte, 5> c{};
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(7);
  sink.appendData(b.data(), b.size());
  CHECK(sink.isChained());
  CHECK(sink.countChainElements() == 3);
  CHECK(sink.computeChainDataLength() == 37);
  CHECK(sink.computeChainDataLengthWithoutHoles() == 30);
  CHECK(sink.back().data == b.data());

  sink.prependHole(3);
  sink.prependData(c.data(), c.size());
  auto const segs = sink.segments();
  REQUIRE(segs.size() == 5);
  CHECK(segs[0].data == c.data());
  CHECK((segs[1].hole && segs[1].data == nullptr && segs[1].length == 3));
  CHECK(segs[2].data == a.data());
  CHECK((segs[3].hole && segs[3].length == 7));
  CHECK(segs[4].data == b.data());
  CHECK(sink.computeChainDataLength() == 45);
  CHECK(sink.computeChainDataLengthWithoutHoles() == 35);
  CHECK(sink.length() == 5);  // the head before the receive starts

  sink.commit(6);  // into the 3-byte hole
  CHECK(sink.length() == 3);
  CHECK_THROWS_AS(sink.prependHole(1), std::logic_error);
  CHECK_THROWS_AS(sink.prependData(a.data(), 1), std::logic_error);
  sink.appendHole(2);  // appending mid-receive is fine
  CHECK(sink.countChainElements() == 6);
  CHECK(sink.remaining() == 41);
}

TEST_CASE("IoBufFragmentedSink: invalid segments throw at construction", "[io][iobuf_sink]")
{
  std::array<std::byte, 4> a{};
  CHECK_THROWS_AS(IoBufFragmentedSink::wrapBuffer(nullptr, 4), std::invalid_argument);
  CHECK_THROWS_AS(IoBufFragmentedSink::wrapBuffer(a.data(), 0), std::invalid_argument);
  auto sink = IoBufFragmentedSink::create();
  CHECK_THROWS_AS(sink.appendHole(0), std::invalid_argument);
  CHECK_THROWS_AS(sink.appendData(nullptr, 1), std::invalid_argument);
  CHECK_THROWS_AS(sink.prependHole(0), std::invalid_argument);
  CHECK(sink.empty());
}

TEST_CASE("IoBufFragmentedSink: chains longer than the inline storage spill and stay ordered",
          "[io][iobuf_sink]")
{
  std::vector<std::array<std::byte, 8>> bufs(6);
  auto sink = IoBufFragmentedSink::create();
  for (auto& buf : bufs) {
    sink.appendData(buf.data(), buf.size());
    sink.appendHole(1);
  }
  sink.prependHole(2);
  REQUIRE(sink.countChainElements() == 13);
  auto const segs = sink.segments();
  CHECK(segs[0].hole);
  for (std::size_t i = 0; i < bufs.size(); ++i) {
    CHECK(segs[1 + 2 * i].data == bufs[i].data());
    CHECK(segs[2 + 2 * i].hole);
  }
  CHECK(sink.computeChainDataLength() == 2 + 6 * 9);

  // Moving keeps the chain (spilled storage is stolen, not copied).
  IoBufFragmentedSink moved = std::move(sink);
  CHECK(moved.countChainElements() == 13);
  CHECK(moved.segments()[1].data == bufs[0].data());

  moved.clear();
  CHECK(moved.empty());
  CHECK(moved.done());
  // Back to inline storage after clear().
  moved.appendData(bufs[0].data(), 8);
  CHECK(moved.segments()[0].data == bufs[0].data());
  CHECK(moved.computeChainDataLength() == 8);
}

// -------------------------------------------------------------------------------------------------
// Fills
// -------------------------------------------------------------------------------------------------

TEST_CASE("IoBufFragmentedSink: fill over a single data segment", "[io][iobuf_sink]")
{
  std::array<std::byte, 100> buf{};
  auto sink = IoBufFragmentedSink::wrapBuffer(buf.data(), buf.size());
  std::array<iovec, 4> iov{};

  auto fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
  CHECK(fill.iov_count == 1);
  CHECK(fill.bytes == 100);
  CHECK(ptr(iov[0]) == buf.data());
  CHECK(iov[0].iov_len == 100);

  fill = sink.fillIovecs(iov.data(), iov.size(), 40);
  CHECK(fill.iov_count == 1);
  CHECK(fill.bytes == 40);

  sink.commit(40);
  fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
  CHECK(fill.iov_count == 1);
  CHECK(ptr(iov[0]) == buf.data() + 40);
  CHECK(iov[0].iov_len == 60);

  CHECK(sink.fillIovecs(iov.data(), 0, 1000).iov_count == 0);
  CHECK(sink.fillIovecs(iov.data(), iov.size(), 0).iov_count == 0);
}

TEST_CASE("IoBufFragmentedSink: data+hole+data fills honour max_iov", "[io][iobuf_sink]")
{
  std::array<std::byte, 100> a{};
  std::array<std::byte, 200> b{};
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(50);  // shorter than the default 64 KiB scratch: one iovec
  sink.appendData(b.data(), b.size());
  std::byte* const scratch = IoBufFragmentedSink::holeScratch();

  std::size_t const max_iov = GENERATE(1u, 2u, 3u, 64u);
  std::vector<iovec> iov(max_iov);
  auto const fill              = sink.fillIovecs(iov.data(), max_iov, 1u << 20);
  std::size_t const expect_iov = std::min<std::size_t>(max_iov, 3);
  REQUIRE(fill.iov_count == expect_iov);
  std::array<std::size_t, 3> const covered{100, 150, 350};
  CHECK(fill.bytes == covered[expect_iov - 1]);
  CHECK(sum_lengths(iov.data(), fill.iov_count) == fill.bytes);
  CHECK(ptr(iov[0]) == a.data());
  if (fill.iov_count > 1) {
    CHECK(ptr(iov[1]) == scratch);
    CHECK(iov[1].iov_len == 50);
  }
  if (fill.iov_count > 2) {
    CHECK(ptr(iov[2]) == b.data());
    CHECK(iov[2].iov_len == 200);
  }
}

TEST_CASE("IoBufFragmentedSink: fills honour max_bytes", "[io][iobuf_sink]")
{
  std::array<std::byte, 100> a{};
  std::array<std::byte, 200> b{};
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(50);
  sink.appendData(b.data(), b.size());
  std::array<iovec, 8> iov{};

  SECTION("smaller than a segment")
  {
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 30);
    CHECK(fill.iov_count == 1);
    CHECK(fill.bytes == 30);
    CHECK(iov[0].iov_len == 30);
  }
  SECTION("equal to the first segment")
  {
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 100);
    CHECK(fill.iov_count == 1);
    CHECK(fill.bytes == 100);
  }
  SECTION("spanning two segments")
  {
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 120);
    CHECK(fill.iov_count == 2);
    CHECK(fill.bytes == 120);
    CHECK(iov[1].iov_len == 20);
  }
  SECTION("from inside the hole into the tail")
  {
    sink.commit(110);
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 60);
    CHECK(fill.iov_count == 2);
    CHECK(fill.bytes == 60);
    CHECK(ptr(iov[0]) == IoBufFragmentedSink::holeScratch() + 10);
    CHECK(iov[0].iov_len == 40);
    CHECK(ptr(iov[1]) == b.data());
    CHECK(iov[1].iov_len == 20);
  }
}

TEST_CASE("IoBufFragmentedSink: holes vs scratch size", "[io][iobuf_sink]")
{
  scratch_guard scratch(100);
  CHECK(IoBufFragmentedSink::holeScratchSize() == 100);
  std::array<std::byte, 10> a{};
  std::array<std::byte, 10> b{};
  std::array<iovec, 16> iov{};

  auto make = [&](std::size_t hole) {
    auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
    sink.appendHole(hole);
    sink.appendData(b.data(), b.size());
    return sink;
  };

  SECTION("shorter")
  {
    auto sink       = make(60);
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
    REQUIRE(fill.iov_count == 3);
    CHECK(ptr(iov[1]) == scratch.data());
    CHECK(iov[1].iov_len == 60);
  }
  SECTION("equal")
  {
    auto sink       = make(100);
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
    REQUIRE(fill.iov_count == 3);
    CHECK(iov[1].iov_len == 100);
  }
  SECTION("longer: repeated iovecs over the same block")
  {
    auto sink       = make(250);
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
    REQUIRE(fill.iov_count == 5);
    CHECK(fill.bytes == 270);
    for (std::size_t i = 1; i <= 3; ++i) {
      CHECK(ptr(iov[i]) == scratch.data());
    }
    CHECK(iov[1].iov_len == 100);
    CHECK(iov[2].iov_len == 100);
    CHECK(iov[3].iov_len == 50);
  }
  SECTION("longer, resumed mid-hole: scratch position is the hole offset modulo the size")
  {
    auto sink = make(250);
    sink.commit(10 + 130);
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
    REQUIRE(fill.iov_count == 3);
    CHECK(ptr(iov[0]) == scratch.data() + 30);
    CHECK(iov[0].iov_len == 70);
    CHECK(ptr(iov[1]) == scratch.data());
    CHECK(iov[1].iov_len == 50);
    CHECK(ptr(iov[2]) == b.data());
  }
  SECTION("longer, iovec budget runs out inside the hole")
  {
    auto sink       = make(250);
    auto const fill = sink.fillIovecs(iov.data(), 2, 1000);
    CHECK(fill.iov_count == 2);
    CHECK(fill.bytes == 110);
  }
}

TEST_CASE("IoBufFragmentedSink: holes at the start and at the end", "[io][iobuf_sink]")
{
  std::array<std::byte, 10> a{};
  std::array<iovec, 4> iov{};
  auto sink = IoBufFragmentedSink::create();
  sink.appendHole(5);
  sink.appendData(a.data(), a.size());
  sink.appendHole(7);

  auto fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
  REQUIRE(fill.iov_count == 3);
  CHECK(fill.bytes == 22);
  CHECK(ptr(iov[0]) == IoBufFragmentedSink::holeScratch());
  CHECK(ptr(iov[1]) == a.data());
  CHECK(ptr(iov[2]) == IoBufFragmentedSink::holeScratch());

  sink.commit(15);
  CHECK_FALSE(sink.done());
  fill = sink.fillIovecs(iov.data(), iov.size(), 1000);
  CHECK(fill.iov_count == 1);
  CHECK(fill.bytes == 7);
  sink.commit(7);
  CHECK(sink.done());
  CHECK(sink.fillIovecs(iov.data(), iov.size(), 1000).iov_count == 0);
}

// -------------------------------------------------------------------------------------------------
// Commits, scatter, reset/clear
// -------------------------------------------------------------------------------------------------

TEST_CASE("IoBufFragmentedSink: commits move the cursor across segments", "[io][iobuf_sink]")
{
  std::array<std::byte, 100> a{};
  std::array<std::byte, 200> b{};
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(50);
  sink.appendData(b.data(), b.size());
  std::array<iovec, 4> iov{};

  SECTION("partial within a segment")
  {
    sink.commit(30);
    CHECK(sink.cursor() == 30);
    CHECK(sink.length() == 100);
    CHECK(sink.fillIovecs(iov.data(), 1, 1000).bytes == 70);
    CHECK(ptr(iov[0]) == a.data() + 30);
  }
  SECTION("exactly at a boundary")
  {
    sink.commit(100);
    CHECK(sink.length() == 50);
    CHECK(sink.fillIovecs(iov.data(), 1, 1000).bytes == 50);
    CHECK(ptr(iov[0]) == IoBufFragmentedSink::holeScratch());
  }
  SECTION("spanning segments")
  {
    sink.commit(160);
    CHECK(sink.cursor() == 160);
    CHECK(sink.remaining() == 190);
    CHECK(sink.fillIovecs(iov.data(), 1, 1000).bytes == 190);
    CHECK(ptr(iov[0]) == b.data() + 10);
  }
  SECTION("whole body in one")
  {
    sink.commit(350);
    CHECK(sink.done());
    CHECK(sink.length() == 0);
  }
  SECTION("zero is a no-op")
  {
    sink.commit(0);
    CHECK(sink.cursor() == 0);
  }
}

TEST_CASE("IoBufFragmentedSink: scatter copies data parts and skips holes", "[io][iobuf_sink]")
{
  std::vector<std::byte> a(10, std::byte{0});
  std::vector<std::byte> b(10, std::byte{0});
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(5);
  sink.appendData(b.data(), b.size());
  std::vector<std::byte> body(25);
  for (std::size_t i = 0; i < body.size(); ++i) {
    body[i] = body_byte(i);
  }
  std::byte* const scratch = IoBufFragmentedSink::holeScratch();
  std::byte const scratch0 = scratch[0];

  sink.scatter(body.data(), 7);       // inside a
  sink.scatter(body.data() + 7, 6);   // a tail + hole head
  sink.scatter(body.data() + 13, 9);  // hole tail + b head
  CHECK(sink.cursor() == 22);
  sink.scatter(body.data() + 22, 3);
  CHECK(sink.done());
  for (std::size_t i = 0; i < 10; ++i) {
    CHECK(a[i] == body[i]);
    CHECK(b[i] == body[15 + i]);
  }
  CHECK(scratch[0] == scratch0);  // the copy path never touches the scratch
}

TEST_CASE("IoBufFragmentedSink: reset and clear", "[io][iobuf_sink]")
{
  std::array<std::byte, 10> a{};
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(5);
  sink.commit(15);
  CHECK(sink.done());
  sink.reset();
  CHECK_FALSE(sink.done());
  CHECK(sink.cursor() == 0);
  CHECK(sink.countChainElements() == 2);
  CHECK(sink.length() == 10);
  sink.commit(12);
  sink.clear();
  CHECK(sink.empty());
  CHECK(sink.done());
  CHECK(sink.cursor() == 0);
  CHECK(sink.computeChainDataLength() == 0);
}

// -------------------------------------------------------------------------------------------------
// Randomised model check against a byte-indexed reference
// -------------------------------------------------------------------------------------------------

TEST_CASE("IoBufFragmentedSink: randomised fills/commits/scatters match a byte reference",
          "[io][iobuf_sink]")
{
  std::mt19937_64 rng(0xC0FFEEu);
  auto uniform = [&](std::size_t lo, std::size_t hi) {
    return std::uniform_int_distribution<std::size_t>(lo, hi)(rng);
  };
  scratch_guard scratch(uniform(1, 70000));
  std::byte* const s_lo = scratch.data();
  std::byte* const s_hi = scratch.data() + scratch.size();

  std::vector<std::byte> staging(std::size_t{2} << 20);
  std::vector<iovec> iov(64);

  for (int round = 0; round < 24; ++round) {
    struct ref_segment {
      std::vector<std::byte> data;  // empty for holes
      std::vector<std::uint8_t> hits;
      std::size_t body_offset;
      std::size_t length;
    };
    std::vector<ref_segment> ref;
    auto sink                = IoBufFragmentedSink::create();
    std::size_t const nsegs  = uniform(1, 6);
    std::size_t const maxlen = round % 3 == 0 ? 64 : (std::size_t{1} << 20);
    std::size_t body         = 0;
    for (std::size_t i = 0; i < nsegs; ++i) {
      std::size_t const len = uniform(1, maxlen);
      bool const hole       = uniform(0, 2) == 0;
      ref_segment r{{}, {}, body, len};
      if (hole) {
        sink.appendHole(len);
      } else {
        r.data.assign(len, std::byte{0xAB});
        r.hits.assign(len, 0);
        sink.appendData(r.data.data(), len);
      }
      ref.push_back(std::move(r));
      body += len;
    }
    REQUIRE(sink.computeChainDataLength() == body);

    // Map a destination pointer back to its reference segment and check it is the right place.
    auto record = [&](std::byte* p, std::size_t len, std::size_t body_offset) {
      if (p >= s_lo && p < s_hi) {
        REQUIRE(p + len <= s_hi);
        return;
      }
      for (auto& r : ref) {
        if (!r.data.empty() && p >= r.data.data() && p < r.data.data() + r.length) {
          auto const at = static_cast<std::size_t>(p - r.data.data());
          REQUIRE(at + len <= r.length);
          REQUIRE(r.body_offset + at == body_offset);
          for (std::size_t j = 0; j < len; ++j) {
            ++r.hits[at + j];
          }
          return;
        }
      }
      FAIL("iovec points outside the data segments and the scratch");
    };

    while (!sink.done()) {
      std::size_t const remaining = sink.remaining();
      if (uniform(0, 7) == 0) {
        // Copy path: a slice of the body that arrived in a foreign buffer.
        std::size_t const n = uniform(1, std::min(remaining, staging.size()));
        for (std::size_t j = 0; j < n; ++j) {
          staging[j] = body_byte(sink.cursor() + j);
        }
        sink.scatter(staging.data(), n);
        // Account for the scattered data bytes in the reference.
        std::size_t const from = sink.cursor() - n;
        for (auto& r : ref) {
          if (r.data.empty()) { continue; }
          std::size_t const lo = std::max(from, r.body_offset);
          std::size_t const hi = std::min(from + n, r.body_offset + r.length);
          for (std::size_t k = lo; k < hi; ++k) {
            ++r.hits[k - r.body_offset];
          }
        }
        continue;
      }
      std::size_t const max_iov = uniform(0, 3) == 0 ? 64 : uniform(1, 4);
      std::size_t const caps[]  = {64, 4096, std::size_t{1} << 21};
      std::size_t const max_b   = uniform(1, caps[uniform(0, 2)]);
      Fill const fill           = sink.fillIovecs(iov.data(), max_iov, max_b);
      REQUIRE(fill.iov_count >= 1);
      REQUIRE(fill.iov_count <= max_iov);
      REQUIRE(fill.bytes <= max_b);
      REQUIRE(sum_lengths(iov.data(), fill.iov_count) == fill.bytes);
      if (fill.iov_count < max_iov) { REQUIRE(fill.bytes == std::min(max_b, remaining)); }
      std::size_t const got = uniform(1, fill.bytes);  // short receive
      std::size_t left      = got;
      std::size_t at        = sink.cursor();
      for (std::size_t i = 0; i < fill.iov_count && left > 0; ++i) {
        std::size_t const take = std::min(left, iov[i].iov_len);
        record(ptr(iov[i]), take, at);
        at += take;
        left -= take;
      }
      deliver(iov.data(), fill.iov_count, sink.cursor(), got);
      sink.commit(got);
    }

    for (auto const& r : ref) {
      for (std::size_t j = 0; j < r.hits.size(); ++j) {
        if (r.hits[j] != 1 || r.data[j] != body_byte(r.body_offset + j)) {
          FAIL("data byte " << j << " of the segment at body offset " << r.body_offset
                            << " received " << int{r.hits[j]} << " times / wrong value");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
// Real recvmsg over a socketpair
// -------------------------------------------------------------------------------------------------

TEST_CASE("IoBufFragmentedSink: recvmsg over a socketpair scatters data and drops holes",
          "[io][iobuf_sink]")
{
  std::size_t const hole_len = GENERATE(std::size_t{700}, std::size_t{2500});
  scratch_guard scratch(1024);

  std::array<int, 2> fds{-1, -1};
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds.data()) == 0);

  std::vector<std::byte> a(3000, std::byte{0});
  std::vector<std::byte> b(2000, std::byte{0});
  auto sink = IoBufFragmentedSink::wrapBuffer(a.data(), a.size());
  sink.appendHole(hole_len);
  sink.appendData(b.data(), b.size());
  std::size_t const body_len = sink.computeChainDataLength();

  std::vector<std::byte> body(body_len);
  for (std::size_t i = 0; i < body_len; ++i) {
    body[i] = body_byte(i);
  }
  // The first bytes "arrived behind the headers" and take the copy path; the rest is on the wire.
  constexpr std::size_t spilled = 100;
  sink.scatter(body.data(), spilled);
  // Fits in the AF_UNIX socket buffer, so a single blocking send is enough.
  REQUIRE(::send(fds[0], body.data() + spilled, body.size() - spilled, 0) ==
          static_cast<ssize_t>(body.size() - spilled));

  std::array<iovec, 2> iov{};
  while (!sink.done()) {
    Fill const fill = sink.fillIovecs(iov.data(), iov.size(), 333);
    REQUIRE(fill.iov_count > 0);
    msghdr msg{};
    msg.msg_iov     = iov.data();
    msg.msg_iovlen  = fill.iov_count;
    ssize_t const n = ::recvmsg(fds[1], &msg, 0);
    REQUIRE(n > 0);
    sink.commit(static_cast<std::size_t>(n));
  }
  ::close(fds[0]);
  ::close(fds[1]);

  CHECK(std::equal(a.begin(), a.end(), body.begin()));
  CHECK(
    std::equal(b.begin(), b.end(), body.begin() + static_cast<std::ptrdiff_t>(3000 + hole_len)));
  // Each scratch byte holds the last hole byte received at its position (hole offset mod size).
  for (std::size_t i = 0; i < scratch.size(); ++i) {
    if (i >= hole_len) {
      CHECK(scratch.data()[i] == std::byte{0xEE});
      continue;
    }
    std::size_t last = i;
    while (last + scratch.size() < hole_len) {
      last += scratch.size();
    }
    if (scratch.data()[i] != body[3000 + last]) {
      FAIL("scratch byte " << i << " is not hole byte " << last);
    }
  }
}
