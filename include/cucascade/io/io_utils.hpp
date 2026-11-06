#pragma once

#include <cucascade/io/byte_range.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <vector>

namespace io_utils {

using range = cucascade::io::byte_range;

//===----------------------------------------------------------------------===//
// Overflow-safe offset arithmetic
//===----------------------------------------------------------------------===//
//
// A range's offset and size are signed 64-bit, so a planned range must end at or
// before INT64_MAX.  No file byte lies there or beyond (file offsets are signed
// 64-bit too), so clamping an end to it never drops a byte a file can hold.

/// Largest end (offset + size) a @c range can represent.
inline constexpr size_t max_range_end = static_cast<size_t>(std::numeric_limits<int64_t>::max());

/// @p lhs + @p rhs, or @c SIZE_MAX when the sum does not fit.
[[nodiscard]] constexpr size_t saturating_add(size_t lhs, size_t rhs) noexcept
{
  return rhs > std::numeric_limits<size_t>::max() - lhs ? std::numeric_limits<size_t>::max()
                                                        : lhs + rhs;
}

/// @p value rounded down to a multiple of @p alignment (0 leaves it unchanged).
[[nodiscard]] constexpr size_t align_down(size_t value, size_t alignment) noexcept
{
  return alignment == 0 ? value : value - value % alignment;
}

/// @p value rounded up to a multiple of @p alignment (0 leaves it unchanged), or
/// @c SIZE_MAX when that multiple does not fit.
[[nodiscard]] constexpr size_t align_up(size_t value, size_t alignment) noexcept
{
  if (alignment == 0) return value;
  auto const remainder = value % alignment;
  return remainder == 0 ? value : saturating_add(value, alignment - remainder);
}

/// True when @p r names at least one byte a file can hold: a non-negative offset below
/// @c max_range_end and a positive size.  The helpers below drop every other range.
[[nodiscard]] constexpr bool is_file_range(range const& r) noexcept
{
  return r.offset() >= 0 && r.offset() < std::numeric_limits<int64_t>::max() && r.size() > 0;
}

/// End of a range accepted by @c is_file_range, clamped to @c max_range_end.
[[nodiscard]] constexpr size_t clamped_end(range const& r) noexcept
{
  return std::min(saturating_add(static_cast<size_t>(r.offset()), static_cast<size_t>(r.size())),
                  max_range_end);
}

/**
 * @brief Expands each range outward so its start is aligned down and its end
 *        is aligned up to @p alignment bytes.
 *
 * Ranges are processed independently; overlaps introduced by alignment are not
 * merged here — call @c coalesce afterwards if desired.
 *
 * Ranges rejected by @c is_file_range (negative offset, no bytes) are dropped.
 * Ends are clamped to @c max_range_end, so an end there may stay unaligned.
 *
 * @param ranges    Input ranges (any order).
 * @param alignment Alignment in bytes; 0 or 1 leaves the ends where they are.
 * @return One aligned range per kept input range, in the same order.
 */
inline std::vector<range> align_ranges(std::span<range const> ranges, size_t alignment)
{
  std::vector<range> out;
  out.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (!is_file_range(r)) continue;
    auto const align_start = align_down(static_cast<size_t>(r.offset()), alignment);
    auto const align_end   = std::min(align_up(clamped_end(r), alignment), max_range_end);
    out.emplace_back(static_cast<int64_t>(align_start),
                     static_cast<int64_t>(align_end - align_start));
  }
  return out;
}

/**
 * @brief Merges adjacent or near-adjacent ranges of an offset-sorted group.
 *
 * Same as @c coalesce_ranges, for input already sorted by offset.
 *
 * @param sorted_ranges Input ranges, sorted by offset.
 * @param max_gap       Maximum byte gap that still triggers a merge.
 * @return Sorted, non-overlapping, coalesced ranges.
 */
inline std::vector<range> coalesce_sorted_ranges(std::span<range const> sorted_ranges,
                                                 size_t max_gap)
{
  std::vector<range> out;
  size_t cur_start = 0;
  size_t cur_end   = 0;

  for (auto const& r : sorted_ranges) {
    if (!is_file_range(r)) continue;
    auto const next_start = static_cast<size_t>(r.offset());
    auto const next_end   = clamped_end(r);

    if (!out.empty() && next_start <= saturating_add(cur_end, max_gap)) {
      cur_end = std::max(cur_end, next_end);
      out.back() =
        range{static_cast<int64_t>(cur_start), static_cast<int64_t>(cur_end - cur_start)};
    } else {
      cur_start = next_start;
      cur_end   = next_end;
      out.emplace_back(static_cast<int64_t>(cur_start), static_cast<int64_t>(cur_end - cur_start));
    }
  }
  return out;
}

/**
 * @brief Merges adjacent or near-adjacent ranges into fewer, larger ranges.
 *
 * Two consecutive ranges (sorted by offset) are merged when the gap between
 * them is at most @p max_gap bytes.  Overlapping ranges are always merged.
 * Ranges rejected by @c is_file_range are dropped and ends are clamped to
 * @c max_range_end.
 *
 * @param ranges  Input ranges (need not be sorted).
 * @param max_gap Maximum byte gap that still triggers a merge.
 * @return Sorted, non-overlapping, coalesced ranges.
 */
inline std::vector<range> coalesce_ranges(std::span<range const> ranges, size_t max_gap)
{
  std::vector<range> sorted(ranges.begin(), ranges.end());
  std::sort(sorted.begin(), sorted.end(), [](range const& a, range const& b) {
    return a.offset() < b.offset();
  });
  return coalesce_sorted_ranges(sorted, max_gap);
}

/**
 * @brief Aligns then coalesces ranges in one step.
 *
 * Equivalent to @c coalesce_ranges(align_ranges(ranges, alignment), max_gap).
 * After alignment neighbouring ranges often become adjacent or overlapping, so
 * the coalescing pass is most useful when called together with alignment.
 *
 * @param ranges    Input ranges.
 * @param alignment Alignment in bytes; 0 or 1 leaves the ends where they are.
 * @param max_gap   Maximum byte gap between aligned ranges to still merge.
 * @return Sorted, aligned, coalesced ranges.
 */
inline std::vector<range> align_and_coalesce(std::span<range const> ranges,
                                             size_t alignment,
                                             size_t max_gap = 0)
{
  auto aligned = align_ranges(ranges, alignment);
  std::sort(aligned.begin(), aligned.end(), [](range const& a, range const& b) {
    return a.offset() < b.offset();
  });
  return coalesce_sorted_ranges(aligned, max_gap);
}

/**
 * @brief Aligns then coalesces offset-sorted ranges in one step.
 *
 * Equivalent to @c coalesce_sorted_ranges(align_ranges(ranges, alignment), max_gap).
 * After alignment neighbouring ranges often become adjacent or overlapping, so
 * the coalescing pass is most useful when called together with alignment.
 *
 * @param ranges    Input ranges, sorted by offset.
 * @param alignment Alignment in bytes; 0 or 1 leaves the ends where they are.
 * @param max_gap   Maximum byte gap between aligned ranges to still merge.
 * @return Sorted, aligned, coalesced ranges.
 */
inline std::vector<range> align_and_coalesce_sorted(std::span<range const> ranges,
                                                    size_t alignment,
                                                    size_t max_gap = 0)
{
  return coalesce_sorted_ranges(align_ranges(ranges, alignment), max_gap);
}

/**
 * @brief Merges two sorted range groups into a single sorted, coalesced group.
 *
 * @param a       First sorted range group.
 * @param b       Second sorted range group.
 * @param max_gap Maximum byte gap between ranges to still merge.
 * @return Sorted, coalesced union of both groups.
 */
inline std::vector<range> union_sorted_ranges(std::span<range const> a,
                                              std::span<range const> b,
                                              size_t max_gap = 0)
{
  std::vector<range> merged;
  merged.reserve(a.size() + b.size());
  std::merge(a.begin(),
             a.end(),
             b.begin(),
             b.end(),
             std::back_inserter(merged),
             [](range const& x, range const& y) { return x.offset() < y.offset(); });
  return coalesce_sorted_ranges(merged, max_gap);
}

/**
 * @brief Returns the byte ranges covered by both @p a and @p b.
 *
 * Partial overlaps are included — only the overlapping portion is kept.
 * Both inputs must be sorted by offset and non-overlapping within themselves.
 *
 * @param a First sorted range group.
 * @param b Second sorted range group.
 * @return Sorted ranges present in both groups (intersection).
 */
inline std::vector<range> common_sorted_ranges(std::span<range const> a, std::span<range const> b)
{
  std::vector<range> out;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    size_t a_start = static_cast<size_t>(a[i].offset());
    size_t a_end   = a_start + static_cast<size_t>(a[i].size());
    size_t b_start = static_cast<size_t>(b[j].offset());
    size_t b_end   = b_start + static_cast<size_t>(b[j].size());

    size_t ovlp_start = std::max(a_start, b_start);
    size_t ovlp_end   = std::min(a_end, b_end);
    if (ovlp_start < ovlp_end)
      out.emplace_back(static_cast<int64_t>(ovlp_start),
                       static_cast<int64_t>(ovlp_end - ovlp_start));

    if (a_end < b_end)
      ++i;
    else if (b_end < a_end)
      ++j;
    else {
      ++i;
      ++j;
    }
  }
  return out;
}

/**
 * @brief Returns the parts of @p source not covered by @p coverage.
 *
 * Partial coverage is handled: if a coverage range clips only one end of a
 * source range, the uncovered portion is still returned.
 * Both inputs must be sorted by offset and non-overlapping within themselves.
 *
 * @param source   Ranges to examine.
 * @param coverage Ranges that are already present / satisfied.
 * @return Sorted sub-ranges of @p source with no coverage.
 */
inline std::vector<range> missing_sorted_ranges(std::span<range const> source,
                                                std::span<range const> coverage)
{
  std::vector<range> out;
  size_t j = 0;
  for (auto const& s : source) {
    size_t cur = static_cast<size_t>(s.offset());
    size_t end = cur + static_cast<size_t>(s.size());

    // Advance j past coverage ranges that end before this source range starts.
    while (j < coverage.size() &&
           static_cast<size_t>(coverage[j].offset()) + static_cast<size_t>(coverage[j].size()) <=
             cur)
      ++j;

    // Walk coverage ranges that overlap this source range with a local cursor
    // so j stays valid for the next source range.
    size_t k = j;
    while (k < coverage.size() && static_cast<size_t>(coverage[k].offset()) < end) {
      size_t c_start = static_cast<size_t>(coverage[k].offset());
      size_t c_end   = c_start + static_cast<size_t>(coverage[k].size());

      if (cur < c_start)
        out.emplace_back(static_cast<int64_t>(cur), static_cast<int64_t>(c_start - cur));
      cur = std::max(cur, c_end);
      ++k;
    }

    if (cur < end) out.emplace_back(static_cast<int64_t>(cur), static_cast<int64_t>(end - cur));
  }
  return out;
}

}  // namespace io_utils
