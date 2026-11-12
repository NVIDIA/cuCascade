/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "row_range_detail.hpp"

#include <cucascade/error.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

namespace cucascade::detail {

namespace {

constexpr unsigned int block_threads = 256;
constexpr std::size_t word_bits      = 32;

unsigned int grid_blocks(std::size_t count)
{
  auto const blocks = (count + block_threads - 1) / block_threads;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("row-range kernel: element count too large for one launch");
  }
  return static_cast<unsigned int>(blocks);
}

/// One thread per destination word in [first_word, first_word + num_words). Each thread is the
/// only writer of its word, so the read-merge-write of the edge words needs no atomics.
__global__ void copy_bits_kernel(std::uint32_t* dst_mask,
                                 std::size_t dst_bit,
                                 const std::uint32_t* src_mask,
                                 std::size_t src_bit,
                                 std::size_t num_bits,
                                 std::size_t first_word,
                                 std::size_t num_words)
{
  auto const tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_words) { return; }

  auto const word       = first_word + tid;
  auto const word_begin = word * word_bits;
  auto const range_end  = dst_bit + num_bits;
  // Part of this word that lies inside the destination range: bits [lo, hi) of the bitmask.
  auto const lo    = word_begin > dst_bit ? word_begin : dst_bit;
  auto const hi    = word_begin + word_bits < range_end ? word_begin + word_bits : range_end;
  auto const n     = static_cast<unsigned int>(hi - lo);
  auto const shift = static_cast<unsigned int>(lo - word_begin);
  auto const mask  = (n == word_bits ? ~std::uint32_t{0} : (std::uint32_t{1} << n) - 1) << shift;

  std::uint32_t bits = ~std::uint32_t{0};
  if (src_mask != nullptr) {
    // Funnel shift: the n bits starting at source bit s span at most two source words.
    auto const s        = src_bit + (lo - dst_bit);
    auto const src_word = s / word_bits;
    auto const src_off  = static_cast<unsigned int>(s % word_bits);
    bits                = src_mask[src_word] >> src_off;
    if (src_off + n > word_bits) { bits |= src_mask[src_word + 1] << (word_bits - src_off); }
  }

  dst_mask[word] = (dst_mask[word] & ~mask) | ((bits << shift) & mask);
}

template <typename Dst, typename Src>
__global__ void rebase_offsets_kernel(Dst* dst,
                                      const Src* src,
                                      std::size_t count,
                                      std::int64_t dst_base)
{
  auto const tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= count) { return; }

  auto const value =
    dst_base + (static_cast<std::int64_t>(src[tid]) - static_cast<std::int64_t>(src[0]));
  // The caller guarantees the value fits the destination width.
  assert(value >= static_cast<std::int64_t>(std::numeric_limits<Dst>::min()) &&
         value <= static_cast<std::int64_t>(std::numeric_limits<Dst>::max()));
  dst[tid] = static_cast<Dst>(value);
}

template <typename Dst, typename Src>
void launch_rebase_offsets(
  void* dst, const void* src, std::size_t count, std::int64_t dst_base, ::cuda::stream_ref stream)
{
  rebase_offsets_kernel<Dst, Src><<<grid_blocks(count), block_threads, 0, stream.get()>>>(
    static_cast<Dst*>(dst), static_cast<const Src*>(src), count, dst_base);
  CUCASCADE_CUDA_TRY(cudaGetLastError());
}

}  // namespace

void copy_bits(std::uint32_t* dst_mask,
               std::size_t dst_bit,
               const std::uint32_t* src_mask,
               std::size_t src_bit,
               std::size_t num_bits,
               ::cuda::stream_ref stream)
{
  if (num_bits == 0) { return; }

  auto const first_word = dst_bit / word_bits;
  auto const last_word  = (dst_bit + num_bits - 1) / word_bits;
  auto const num_words  = last_word - first_word + 1;

  copy_bits_kernel<<<grid_blocks(num_words), block_threads, 0, stream.get()>>>(
    dst_mask, dst_bit, src_mask, src_bit, num_bits, first_word, num_words);
  CUCASCADE_CUDA_TRY(cudaGetLastError());
}

void rebase_offsets(void* dst,
                    std::uint8_t dst_width,
                    const void* src,
                    std::uint8_t src_width,
                    std::size_t count,
                    std::int64_t dst_base,
                    ::cuda::stream_ref stream)
{
  auto const valid = [](std::uint8_t width) { return width == 4 || width == 8; };
  if (!valid(dst_width) || !valid(src_width)) {
    throw std::invalid_argument("rebase_offsets: offset widths must be 4 or 8 bytes, got dst=" +
                                std::to_string(dst_width) + " src=" + std::to_string(src_width));
  }
  if (count == 0) { return; }

  if (dst_width == 4) {
    if (src_width == 4) {
      launch_rebase_offsets<std::int32_t, std::int32_t>(dst, src, count, dst_base, stream);
    } else {
      launch_rebase_offsets<std::int32_t, std::int64_t>(dst, src, count, dst_base, stream);
    }
  } else {
    if (src_width == 4) {
      launch_rebase_offsets<std::int64_t, std::int32_t>(dst, src, count, dst_base, stream);
    } else {
      launch_rebase_offsets<std::int64_t, std::int64_t>(dst, src, count, dst_base, stream);
    }
  }
}

rmm::device_buffer allocate_scratch(memory::reservation* scratch,
                                    std::size_t bytes,
                                    ::cuda::stream_ref stream)
{
  auto const mr = scratch != nullptr ? scratch->get_memory_resource()
                                     : rmm::mr::get_current_device_resource_ref();
  return rmm::device_buffer(bytes, stream, mr);
}

}  // namespace cucascade::detail
