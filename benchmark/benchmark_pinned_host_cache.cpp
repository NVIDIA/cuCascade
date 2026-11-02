/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/error.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv)
{
  try {
    if (argc != 5) {
      throw std::invalid_argument{"Expected mode, threads, iterations, warmup iterations"};
    }
    std::string const mode{argv[1]};
    if (mode != "hits" && mode != "full" && mode != "half") {
      throw std::invalid_argument{"Unknown mode"};
    }
    auto const thread_count        = static_cast<std::size_t>(std::stoull(argv[2]));
    auto const iterations          = static_cast<std::size_t>(std::stoull(argv[3]));
    auto const warmup_iterations   = static_cast<std::size_t>(std::stoull(argv[4]));
    std::size_t const cache_limit  = mode == "half"   ? 256ULL << 20
                                     : mode == "hits" ? 1ULL << 20
                                                      : 64ULL << 10;
    std::size_t const first_bytes  = mode == "half" ? cache_limit / 2 + 1 : 9000;
    std::size_t const second_bytes = mode == "half" ? first_bytes : 40000;
    if (thread_count == 0 || iterations == 0) { throw std::invalid_argument{"Zero work"}; }
    CUCASCADE_CUDA_TRY(cudaSetDevice(0));
    rmm::mr::pinned_host_memory_resource pinned;
    cucascade::memory::fixed_size_host_memory_resource upstream{
      0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
    cucascade::memory::small_pinned_host_memory_resource resource{upstream, cache_limit};
    std::vector<void*> warm;
    for (std::size_t i = 0; i < thread_count; ++i) {
      warm.push_back(resource.allocate_sync(first_bytes));
    }
    for (void* pointer : warm) {
      resource.deallocate_sync(pointer, first_bytes);
    }

    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::exception_ptr> errors(thread_count);
    std::vector<std::jthread> workers;
    workers.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      workers.emplace_back([&, i](std::stop_token stop) {
        bool announced = false;
        try {
          CUCASCADE_CUDA_TRY(cudaSetDevice(0));
          rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
          auto work = [&] {
            auto* first = resource.allocate(stream, first_bytes);
            if (mode != "hits") {
              auto* second = resource.allocate(stream, second_bytes);
              resource.deallocate(stream, first, first_bytes);
              resource.deallocate(stream, second, second_bytes);
            } else {
              resource.deallocate(stream, first, first_bytes);
            }
          };
          for (std::size_t n = 0; n < warmup_iterations; ++n) {
            work();
          }
          stream.synchronize();
          ready.fetch_add(1, std::memory_order_release);
          announced = true;
          while (!start.load(std::memory_order_acquire)) {
            if (stop.stop_requested()) { return; }
            std::this_thread::yield();
          }
          for (std::size_t n = 0; n < iterations; ++n) {
            work();
          }
          stream.synchronize();
        } catch (...) {
          errors[i] = std::current_exception();
          if (!announced) { ready.fetch_add(1, std::memory_order_release); }
        }
      });
    }
    while (ready.load(std::memory_order_acquire) != thread_count) {
      std::this_thread::yield();
    }
    auto const before = resource.get_statistics();
    auto const begin  = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
      worker.join();
    }
    auto const elapsed = std::chrono::steady_clock::now() - begin;
    for (auto const& error : errors) {
      if (error) { std::rethrow_exception(error); }
    }
    auto const nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    auto const operations  = thread_count * iterations * (mode == "hits" ? 1ULL : 2ULL);
    auto const stats       = resource.get_statistics();
    std::cout << std::setprecision(17) << "{\"mode\":\"" << mode
              << "\",\"threads\":" << thread_count << ",\"iterations\":" << iterations
              << ",\"warmup_iterations\":" << warmup_iterations
              << ",\"cache_limit_bytes\":" << cache_limit
              << ",\"first_request_bytes\":" << first_bytes
              << ",\"second_request_bytes\":" << (mode == "hits" ? 0 : second_bytes)
              << ",\"operations\":" << operations << ",\"elapsed_ns\":" << nanoseconds
              << ",\"operations_per_second\":"
              << static_cast<double>(operations) * 1.0e9 / static_cast<double>(nanoseconds)
              << ",\"hits\":" << stats.large_hits - before.large_hits
              << ",\"misses\":" << stats.large_misses - before.large_misses
              << ",\"evictions\":" << stats.large_evictions - before.large_evictions << "}\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
