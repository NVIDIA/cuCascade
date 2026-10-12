#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using resource = cucascade::memory::small_pinned_host_memory_resource;
constexpr std::size_t cache_cap = 512 * 1024;
constexpr std::size_t iterations = 1000;
constexpr std::size_t worker_count = 8;
constexpr std::array<std::size_t, 6> sizes{9000, 16384, 16385, 65000, 100000, 300000};

void cuda_ok(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error{cudaGetErrorString(status)}; }
}

class stream_owner {
 public:
  stream_owner() { cuda_ok(::cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking)); }
  stream_owner(stream_owner const&) = delete;
  stream_owner& operator=(stream_owner const&) = delete;
  stream_owner(stream_owner&&) = delete;
  stream_owner& operator=(stream_owner&&) = delete;
  ~stream_owner()
  {
    (void)::cudaStreamSynchronize(handle_);
    (void)::cudaStreamDestroy(handle_);
  }
  ::cuda::stream_ref get() const { return ::cuda::stream_ref{handle_}; }
  void synchronize() const { cuda_ok(::cudaStreamSynchronize(handle_)); }

 private:
  cudaStream_t handle_{};
};

class host_allocation {
 public:
  host_allocation(resource& mr, ::cuda::stream_ref stream, std::size_t bytes)
    : mr_(mr), stream_(stream), bytes_(bytes), pointer_(mr.allocate(stream, bytes))
  {
  }
  host_allocation(host_allocation const&) = delete;
  host_allocation& operator=(host_allocation const&) = delete;
  host_allocation(host_allocation&&) = delete;
  host_allocation& operator=(host_allocation&&) = delete;
  ~host_allocation() { mr_.deallocate(stream_, pointer_, bytes_); }
  void* get() const { return pointer_; }

 private:
  resource& mr_;
  ::cuda::stream_ref stream_;
  std::size_t bytes_;
  void* pointer_;
};

}  // namespace

int main()
{
  try {
    cuda_ok(::cudaSetDevice(0));
    rmm::mr::pinned_host_memory_resource pinned;
    cucascade::memory::fixed_size_host_memory_resource upstream{
      0, pinned, 16 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024, 16, 1};
    resource mr{upstream, cache_cap};
    std::atomic<bool> begin{false};
    std::mutex state_mutex;
    std::set<void*> live;
    std::exception_ptr failure;
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker] {
        try {
          cuda_ok(::cudaSetDevice(0));
          stream_owner stream;
          begin.wait(false);
          for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            auto const bytes = sizes[(iteration + worker) % sizes.size()];
            host_allocation allocation{mr, stream.get(), bytes};
            stream.synchronize();
            {
              std::lock_guard lock(state_mutex);
              if (!live.insert(allocation.get()).second) {
                throw std::runtime_error{"allocator returned an already-live pointer"};
              }
            }
            auto const pattern = static_cast<unsigned char>((worker + iteration) & 0xFF);
            std::memset(allocation.get(), pattern, bytes);
            auto const* data = static_cast<unsigned char const*>(allocation.get());
            bool const valid = data[0] == pattern && data[bytes - 1] == pattern;
            {
              std::lock_guard lock(state_mutex);
              live.erase(allocation.get());
            }
            if (!valid) { throw std::runtime_error{"host data changed while allocation was live"}; }
            if (mr.large_cache_bytes() > cache_cap) {
              throw std::runtime_error{"cache exceeded its configured cap"};
            }
          }
          stream.synchronize();
        } catch (...) {
          std::lock_guard lock(state_mutex);
          if (!failure) { failure = std::current_exception(); }
        }
      });
    }
    begin.store(true);
    begin.notify_all();
    for (auto& worker : workers) {
      worker.join();
    }
    if (failure) { std::rethrow_exception(failure); }
    if (!live.empty()) { throw std::runtime_error{"live-pointer tracker was not drained"}; }
    std::printf("PASS %zu threads x %zu large-cache cycles; cap=%zu bytes, retained=%zu bytes\n",
                worker_count,
                iterations,
                cache_cap,
                mr.large_cache_bytes());
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
