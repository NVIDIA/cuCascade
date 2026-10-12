#include "fault_control.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace {
constexpr unsigned operation_count = operation_id(fault_operation::count);
struct operation_state {
  std::atomic<unsigned> remaining{};
  std::atomic<int> error{};
  std::atomic<std::size_t> calls{};
  std::atomic<std::size_t> hits{};
};
std::array<operation_state, operation_count> operations;
std::atomic<void*> watched_pointer{};
std::atomic<std::size_t> watched_frees{};
std::atomic<unsigned> last_host_flags{};

cudaError_t injected(fault_operation operation) noexcept
{
  auto& state = operations[operation_id(operation)];
  ++state.calls;
  unsigned remaining = state.remaining.load();
  while (remaining != 0) {
    if (state.remaining.compare_exchange_weak(remaining, remaining - 1)) {
      ++state.hits;
      return static_cast<cudaError_t>(state.error.load());
    }
  }
  return cudaSuccess;
}

template <typename Function>
Function original(char const* name)
{
  void* symbol = dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    std::fprintf(stderr, "missing original CUDA symbol: %s\n", name);
    std::abort();
  }
  return reinterpret_cast<Function>(symbol);
}
}  // namespace

extern "C" void pinned_fault_reset() noexcept
{
  for (auto& state : operations) {
    state.remaining.store(0);
    state.error.store(static_cast<int>(cudaErrorUnknown));
    state.calls.store(0);
    state.hits.store(0);
  }
  watched_pointer.store(nullptr);
  watched_frees.store(0);
  last_host_flags.store(0);
}

extern "C" void pinned_fault_arm(unsigned operation, int error, unsigned failures) noexcept
{
  operations[operation].error.store(error);
  operations[operation].remaining.store(failures);
}

extern "C" void pinned_fault_watch(void* pointer) noexcept
{
  watched_pointer.store(pointer);
  watched_frees.store(0);
}

extern "C" std::size_t pinned_fault_calls(unsigned operation) noexcept
{
  return operations[operation].calls.load();
}

extern "C" std::size_t pinned_fault_hits(unsigned operation) noexcept
{
  return operations[operation].hits.load();
}

extern "C" std::size_t pinned_fault_watched_frees() noexcept
{
  return watched_frees.load();
}

extern "C" unsigned pinned_fault_last_host_flags() noexcept
{
  return last_host_flags.load();
}

extern "C" cudaError_t cudaHostAlloc(void** pointer, std::size_t bytes, unsigned flags)
{
  last_host_flags.store(flags);
  auto const status = injected(fault_operation::host_allocate);
  if (status != cudaSuccess) { return status; }
  static auto function = original<decltype(&cudaHostAlloc)>("cudaHostAlloc");
  return function(pointer, bytes, flags);
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags)
{
  auto const status = injected(fault_operation::event_create);
  if (status != cudaSuccess) { return status; }
  static auto function = original<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags");
  return function(event, flags);
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  auto const status = injected(fault_operation::event_record);
  if (status != cudaSuccess) { return status; }
  static auto function = original<decltype(&cudaEventRecord)>("cudaEventRecord");
  return function(event, stream);
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags)
{
  auto const status = injected(fault_operation::event_wait);
  if (status != cudaSuccess) { return status; }
  static auto function = original<decltype(&cudaStreamWaitEvent)>("cudaStreamWaitEvent");
  return function(stream, event, flags);
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  auto const status = injected(fault_operation::stream_synchronize);
  if (status != cudaSuccess) { return status; }
  static auto function = original<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize");
  return function(stream);
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
  auto const status = injected(fault_operation::event_synchronize);
  if (status != cudaSuccess) { return status; }
  static auto function = original<decltype(&cudaEventSynchronize)>("cudaEventSynchronize");
  return function(event);
}

extern "C" cudaError_t cudaFreeHost(void* pointer)
{
  (void)injected(fault_operation::host_free);
  if (pointer == watched_pointer.load()) { ++watched_frees; }
  static auto function = original<decltype(&cudaFreeHost)>("cudaFreeHost");
  return function(pointer);
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event)
{
  (void)injected(fault_operation::event_destroy);
  static auto function = original<decltype(&cudaEventDestroy)>("cudaEventDestroy");
  return function(event);
}
