#include "fault_control.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <string_view>
#include <thread>

namespace {
thread_local int new_failure_after = -1;
thread_local unsigned new_failure_hits = 0;

void check(bool condition, char const* message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(2);
  }
}

void cuda_ok(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA setup/cleanup failed: %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(3);
  }
}

void fail_new_after(int successes) noexcept
{
  new_failure_hits = 0;
  new_failure_after = successes;
}

void disarm_new() noexcept { new_failure_after = -1; }

void fail_once(fault_operation operation, cudaError_t status)
{
  pinned_fault_arm(operation_id(operation), static_cast<int>(status), 1);
}

void require_hit(fault_operation operation)
{
  check(pinned_fault_hits(operation_id(operation)) == 1, "expected CUDA fault was not injected exactly once");
}

using resource = cucascade::memory::small_pinned_host_memory_resource;
constexpr std::size_t request = 9000;
constexpr std::size_t bucket = 16384;

struct fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 16 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024, 16, 1};
  cudaStream_t handle{};

  fixture() { cuda_ok(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "stream create"); }
  ~fixture() { cuda_ok(cudaStreamDestroy(handle), "stream destroy"); }
  ::cuda::stream_ref stream() const { return ::cuda::stream_ref{handle}; }
  void drain() const { cuda_ok(cudaStreamSynchronize(handle), "stream drain"); }
};

void wait_failure_keeps_entry(bool slab)
{
  fixture f;
  resource mr{f.upstream};
  std::size_t const bytes = slab ? 512 : request;
  auto* pointer = mr.allocate(f.stream(), bytes);
  mr.deallocate(f.stream(), pointer, bytes);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  auto const retained = mr.large_cache_bytes();
  fail_once(fault_operation::event_wait, cudaErrorInvalidResourceHandle);
  bool threw = false;
  try {
    (void)mr.allocate(f.stream(), bytes);
  } catch (std::exception const&) {
    threw = true;
  }
  check(threw, "failed wait did not propagate an allocation error");
  require_hit(fault_operation::event_wait);
  check(mr.large_cache_bytes() == retained, "failed wait consumed cache accounting");
  check(pinned_fault_watched_frees() == 0, "failed wait freed candidate");
  auto* retry = mr.allocate(f.stream(), bytes);
  check(retry == pointer, "failed wait did not preserve candidate for retry");
  mr.deallocate(f.stream(), retry, bytes);
  f.drain();
}

void event_fallback_sync_failure(fault_operation event_operation, bool slab)
{
  fixture f;
  auto const free_blocks_before = f.upstream.get_free_blocks();
  auto mr = std::make_unique<resource>(f.upstream);
  std::size_t const bytes = slab ? 512 : request;
  auto* pointer = mr->allocate(f.stream(), bytes);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  fail_once(event_operation, cudaErrorUnknown);
  fail_once(fault_operation::stream_synchronize, cudaErrorUnknown);
  mr->deallocate(f.stream(), pointer, bytes);
  require_hit(event_operation);
  require_hit(fault_operation::stream_synchronize);
  check(pinned_fault_watched_frees() == 0, "unknown stream completion freed storage");
  check(mr->large_cache_bytes() == 0, "unknown stream completion published large storage");
  if (slab) {
    auto* next = mr->allocate(f.stream(), bytes);
    check(next != pointer, "quarantined slab was republished");
    mr->deallocate(f.stream(), next, bytes);
  }
  f.drain();
  mr.reset();
  check(pinned_fault_watched_frees() == 0, "destruction freed quarantined large storage");
  if (slab) {
    check(f.upstream.get_free_blocks() + 1 == free_blocks_before,
          "slab quarantine returned backing block upstream");
  } else {
    // The injected error deliberately leaves an allocation retained. No work was submitted,
    // and the real drain above succeeded, so the harness can reclaim it after observing safety.
    cuda_ok(cudaFreeHost(pointer), "reclaim intentionally quarantined large pointer");
  }
}

void eviction_sync_failure()
{
  fixture f;
  auto mr = std::make_unique<resource>(f.upstream, bucket);
  auto* victim = mr->allocate(f.stream(), request);
  auto* successor = mr->allocate(f.stream(), request);
  mr->deallocate(f.stream(), victim, request);
  pinned_fault_reset();
  pinned_fault_watch(victim);
  fail_once(fault_operation::event_synchronize, cudaErrorUnknown);
  mr->deallocate(f.stream(), successor, request);
  require_hit(fault_operation::event_synchronize);
  check(pinned_fault_watched_frees() == 0, "failed victim sync freed evicted storage");
  check(mr->large_cache_bytes() == bucket, "eviction cache accounting is inconsistent");
  f.drain();
  mr.reset();
  check(pinned_fault_watched_frees() == 0, "destruction freed failed eviction victim");
  cuda_ok(cudaFreeHost(victim), "reclaim failed eviction victim");
}

void purge_sync_failure()
{
  fixture f;
  auto mr = std::make_unique<resource>(f.upstream, 2 * bucket);
  auto* victim = mr->allocate(f.stream(), request);
  mr->deallocate(f.stream(), victim, request);
  pinned_fault_reset();
  pinned_fault_watch(victim);
  fail_once(fault_operation::host_allocate, cudaErrorMemoryAllocation);
  fail_once(fault_operation::event_synchronize, cudaErrorUnknown);
  auto* next = mr->allocate(f.stream(), 20000);
  require_hit(fault_operation::host_allocate);
  require_hit(fault_operation::event_synchronize);
  check(next != nullptr, "allocation retry failed after failed victim sync");
  check(pinned_fault_watched_frees() == 0, "failed purge sync freed unsafe storage");
  check(mr->large_cache_bytes() == 0, "purge did not detach cache entries");
  mr->deallocate(f.stream(), next, 20000);
  f.drain();
  mr.reset();
  check(pinned_fault_watched_frees() == 0, "destruction freed failed purge victim");
  cuda_ok(cudaFreeHost(victim), "reclaim failed purge victim");
}

void destructor_sync_failure()
{
  fixture f;
  auto mr = std::make_unique<resource>(f.upstream);
  auto* pointer = mr->allocate(f.stream(), request);
  mr->deallocate(f.stream(), pointer, request);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  fail_once(fault_operation::event_synchronize, cudaErrorUnknown);
  mr.reset();
  require_hit(fault_operation::event_synchronize);
  check(pinned_fault_watched_frees() == 0, "failed destructor sync freed storage");
  f.drain();
  cuda_ok(cudaFreeHost(pointer), "reclaim failed destructor victim");
}

void host_oom_recovery(bool metadata_oom, bool fail_retry)
{
  fixture f;
  resource mr{f.upstream};
  auto* victim = mr.allocate(f.stream(), request);
  mr.deallocate(f.stream(), victim, request);
  pinned_fault_reset();
  pinned_fault_watch(victim);
  pinned_fault_arm(operation_id(fault_operation::host_allocate),
                   static_cast<int>(cudaErrorMemoryAllocation), fail_retry ? 2U : 1U);
  if (metadata_oom) { fail_new_after(0); }
  void* result = nullptr;
  bool threw = false;
  try {
    result = mr.allocate(f.stream(), 20000);
  } catch (std::bad_alloc const&) {
    threw = true;
  }
  disarm_new();
  check(pinned_fault_calls(operation_id(fault_operation::host_allocate)) == 2,
        "host allocation recovery did not make exactly one retry");
  check(pinned_fault_watched_frees() == 1, "recovery failed to free retained pinned storage");
  check(mr.large_cache_bytes() == 0, "recovery left stale cache accounting");
  check(threw == fail_retry, "unexpected allocation exception outcome");
  if (metadata_oom) {
    check(new_failure_hits == 1, "recovery metadata OOM was not injected");
  }
  if (result != nullptr) { mr.deallocate(f.stream(), result, 20000); }
  f.drain();
}

void insertion_metadata_oom(int successful_allocations)
{
  fixture f;
  resource mr{f.upstream};
  auto* pointer = mr.allocate(f.stream(), request);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  fail_new_after(successful_allocations);
  mr.deallocate(f.stream(), pointer, request);
  disarm_new();
  check(new_failure_hits == 1, "cache insertion metadata OOM was not injected");
  check(pinned_fault_watched_frees() == 1, "metadata OOM fallback did not release synchronized pointer");
  check(mr.large_cache_bytes() == 0, "failed cache insertion retained empty/stale bucket");
  // A later insertion/lookup must remain valid after a partially constructed bucket failed.
  auto* next = mr.allocate(f.stream(), request);
  mr.deallocate(f.stream(), next, request);
  check(mr.large_cache_bytes() == bucket, "cache failed to recover after metadata OOM");
  f.drain();
}

void event_recycling_metadata_oom()
{
  fixture f;
  resource mr{f.upstream};
  auto* pointer = mr.allocate(f.stream(), request);
  mr.deallocate(f.stream(), pointer, request);
  pinned_fault_reset();
  fail_new_after(0);
  auto* next = mr.allocate(f.stream(), request);
  disarm_new();
  check(new_failure_hits == 1, "event-pool metadata OOM was not injected");
  check(next == pointer, "event recycling OOM lost live allocation");
  check(mr.large_cache_bytes() == 0, "event recycling OOM left stale cache accounting");
  mr.deallocate(f.stream(), next, request);
  f.drain();
}

void record_failure_and_metadata_oom()
{
  fixture f;
  resource mr{f.upstream};
  auto* pointer = mr.allocate(f.stream(), request);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  fail_once(fault_operation::event_record, cudaErrorUnknown);
  fail_new_after(0);
  mr.deallocate(f.stream(), pointer, request);
  disarm_new();
  require_hit(fault_operation::event_record);
  check(new_failure_hits == 1, "failed-record event-pool metadata OOM was not injected");
  check(pinned_fault_watched_frees() == 1, "failed-record fallback did not release synchronized pointer");
  check(mr.large_cache_bytes() == 0, "failed-record fallback cached storage");
  f.drain();
}

void uncached_sync_failure()
{
  fixture f;
  resource mr{f.upstream, 0};
  auto* pointer = mr.allocate(f.stream(), request);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  fail_once(fault_operation::stream_synchronize, cudaErrorUnknown);
  mr.deallocate(f.stream(), pointer, request);
  require_hit(fault_operation::stream_synchronize);
  check(pinned_fault_watched_frees() == 0, "uncached synchronization failure freed storage");
  f.drain();
  cuda_ok(cudaFreeHost(pointer), "reclaim uncached synchronization failure pointer");
}

void mapped_portable_flags()
{
  fixture f;
  resource mr{f.upstream};
  pinned_fault_reset();
  auto* pointer = mr.allocate(f.stream(), request);
  auto const requested_flags = pinned_fault_last_host_flags();
  check(requested_flags == (cudaHostAllocPortable | cudaHostAllocMapped),
        "actual allocator cudaHostAlloc call omitted portable/mapped flags");
  auto* bytes = static_cast<unsigned char*>(pointer);
  bytes[0] = 0x5a;
  check(bytes[0] == 0x5a, "actual allocation is not host-accessible");
  std::printf("observed cudaHostAlloc requested flags=%u\n", requested_flags);
  mr.deallocate(f.stream(), pointer, request);
  f.drain();
}

void pending_dma_destructor()
{
  fixture f;
  auto mr = std::make_unique<resource>(f.upstream);
  auto* pointer = mr->allocate(f.stream(), request);
  static_cast<unsigned char*>(pointer)[0] = 0x6b;
  void* device = nullptr;
  cuda_ok(cudaMalloc(&device, request), "teardown device allocation");
  struct gate_state {
    std::atomic<bool> release{false};
    std::atomic<bool> destructor_finished{false};
    std::atomic<bool> observed_wait{false};
  } gate;
  cuda_ok(cudaLaunchHostFunc(f.handle, [](void* context) {
    auto& state = *static_cast<gate_state*>(context);
    while (!state.release.load()) { std::this_thread::yield(); }
  }, &gate), "teardown stream gate");
  cuda_ok(cudaMemcpyAsync(device, pointer, request, cudaMemcpyHostToDevice, f.handle),
          "teardown pending host read");
  mr->deallocate(f.stream(), pointer, request);
  pinned_fault_reset();
  pinned_fault_watch(pointer);
  std::jthread release_thread{[&gate] {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (std::chrono::steady_clock::now() < deadline) {
      if (pinned_fault_calls(operation_id(fault_operation::event_synchronize)) != 0 &&
          !gate.destructor_finished.load()) {
        gate.observed_wait.store(true);
        break;
      }
      if (gate.destructor_finished.load()) { break; }
      std::this_thread::yield();
    }
    gate.release.store(true);
  }};
  mr.reset();
  gate.destructor_finished.store(true);
  release_thread.join();
  check(gate.observed_wait.load(), "destructor did not wait before releasing pending DMA source");
  check(pinned_fault_watched_frees() == 1, "successful destructor did not free completed storage once");
  f.drain();
  unsigned char observed = 0;
  cuda_ok(cudaMemcpy(&observed, device, 1, cudaMemcpyDeviceToHost), "teardown data verification");
  check(observed == 0x6b, "teardown corrupted queued DMA source");
  cuda_ok(cudaFree(device), "teardown device release");
}
}  // namespace

void* operator new(std::size_t bytes)
{
  if (new_failure_after == 0) {
    new_failure_after = -1;
    ++new_failure_hits;
    throw std::bad_alloc{};
  }
  if (new_failure_after > 0) { --new_failure_after; }
  if (auto* pointer = std::malloc(bytes == 0 ? 1 : bytes)) { return pointer; }
  throw std::bad_alloc{};
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

int main(int argc, char** argv)
{
  cuda_ok(cudaSetDevice(0), "set device");
  cuda_ok(cudaFree(nullptr), "initialize CUDA runtime");
  struct test_case {
    char const* name;
    void (*run)();
  };
  std::array const cases{
    test_case{"large_wait_failure", [] { wait_failure_keeps_entry(false); }},
    test_case{"slab_wait_failure", [] { wait_failure_keeps_entry(true); }},
    test_case{"large_create_and_sync_failure", [] { event_fallback_sync_failure(fault_operation::event_create, false); }},
    test_case{"large_record_and_sync_failure", [] { event_fallback_sync_failure(fault_operation::event_record, false); }},
    test_case{"slab_create_and_sync_failure", [] { event_fallback_sync_failure(fault_operation::event_create, true); }},
    test_case{"slab_record_and_sync_failure", [] { event_fallback_sync_failure(fault_operation::event_record, true); }},
    test_case{"eviction_sync_failure", eviction_sync_failure},
    test_case{"purge_sync_failure", purge_sync_failure},
    test_case{"destructor_sync_failure", destructor_sync_failure},
    test_case{"host_oom_recovery", [] { host_oom_recovery(false, false); }},
    test_case{"host_oom_recovery_with_metadata_oom", [] { host_oom_recovery(true, false); }},
    test_case{"host_oom_retry_failure", [] { host_oom_recovery(false, true); }},
    test_case{"metadata_oom_first_allocation", [] { insertion_metadata_oom(0); }},
    test_case{"metadata_oom_second_allocation", [] { insertion_metadata_oom(1); }},
    test_case{"metadata_oom_third_allocation", [] { insertion_metadata_oom(2); }},
    test_case{"event_recycling_metadata_oom", event_recycling_metadata_oom},
    test_case{"record_failure_and_metadata_oom", record_failure_and_metadata_oom},
    test_case{"uncached_sync_failure", uncached_sync_failure},
    test_case{"mapped_portable_flags", mapped_portable_flags},
    test_case{"pending_dma_destructor", pending_dma_destructor}
  };
  unsigned executed = 0;
  for (auto const& test : cases) {
    if (argc > 1 && std::string_view{argv[1]} != test.name) { continue; }
    pinned_fault_reset();
    test.run();
    disarm_new();
    std::printf("PASS %s\n", test.name);
    std::fflush(stdout);
    ++executed;
  }
  check(executed != 0, "unknown selected fault case");
  std::printf("%u fault cases passed\n", executed);
}
