# Testing Patterns

**Analysis Date:** 2026-04-13

## Test Framework

**Runner:**
- Catch2 v2.13.10 (single-header)
- Fetched via CMake `FetchContent` in `test/CMakeLists.txt`
- Config: `test/CMakeLists.txt` (no separate config file)
- CTest integration: single test target `cucascade_tests`

**Assertion Library:**
- Catch2 built-in: `REQUIRE`, `CHECK`, `REQUIRE_THROWS_AS`, `REQUIRE_THROWS_WITH`, `REQUIRE_NOTHROW`, `SUCCEED`

**Run Commands:**
```bash
# Build
pixi run build                            # cmake --preset release && cmake --build build/release

# Run all tests
pixi run test                             # cd build/release && ctest --output-on-failure

# Run specific tests by tag (after cd to build dir)
./build/release/test/cucascade_tests "[data_batch]"
./build/release/test/cucascade_tests "[gpu]"
./build/release/test/cucascade_tests "[.multi-device]"   # hidden by default, opt-in

# Exclude a tag
./build/release/test/cucascade_tests "~[.multi-device]"

# Debug build
pixi run build-debug                      # cmake --preset debug && cmake --build build/debug
```

**GPU Pool:**
- `test/unittest.cpp` sets up a global `rmm::mr::cuda_async_memory_resource` pool at startup
- Pool initial size: 2 GB; max: 10 GB (or 90% of device memory)
- Override via env var: `CUCASCADE_TEST_GPU_POOL_BYTES=<bytes>`
- Per-test CUDA device sync: `device_sync_listener` (registered via `CATCH_REGISTER_LISTENER`) calls `cudaDeviceSynchronize()` after every test case

## Test File Organization

**Location:** Separate `test/` directory (not co-located with source)

**Directory structure:**
```
test/
├── unittest.cpp                     # Main runner: GPU pool init, device sync listener, Catch2 Session
├── utils/
│   ├── mock_test_utils.hpp          # Mock objects: mock_data_representation, make_mock_memory_space
│   ├── cudf_test_utils.hpp          # cuDF helpers: create_simple_cudf_table, expect_cudf_tables_equal_on_stream
│   ├── cudf_test_utils.cpp          # cuDF helper implementations
│   └── test_memory_resources.hpp    # shared_device_resource, make_shared_current_device_resource
├── data/
│   ├── test_data_batch.cpp
│   ├── test_data_repository.cpp
│   ├── test_data_repository_manager.cpp
│   ├── test_data_representation.cpp
│   └── test_representation_converter.cpp
└── memory/
    ├── test_memory_reservation_manager.cpp
    ├── test_small_pinned_host_memory_resource.cpp
    ├── test_topology_discovery.cpp
    ├── test_gpu_kernels.cu
    └── test_gpu_kernels.cuh
```

**Naming:**
- Test files: `test_<module_name>.cpp` (mirrors `src/<domain>/<module_name>.cpp`)
- CUDA kernel test files: `.cu` extension

**All sources compiled into one binary:** `cucascade_tests` — no per-file test executables.

## Test Structure

**Suite Organization:**
```cpp
// Groups of related tests separated by comment banners
// =============================================================================
// Construction / move tests
// =============================================================================

TEST_CASE("synchronized_data_batch Construction", "[data_batch]")
{
  // Arrange
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  synchronized_data_batch batch(1, std::move(data));

  // Assert
  REQUIRE(batch.get_batch_id() == 1);
  REQUIRE(batch.get_subscriber_count() == 0);
}
```

**Tags:**
- `[data_batch]`, `[data_repository]`, `[memory_space]`, `[cpu_data_representation]`, `[gpu_data_representation]`
- `[gpu]` — tests requiring an actual GPU and CUDA allocations
- `[threading]` — multi-threaded concurrency tests
- `[.multi-device]` — hidden by default; require 2+ GPU devices (opt-in with `"[.multi-device]"`)
- `[.disabled]` — Catch2 hidden tag used to mark skipped/unsupported tests (e.g., tests requiring now-private internal APIs)

**SECTION usage:**
```cpp
TEST_CASE("synchronized_data_batch clone preserves tier information", "[data_batch]")
{
  SECTION("GPU tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch  = std::make_shared<synchronized_data_batch>(1, std::move(data));
    auto cloned = batch->clone(2, rmm::cuda_stream_view{});
    auto ro     = cloned->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::GPU);
  }
  SECTION("HOST tier") { /* ... */ }
  SECTION("DISK tier") { /* ... */ }
}
```

**Disabled tests:**
```cpp
// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_data_packed_representation Construction", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}
```

**Patterns:**
- Arrange-Assert (no explicit Act step) — construct object, then REQUIRE its observable state
- No shared test fixtures via `struct`/`class` Catch2 fixtures — prefer local variables in each `TEST_CASE`
- Helper factory functions at file scope for repeated setup (e.g., `create_test_batches(std::vector<uint64_t>)`, `createSingleDeviceMemoryManager()`)

## Mocking

**Framework:** Manual mock classes — no gmock or other mocking framework.

**Location:** `test/utils/mock_test_utils.hpp`

**Mock objects provided:**

`mock_data_representation` — lightweight `idata_representation` subclass:
```cpp
class mock_data_representation : private mock_memory_space_holder, public idata_representation {
 public:
  explicit mock_data_representation(memory::Tier tier, size_t size = 1024, size_t device_id = 0);

  std::size_t get_size_in_bytes() const override { return _size; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return _size; }
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

 private:
  size_t _size;
};
```

`make_mock_memory_space(Tier, device_id)` — creates a lightweight `memory_space` for any tier (GPU, HOST, DISK) using minimal real allocators.

`shared_device_resource` (in `test_memory_resources.hpp`) — wraps the current device resource so multiple `memory_space` configs can share one pool without fighting over ownership.

**What to mock:**
- `idata_representation` — use `mock_data_representation` when testing batch/repository logic that doesn't care about actual data content
- `memory_space` — use `make_mock_memory_space()` for unit tests; use `memory_reservation_manager` for integration tests

**What NOT to mock:**
- `synchronized_data_batch` — use real instances even in repository tests
- `memory_reservation_manager` — use real instances in converter and data representation tests
- CUDA streams — use real `rmm::cuda_stream` instances in GPU tests

## Fixtures and Factories

**Test data factories in `test/utils/mock_test_utils.hpp`:**
```cpp
// Create a memory space for a given tier
inline std::shared_ptr<memory::memory_space> make_mock_memory_space(memory::Tier tier,
                                                                    size_t device_id = 0);

// Create memory manager configs for GPU+HOST conversion tests
inline std::vector<memory::memory_space_config> create_conversion_test_configs();

// Create a simple cuDF table with 1 or 2 columns filled with known byte patterns
inline cudf::table create_simple_cudf_table(int num_rows, int num_columns, mr*, stream);
inline cudf::table create_simple_cudf_table(int num_rows, mr*, stream);   // 2 columns
inline cudf::table create_simple_cudf_table(mr*, stream);                 // 100 rows, 2 columns
```

**File-local helper in test files:**
```cpp
// test_data_repository.cpp
std::vector<std::shared_ptr<synchronized_data_batch>> create_test_batches(
  std::vector<uint64_t> batch_ids)
{
  std::vector<std::shared_ptr<synchronized_data_batch>> batches;
  for (auto batch_id : batch_ids) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    batches.emplace_back(std::make_shared<synchronized_data_batch>(batch_id, std::move(data)));
  }
  return batches;
}
```

**Memory resource fixtures:**
- `make_shared_current_device_resource` (in `test_memory_resources.hpp`) — factory fn used in `reservation_manager_configurator` to share the global test pool across memory spaces

## Coverage

**Requirements:** None enforced (no coverage tooling configured).

**Coverage approach:** Tests are written to cover:
1. Happy path construction and basic operations
2. Edge cases (empty table, 0 rows, `UINT64_MAX` IDs)
3. Concurrency/threading: multiple reader threads, reader-writer exclusion, atomic subscriber counts
4. Error paths: `REQUIRE_THROWS_AS(..., std::runtime_error)`, `REQUIRE_THROWS_WITH`
5. GPU data integrity: byte-level comparison via `expect_cudf_tables_equal_on_stream`

## Test Types

**Unit Tests:**
- Scope: single class or function in isolation using mock objects
- Examples: `test_data_batch.cpp` (all lock/accessor semantics), `test_representation_converter.cpp` (converter registry)

**Integration Tests:**
- Scope: multiple real subsystems interacting end-to-end
- Examples: `test_data_representation.cpp` (GPU-to-HOST conversion with real `memory_reservation_manager` and real cuDF data), `test_memory_reservation_manager.cpp` (allocator lifecycle with real GPU pools)

**GPU Kernel Tests:**
- File: `test/memory/test_gpu_kernels.cu`
- CUDA `.cu` file compiled with NVCC; tests GPU-side code paths

**E2E Tests:** Not present as a separate category; integration tests serve that role.

## Common Patterns

**Thread-safety testing pattern:**
```cpp
TEST_CASE("synchronized_data_batch try_get_read_only fails when mutable lock held", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto rw = batch.get_mutable();  // hold exclusive lock

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto ro = batch.try_get_read_only();
    got_lock.store(ro.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}
```

**Blocking-until-released pattern (uses `sleep_for` to ensure waiter is blocked):**
```cpp
auto ro = std::make_unique<read_only_data_batch>(batch.get_read_only());
std::atomic<bool> got_mutable{false};

std::thread writer([&batch, &got_mutable]() {
  auto rw = batch.get_mutable();
  got_mutable.store(true);
});

std::this_thread::sleep_for(std::chrono::milliseconds(20));
REQUIRE(got_mutable.load() == false);

ro.reset();     // release reader
writer.join();
REQUIRE(got_mutable.load() == true);
```

**GPU data integrity testing:**
```cpp
// After GPU operations, synchronize stream before comparing
stream.synchronize();
expect_cudf_tables_equal_on_stream(
  original_repr->get_table(), cloned_repr->get_table(), stream.view());
```

`expect_cudf_tables_equal_on_stream` is declared in `test/utils/cudf_test_utils.hpp` and implemented in `test/utils/cudf_test_utils.cpp`. It handles stream ordering with async allocations before comparing column contents.

**Error testing:**
```cpp
REQUIRE_THROWS_AS(batch.unsubscribe(), std::runtime_error);

REQUIRE_THROWS_WITH(repository.get_data_batch_by_id(0, 0),
                    "get_data_batch_by_id is not supported for unique_ptr repositories. "
                    "Use pop_data_batch to move ownership instead.");
```

**Large-scale concurrency stress tests:**
```cpp
constexpr int num_threads        = 10;
constexpr int batches_per_thread = 50;
std::vector<std::thread> threads;
for (int i = 0; i < num_threads; ++i) {
  threads.emplace_back([&, i]() { /* work */ });
}
for (auto& thread : threads) { thread.join(); }
REQUIRE(total_count == num_threads * batches_per_thread);
```

---

*Testing analysis: 2026-04-13*
