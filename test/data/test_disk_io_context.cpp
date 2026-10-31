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

#include "utils/child_test_runner.hpp"
#include "utils/cudf_test_utils.hpp"
#include "utils/disk_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

using namespace cucascade;

namespace {

/// DISK spaces kept alive until static destruction by the exit test's child: they are
/// namespace-scope, so they are constructed before CUDA is first used and destroyed after the CUDA
/// runtime and the cuFile driver have been torn down -- as with a reservation manager a user keeps
/// in a global.
std::shared_ptr<memory::memory_space> g_exit_default_disk;
std::shared_ptr<memory::memory_space> g_exit_cufile_disk;

auto& shared_gpu_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::GPU, 0);
  return s;
}

auto& shared_host_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::HOST, 0);
  return s;
}

::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s;
}

representation_converter_registry& registry()
{
  static auto r = [] {
    auto result = std::make_unique<representation_converter_registry>();
    register_builtin_converters(*result);
    return result;
  }();
  return *r;
}

/// A fresh directory under the temp dir, removed (with its content) on destruction, so a test
/// can check exactly which batch files a conversion left behind.
class scratch_dir {
 public:
  scratch_dir()
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_disk_io_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1))))
  {
    std::filesystem::create_directories(_path);
  }
  ~scratch_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(_path, ec);
  }
  scratch_dir(scratch_dir const&)            = delete;
  scratch_dir& operator=(scratch_dir const&) = delete;

  [[nodiscard]] std::string string() const { return _path.string(); }

  [[nodiscard]] std::size_t file_count() const
  {
    std::size_t count = 0;
    for ([[maybe_unused]] auto const& entry : std::filesystem::directory_iterator(_path)) {
      ++count;
    }
    return count;
  }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
};

/// A table of @p n_columns INT64 sequence columns of @p rows rows; the second column, if any,
/// carries a null mask so masks travel too.
std::unique_ptr<cudf::table> make_sequence_table(cudf::size_type rows,
                                                 std::size_t n_columns,
                                                 ::cuda::stream_ref stream)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t i = 0; i < n_columns; ++i) {
    auto const start = cudf::numeric_scalar<std::int64_t>(static_cast<std::int64_t>(i) * 1000);
    auto column      = cudf::sequence(rows, start, stream);
    if (i == 1) {
      column->set_null_mask(cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream), 0);
    }
    columns.push_back(std::move(column));
  }
  stream.sync();
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Run the test case @p name in a child process unless this already is that child.
 *
 * The child is this test executable re-run on the single test case with @p marker set to "1"; it
 * must exit with status 0. Returns true in the parent (the caller returns), false in the child
 * (the caller runs the test body).
 */
bool run_in_child_process(char const* name, char const* marker)
{
  auto const* value = std::getenv(marker);
  if (value != nullptr && std::string_view{value} == "1") { return false; }
  test::child_test_runner child{name, {{marker, std::string{"1"}}}};
  bool const finished = child.wait_for(std::chrono::seconds{300});
  if (!finished) { child.stop(); }
  INFO(child.output());
  REQUIRE(finished);
  REQUIRE(WIFEXITED(child.status()));
  REQUIRE(WEXITSTATUS(child.status()) == 0);
  return true;
}

/// Make io_uring_setup fail with EPERM in this thread and every thread it creates from now on, as
/// container seccomp profiles (Docker 25+), gVisor and kernel.io_uring_disabled do.
void forbid_io_uring()
{
  sock_filter filter[] = {
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, static_cast<std::uint32_t>(offsetof(seccomp_data, nr))),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, static_cast<std::uint32_t>(__NR_io_uring_setup), 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0])), filter};
  REQUIRE(::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0);
  REQUIRE(::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0);
}

std::unique_ptr<gpu_table_representation> make_gpu_rep(std::unique_ptr<cudf::table> table)
{
  return std::make_unique<gpu_table_representation>(
    std::move(table), *shared_gpu_space(), ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
}

/**
 * @brief An io context that claims write support but fails to open anything for writing, to
 *        drive the converters' failure paths after the batch file was created.
 */
class failing_write_ioctx final : public io::ioctx {
 public:
  [[nodiscard]] io::io_context_type type() const noexcept override
  {
    return io::io_context_type::uring;
  }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view) const noexcept override { return true; }
  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_write() const noexcept override { return true; }
  [[nodiscard]] bool supports_device_write() const noexcept override { return true; }
  [[nodiscard]] std::vector<io::byte_range> align_and_coalesce(
    std::span<io::byte_range const>, std::optional<std::size_t>) const noexcept override
  {
    return {};
  }
  std::size_t host_read_io(io::io_object const&, std::size_t, std::size_t, std::uint8_t*) override
  {
    throw std::runtime_error("failing_write_ioctx: no reads");
  }
  exec::semi_future<std::size_t> mixed_readv_async_io(
    io::io_object const&, std::vector<io::prepared_io_slice>&&) noexcept override
  {
    return exec::make_semi_future<std::size_t>(
      std::make_exception_ptr(std::runtime_error("failing_write_ioctx: no reads")));
  }

 protected:
  std::shared_ptr<io::io_object> create_io_object(std::string) override
  {
    throw std::runtime_error("failing_write_ioctx: no reads");
  }
  // create_io_object_for_write keeps the base default: std::errc::not_supported.
};

}  // namespace

// =============================================================================
// Context construction and selection
// =============================================================================

TEST_CASE("make_disk_io_context builds a context of the requested backend", "[disk][io][context]")
{
  using io::io_context_type;
  auto const [backend, expected] =
    GENERATE(std::pair{disk_io_backend::uring, io_context_type::uring},
             std::pair{disk_io_backend::cufile, io_context_type::cufile});
  CAPTURE(test::to_string(backend));

  auto context = test::try_make_disk_io_context(backend);
  if (context == nullptr) {
    WARN("the " << test::to_string(backend) << " backend cannot run in this process");
    return;
  }
  REQUIRE(context->type() == expected);
}

TEST_CASE("make_disk_io_context resolves automatic to a concrete backend", "[disk][io][context]")
{
  auto context = make_disk_io_context();
  REQUIRE(context != nullptr);
  auto const type = context->type();
  REQUIRE((type == io::io_context_type::uring || type == io::io_context_type::cufile));
}

TEST_CASE("make_disk_io_context rejects a backend without reactors", "[disk][io][context]")
{
  disk_io_context_options options;
  options.backend                   = disk_io_backend::uring;
  options.io.uring_reactors_per_gpu = 0;
  REQUIRE_THROWS_AS(make_disk_io_context(options), std::invalid_argument);

  options.backend                    = disk_io_backend::cufile;
  options.io.cufile_reactors_per_gpu = 0;
  REQUIRE_THROWS_AS(make_disk_io_context(options), std::invalid_argument);
}

TEST_CASE("without io_uring the DISK tier fails fast or falls back", "[disk][io][no_uring]")
{
  if (run_in_child_process("without io_uring the DISK tier fails fast or falls back",
                           "CUCASCADE_DISK_NO_URING_CHILD")) {
    return;
  }
  // Child: io_uring is forbidden from here on, before anything set up a ring.
  forbid_io_uring();

  // A forced uring context fails at construction with the real cause ...
  disk_io_context_options forced;
  forced.backend = disk_io_backend::uring;
  try {
    [[maybe_unused]] auto context = make_disk_io_context(forced);
    FAIL("make_disk_io_context built a uring context although io_uring is forbidden");
  } catch (std::system_error const& error) {
    REQUIRE(error.code().value() == EPERM);
  }

  // ... while automatic, and so a DISK space without a context, picks a backend that works.
  REQUIRE(make_disk_io_context()->type() != io::io_context_type::uring);
  scratch_dir dir;
  auto space   = test::make_disk_space(disk_io_backend::automatic, 16, dir.string());
  auto gpu_rep = make_gpu_rep(make_sequence_table(300'000, 2, shared_stream()));
  auto from_gpu =
    registry().convert<disk_data_representation>(*gpu_rep, space.get(), shared_stream());
  REQUIRE(space->get_io_context()->type() != io::io_context_type::uring);
  auto back = registry().convert<gpu_table_representation>(
    *from_gpu, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table_view(), back->get_table_view(), shared_stream());
  auto host_rep = registry().convert<host_data_representation>(
    *gpu_rep, shared_host_space().get(), shared_stream());
  auto from_cpu =
    registry().convert<disk_data_representation>(*host_rep, space.get(), shared_stream());
  auto host_back = registry().convert<host_data_representation>(
    *from_cpu, shared_host_space().get(), shared_stream());
  auto back_from_host = registry().convert<gpu_table_representation>(
    *host_back, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table_view(), back_from_host->get_table_view(), shared_stream());
}

TEST_CASE("default_disk_io_context is shared by every caller", "[disk][io][context]")
{
  auto first  = default_disk_io_context();
  auto second = default_disk_io_context();
  REQUIRE(first != nullptr);
  REQUIRE(first == second);
}

TEST_CASE("disk spaces without a context adopt the shared default on first conversion",
          "[disk][io][context]")
{
  scratch_dir dir;
  auto space_a = test::make_disk_space(disk_io_backend::automatic, 10, dir.string());
  auto space_b = test::make_disk_space(disk_io_backend::automatic, 11, dir.string());
  REQUIRE(space_a->get_io_context() == nullptr);
  REQUIRE(space_b->get_io_context() == nullptr);
  if (!test::disk_space_can_write(*space_a, /*device=*/true)) {
    WARN("the default disk context cannot write device memory yet");
    return;
  }

  auto gpu_rep = make_gpu_rep(make_sequence_table(1000, 2, shared_stream()));
  auto disk_a =
    registry().convert<disk_data_representation>(*gpu_rep, space_a.get(), shared_stream());
  auto disk_b =
    registry().convert<disk_data_representation>(*gpu_rep, space_b.get(), shared_stream());

  auto const shared_default = default_disk_io_context();
  REQUIRE(space_a->get_io_context() == shared_default);
  REQUIRE(space_b->get_io_context() == shared_default);
}

TEST_CASE("set_io_context switches the context of later conversions", "[disk][io][context]")
{
  scratch_dir dir;
  auto first  = test::try_make_disk_io_context(disk_io_backend::uring);
  auto second = test::try_make_disk_io_context(disk_io_backend::uring);
  if (first == nullptr || second == nullptr) {
    WARN("the uring backend cannot run in this process");
    return;
  }
  if (!first->supports_device_write()) {
    WARN("the uring context cannot write device memory yet");
    return;
  }

  memory::disk_memory_space_config config;
  config.disk_id         = 12;
  config.memory_capacity = 1UL << 30;
  config.mount_paths     = dir.string();
  config.io_context      = first;
  memory::memory_space space(config);
  REQUIRE(space.get_io_context() == first);

  auto gpu_rep = make_gpu_rep(make_sequence_table(4096, 2, shared_stream()));
  auto written = registry().convert<disk_data_representation>(*gpu_rep, &space, shared_stream());
  space.set_io_context(second);
  REQUIRE(space.get_io_context() == second);

  // A file written through the first context reads back through the second one.
  auto read_back = registry().convert<gpu_table_representation>(
    *written, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table_view(), read_back->get_table_view(), shared_stream());
}

// =============================================================================
// Large tables, both paths, every backend
// =============================================================================

TEST_CASE("large multi-column tables round-trip through every disk backend", "[disk][io][large]")
{
  // 3 columns of 4.5M odd rows x 8 B = 34 MiB each: larger than the write split, many staging
  // slots per column, unaligned tails, and a null mask on the second column.
  auto gpu_rep = make_gpu_rep(make_sequence_table(4'500'007, 3, shared_stream()));

  for (auto const backend : test::disk_test_backends) {
    auto* disk_space = test::shared_disk_space_for(backend);
    CAPTURE(test::to_string(backend));
    if (disk_space == nullptr) {
      WARN("skipping the " << test::to_string(backend) << " disk backend: unavailable here");
      continue;
    }

    if (test::disk_space_can_write(*disk_space, /*device=*/true)) {
      auto disk_rep =
        registry().convert<disk_data_representation>(*gpu_rep, disk_space, shared_stream());
      auto back = registry().convert<gpu_table_representation>(
        *disk_rep, shared_gpu_space().get(), shared_stream());
      test::expect_cudf_tables_equal_on_stream(
        gpu_rep->get_table_view(), back->get_table_view(), shared_stream());
    } else {
      WARN("skipping GPU <-> DISK for " << test::to_string(backend) << ": no device writes");
    }

    if (test::disk_space_can_write(*disk_space, /*device=*/false)) {
      auto host_rep = registry().convert<host_data_representation>(
        *gpu_rep, shared_host_space().get(), shared_stream());
      auto disk_rep =
        registry().convert<disk_data_representation>(*host_rep, disk_space, shared_stream());
      auto host_back = registry().convert<host_data_representation>(
        *disk_rep, shared_host_space().get(), shared_stream());
      auto back = registry().convert<gpu_table_representation>(
        *host_back, shared_gpu_space().get(), shared_stream());
      test::expect_cudf_tables_equal_on_stream(
        gpu_rep->get_table_view(), back->get_table_view(), shared_stream());
    } else {
      WARN("skipping HOST <-> DISK for " << test::to_string(backend) << ": no host writes");
    }
  }
}

// =============================================================================
// Concurrency
// =============================================================================

TEST_CASE("concurrent conversions on one disk space", "[disk][io][concurrency]")
{
  auto& disk_space = *test::shared_disk_space_for(disk_io_backend::automatic);
  if (!test::disk_space_can_write(disk_space, /*device=*/true)) {
    WARN("the default disk context cannot write device memory yet");
    return;
  }

  constexpr int n_threads = 8;
  std::vector<std::unique_ptr<gpu_table_representation>> sources;
  for (int i = 0; i < n_threads; ++i) {
    auto const rows = static_cast<cudf::size_type>(500'000 + i * 4097);
    sources.push_back(make_gpu_rep(make_sequence_table(rows, 2, shared_stream())));
  }

  // Catch2 assertions are not thread-safe: threads only convert, the main thread checks.
  std::vector<std::unique_ptr<gpu_table_representation>> results(n_threads);
  std::vector<std::exception_ptr> errors(n_threads);
  std::vector<rmm::cuda_stream> streams(n_threads);
  {
    std::vector<std::jthread> threads;
    for (int i = 0; i < n_threads; ++i) {
      threads.emplace_back([&, i] {
        try {
          auto const index = static_cast<std::size_t>(i);
          ::cuda::stream_ref stream{streams[index]};
          auto disk_rep =
            registry().convert<disk_data_representation>(*sources[index], &disk_space, stream);
          results[index] = registry().convert<gpu_table_representation>(
            *disk_rep, shared_gpu_space().get(), stream);
          stream.sync();
        } catch (...) {
          errors[static_cast<std::size_t>(i)] = std::current_exception();
        }
      });
    }
  }

  for (std::size_t i = 0; i < static_cast<std::size_t>(n_threads); ++i) {
    if (errors[i]) { std::rethrow_exception(errors[i]); }
    REQUIRE(results[i] != nullptr);
    test::expect_cudf_tables_equal_on_stream(
      sources[i]->get_table_view(), results[i]->get_table_view(), shared_stream());
  }
}

// =============================================================================
// Failure paths and file lifetime
// =============================================================================

TEST_CASE("a failed disk write leaves no batch file behind", "[disk][io][error]")
{
  scratch_dir dir;
  memory::disk_memory_space_config config;
  config.disk_id         = 13;
  config.memory_capacity = 1UL << 30;
  config.mount_paths     = dir.string();
  config.io_context      = std::make_shared<failing_write_ioctx>();
  memory::memory_space space(config);

  auto gpu_rep = make_gpu_rep(make_sequence_table(1000, 2, shared_stream()));
  REQUIRE_THROWS(registry().convert<disk_data_representation>(*gpu_rep, &space, shared_stream()));
  REQUIRE(dir.file_count() == 0);

  auto host_rep = registry().convert<host_data_representation>(
    *gpu_rep, shared_host_space().get(), shared_stream());
  REQUIRE_THROWS(registry().convert<disk_data_representation>(*host_rep, &space, shared_stream()));
  REQUIRE(dir.file_count() == 0);
}

TEST_CASE("a truncated or missing batch file fails the read", "[disk][io][error]")
{
  scratch_dir dir;
  auto space = test::make_disk_space(disk_io_backend::automatic, 14, dir.string());
  if (!test::disk_space_can_write(*space, /*device=*/true)) {
    WARN("the default disk context cannot write device memory yet");
    return;
  }

  auto gpu_rep = make_gpu_rep(make_sequence_table(100'000, 2, shared_stream()));
  auto disk_rep =
    registry().convert<disk_data_representation>(*gpu_rep, space.get(), shared_stream());
  auto const& path = disk_rep->get_disk_table().file_path;

  SECTION("truncated")
  {
    std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
    REQUIRE_THROWS(registry().convert<gpu_table_representation>(
      *disk_rep, shared_gpu_space().get(), shared_stream()));
    REQUIRE_THROWS(registry().convert<host_data_representation>(
      *disk_rep, shared_host_space().get(), shared_stream()));
  }
  SECTION("missing")
  {
    std::filesystem::remove(path);
    REQUIRE_THROWS(registry().convert<gpu_table_representation>(
      *disk_rep, shared_gpu_space().get(), shared_stream()));
    REQUIRE_THROWS(registry().convert<host_data_representation>(
      *disk_rep, shared_host_space().get(), shared_stream()));
  }
}

TEST_CASE("a disk write failing after submit throws and leaves no batch file",
          "[disk][io][error][efbig]")
{
  if (run_in_child_process("a disk write failing after submit throws and leaves no batch file",
                           "CUCASCADE_DISK_EFBIG_CHILD")) {
    return;
  }
  // Child: 3 x 4M INT64 columns (~96 MiB per file) against an 8 MiB file-size limit, so the
  // request's first ops land and later ones fail with EFBIG while siblings are in flight.
  auto gpu_rep  = make_gpu_rep(make_sequence_table(4'000'003, 3, shared_stream()));
  auto host_rep = registry().convert<host_data_representation>(
    *gpu_rep, shared_host_space().get(), shared_stream());
  shared_stream().sync();
  std::signal(SIGXFSZ, SIG_IGN);  // EFBIG instead of the default SIGXFSZ termination

  for (std::size_t i = 0; i < test::disk_test_backends.size(); ++i) {
    auto const backend = test::disk_test_backends[i];
    CAPTURE(test::to_string(backend));
    scratch_dir dir;
    auto space = test::make_disk_space(backend, 30 + static_cast<int>(i), dir.string());
    if (space == nullptr) {
      WARN("skipping the " << test::to_string(backend) << " disk backend: unavailable here");
      continue;
    }

    rlimit previous{};
    REQUIRE(::getrlimit(RLIMIT_FSIZE, &previous) == 0);
    rlimit limited   = previous;
    limited.rlim_cur = 8UL << 20;
    REQUIRE(::setrlimit(RLIMIT_FSIZE, &limited) == 0);
    REQUIRE_THROWS(
      registry().convert<disk_data_representation>(*gpu_rep, space.get(), shared_stream()));
    REQUIRE(dir.file_count() == 0);
    REQUIRE_THROWS(
      registry().convert<disk_data_representation>(*host_rep, space.get(), shared_stream()));
    REQUIRE(dir.file_count() == 0);
    REQUIRE(::setrlimit(RLIMIT_FSIZE, &previous) == 0);

    // The space keeps working once the limit is gone.
    auto disk_rep =
      registry().convert<disk_data_representation>(*gpu_rep, space.get(), shared_stream());
    auto back = registry().convert<gpu_table_representation>(
      *disk_rep, shared_gpu_space().get(), shared_stream());
    test::expect_cudf_tables_equal_on_stream(
      gpu_rep->get_table_view(), back->get_table_view(), shared_stream());
  }
}

TEST_CASE("disk_data_representation deletes its batch file", "[disk][io][lifetime]")
{
  scratch_dir dir;
  auto space = test::make_disk_space(disk_io_backend::automatic, 15, dir.string());
  if (!test::disk_space_can_write(*space, /*device=*/true)) {
    WARN("the default disk context cannot write device memory yet");
    return;
  }

  auto gpu_rep = make_gpu_rep(make_sequence_table(1000, 1, shared_stream()));
  auto disk_rep =
    registry().convert<disk_data_representation>(*gpu_rep, space.get(), shared_stream());
  auto const path = disk_rep->get_disk_table().file_path;
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(std::filesystem::file_size(path) == disk_rep->get_size_in_bytes());
  disk_rep.reset();
  REQUIRE_FALSE(std::filesystem::exists(path));
  REQUIRE(dir.file_count() == 0);
}

// =============================================================================
// Process exit
// =============================================================================

TEST_CASE("disk spaces kept in globals survive process exit", "[disk][io][lifetime][exit]")
{
#ifndef NDEBUG
  // Debug builds assert on every CUDA call that fails after the runtime was torn down
  // (CUCASCADE_ASSERT_CUDA_SUCCESS), which pinned memory freed from a global always does -- the
  // same holds for HOST memory spaces kept in a global.
  SKIP("debug builds assert on CUDA calls made after the CUDA runtime was torn down");
#endif
  if (run_in_child_process("disk spaces kept in globals survive process exit",
                           "CUCASCADE_DISK_EXIT_CHILD")) {
    return;
  }

  // Child: bring up both kinds of context (the shared default and a forced cuFile one), run every
  // conversion through them, and leave the spaces -- and so the contexts, their reactor threads and
  // pinned staging -- to static destruction.
  g_exit_default_disk = test::make_disk_space(disk_io_backend::automatic, 20);
  g_exit_cufile_disk  = test::make_disk_space(disk_io_backend::cufile, 21);
  auto gpu_rep        = make_gpu_rep(make_sequence_table(300'000, 2, shared_stream()));
  for (auto* space : {g_exit_default_disk.get(), g_exit_cufile_disk.get()}) {
    if (test::disk_space_can_write(*space, /*device=*/true)) {
      auto disk_rep =
        registry().convert<disk_data_representation>(*gpu_rep, space, shared_stream());
      auto back = registry().convert<gpu_table_representation>(
        *disk_rep, shared_gpu_space().get(), shared_stream());
      shared_stream().sync();
    }
    if (test::disk_space_can_write(*space, /*device=*/false)) {
      auto host_rep = registry().convert<host_data_representation>(
        *gpu_rep, shared_host_space().get(), shared_stream());
      auto disk_rep =
        registry().convert<disk_data_representation>(*host_rep, space, shared_stream());
      auto back = registry().convert<host_data_representation>(
        *disk_rep, shared_host_space().get(), shared_stream());
    }
  }
  REQUIRE(g_exit_default_disk->get_io_context() != nullptr);
  REQUIRE(g_exit_cufile_disk->get_io_context() != nullptr);
}
