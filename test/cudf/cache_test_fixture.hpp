#pragma once

#include "io/cache/controlled_reactor.hpp"
#include "utils/test_memory_resources.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::test {

inline std::unique_ptr<memory::memory_reservation_manager> identity_memory(
  double host_reservation_fraction = 1.0, std::size_t host_capacity = 256ULL << 20)
{
  memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ULL << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_gpu_memory_resource_factory(make_shared_current_device_resource)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(host_reservation_fraction);
  return std::make_unique<memory::memory_reservation_manager>(builder.build());
}

inline io::cache::config identity_cache_config()
{
  io::cache::config cfg;
  cfg.mode                            = io::cache::cache_mode::cucs;
  cfg.eviction                        = io::cache::eviction_policy::lru;
  cfg.min_prefetching_budget_fraction = 0.5;
  cfg.eviction_threshold_fraction     = 1.0;
  cfg.apply_mode();
  return cfg;
}

/// Hint [0, size) of @p ds and take staging buffers for it, as a scan does.
inline void advise(io::datasource& ds, std::size_t size)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<std::int64_t>(size));
  ds.fadvise(ranges, 0);
  REQUIRE(ds.prepare_prefetch(false) == io::prepare_result::prepared);
}

struct cache_test_fixture {
  explicit cache_test_fixture(io::cache::config config                 = identity_cache_config(),
                              controlled_reactor_config backend_config = {},
                              double host_reservation_fraction         = 1.0,
                              std::size_t host_capacity                = 256ULL << 20)
    : memory(identity_memory(host_reservation_fraction, host_capacity)),
      context(std::make_shared<controlled_ioctx>(backend_config))
  {
    context->start();
    context->initialize_cache(*memory, config, nullptr);
  }

  ~cache_test_fixture()
  {
    context->shutdown();
    context->shutdown_cache();
  }

  cache_test_fixture(cache_test_fixture const&)            = delete;
  cache_test_fixture& operator=(cache_test_fixture const&) = delete;

  std::unique_ptr<io::datasource> open(std::string const& path,
                                       std::size_t bytes,
                                       std::string const& tag = "\"one\"")
  {
    return std::make_unique<io::datasource>(context, held_object(path, tag, bytes));
  }

  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::shared_ptr<controlled_ioctx> context;
};

}  // namespace cucascade::test
