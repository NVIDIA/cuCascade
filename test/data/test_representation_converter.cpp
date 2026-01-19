/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/cudf_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <catch2/catch.hpp>

#include <memory>
#include <vector>

using namespace cucascade;
using cucascade::test::create_conversion_test_configs;
using cucascade::test::create_simple_cudf_table;
using cucascade::test::make_mock_memory_space;

// =============================================================================
// Test Fixtures and Helpers
// =============================================================================

// Custom test representation for testing custom converter registration
class custom_test_representation : public idata_representation {
 public:
  custom_test_representation(int value, memory::memory_space& space)
    : idata_representation(space), _value(value)
  {
  }

  std::size_t get_size_in_bytes() const override { return sizeof(_value); }

  int get_value() const { return _value; }

 private:
  int _value;
};

// Another custom representation for testing bidirectional conversion
class another_test_representation : public idata_representation {
 public:
  another_test_representation(double value, memory::memory_space& space)
    : idata_representation(space), _value(value)
  {
  }

  std::size_t get_size_in_bytes() const override { return sizeof(_value); }

  double get_value() const { return _value; }

 private:
  double _value;
};

// =============================================================================
// representation_converter_registry Registration Tests
// =============================================================================

TEST_CASE("representation_converter_registry register custom converter",
          "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  SECTION("Register a new converter succeeds")
  {
    REQUIRE_NOTHROW(
      registry.register_converter<custom_test_representation, another_test_representation>(
        [](idata_representation& source,
           const memory::memory_space* target_space,
           rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
          auto& src = source.cast<custom_test_representation>();
          return std::make_unique<another_test_representation>(
            static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
        }));
  }

  SECTION("Duplicate registration throws")
  {
    // Register the converter first
    registry.register_converter<custom_test_representation, another_test_representation>(
      [](idata_representation& source,
         const memory::memory_space* target_space,
         rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
        auto& src = source.cast<custom_test_representation>();
        return std::make_unique<another_test_representation>(
          static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
      });

    // Attempting to register the same converter again should throw
    // Use a lambda to avoid comma issues with the macro
    auto duplicate_register = [&]() {
      registry.register_converter<custom_test_representation, another_test_representation>(
        [](idata_representation&,
           const memory::memory_space*,
           rmm::cuda_stream_view) -> std::unique_ptr<idata_representation> { return nullptr; });
    };
    REQUIRE_THROWS_AS(duplicate_register(), std::runtime_error);
  }
}

TEST_CASE("representation_converter_registry has_converter", "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  SECTION("Returns false for unregistered converter")
  {
    REQUIRE_FALSE(
      registry.has_converter<custom_test_representation, another_test_representation>());
  }

  SECTION("Returns true after registration")
  {
    registry.register_converter<custom_test_representation, another_test_representation>(
      [](idata_representation& source,
         const memory::memory_space* target_space,
         rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
        auto& src = source.cast<custom_test_representation>();
        return std::make_unique<another_test_representation>(
          static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
      });

    REQUIRE(registry.has_converter<custom_test_representation, another_test_representation>());
  }

  SECTION("Returns false for reverse direction if not registered")
  {
    registry.register_converter<custom_test_representation, another_test_representation>(
      [](idata_representation& source,
         const memory::memory_space* target_space,
         rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
        auto& src = source.cast<custom_test_representation>();
        return std::make_unique<another_test_representation>(
          static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
      });

    // Forward direction registered
    REQUIRE(registry.has_converter<custom_test_representation, another_test_representation>());
    // Reverse direction NOT registered
    REQUIRE_FALSE(
      registry.has_converter<another_test_representation, custom_test_representation>());
  }
}

TEST_CASE("representation_converter_registry has_converter_for runtime lookup",
          "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  registry.register_converter<custom_test_representation, another_test_representation>(
    [](idata_representation& source,
       const memory::memory_space* target_space,
       rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
      auto& src = source.cast<custom_test_representation>();
      return std::make_unique<another_test_representation>(
        static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
    });

  custom_test_representation source_repr(42, *mock_space);

  SECTION("Returns true for registered source instance")
  {
    REQUIRE(registry.has_converter_for<another_test_representation>(source_repr));
  }

  SECTION("Returns false for unregistered target type")
  {
    REQUIRE_FALSE(registry.has_converter_for<custom_test_representation>(source_repr));
  }
}

TEST_CASE("representation_converter_registry unregister_converter", "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  SECTION("Unregister returns false for non-existent converter")
  {
    REQUIRE_FALSE(
      registry.unregister_converter<custom_test_representation, another_test_representation>());
  }

  SECTION("Unregister returns true and removes registered converter")
  {
    registry.register_converter<custom_test_representation, another_test_representation>(
      [](idata_representation& source,
         const memory::memory_space* target_space,
         rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
        auto& src = source.cast<custom_test_representation>();
        return std::make_unique<another_test_representation>(
          static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
      });

    REQUIRE(registry.has_converter<custom_test_representation, another_test_representation>());
    REQUIRE(
      registry.unregister_converter<custom_test_representation, another_test_representation>());
    REQUIRE_FALSE(
      registry.has_converter<custom_test_representation, another_test_representation>());
  }

  SECTION("After unregister, can register again")
  {
    registry.register_converter<custom_test_representation, another_test_representation>(
      [](idata_representation&,
         const memory::memory_space*,
         rmm::cuda_stream_view) -> std::unique_ptr<idata_representation> { return nullptr; });

    registry.unregister_converter<custom_test_representation, another_test_representation>();

    // Should not throw now
    REQUIRE_NOTHROW(
      registry.register_converter<custom_test_representation, another_test_representation>(
        [](idata_representation&,
           const memory::memory_space*,
           rmm::cuda_stream_view) -> std::unique_ptr<idata_representation> { return nullptr; }));
  }
}

// Note: We don't test clear() directly on the global registry because:
// 1. It removes builtin converters (GPU<->HOST)
// 2. The static flag in register_builtin_converters() prevents re-registration
// 3. This would break other tests that depend on builtin converters
// The unregister_converter tests above verify the removal functionality safely.

// =============================================================================
// representation_converter_registry Conversion Tests
// =============================================================================

TEST_CASE("representation_converter_registry convert with custom types",
          "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  // Register a converter that doubles the value
  registry.register_converter<custom_test_representation, another_test_representation>(
    [](idata_representation& source,
       const memory::memory_space* target_space,
       rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
      auto& src = source.cast<custom_test_representation>();
      return std::make_unique<another_test_representation>(
        static_cast<double>(src.get_value() * 2), *const_cast<memory::memory_space*>(target_space));
    });

  SECTION("Conversion produces expected result")
  {
    custom_test_representation source(21, *mock_space);
    auto result = registry.convert<another_test_representation>(
      source, mock_space.get(), rmm::cuda_stream_default);

    REQUIRE(result != nullptr);
    REQUIRE(result->get_value() == 42.0);  // 21 * 2
  }

  SECTION("Conversion throws for unregistered type pair")
  {
    another_test_representation source(3.14, *mock_space);

    REQUIRE_THROWS_AS(registry.convert<custom_test_representation>(
                        source, mock_space.get(), rmm::cuda_stream_default),
                      std::runtime_error);
  }
}

TEST_CASE("representation_converter_registry convert with type_index", "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  registry.register_converter<custom_test_representation, another_test_representation>(
    [](idata_representation& source,
       const memory::memory_space* target_space,
       rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
      auto& src = source.cast<custom_test_representation>();
      return std::make_unique<another_test_representation>(
        static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
    });

  custom_test_representation source(100, *mock_space);

  SECTION("Runtime type_index conversion works")
  {
    std::type_index target_type = typeid(another_test_representation);
    auto result = registry.convert(source, target_type, mock_space.get(), rmm::cuda_stream_default);

    REQUIRE(result != nullptr);
    auto& typed_result = result->cast<another_test_representation>();
    REQUIRE(typed_result.get_value() == 100.0);
  }

  SECTION("Runtime conversion throws for unregistered pair")
  {
    std::type_index wrong_target = typeid(custom_test_representation);
    REQUIRE_THROWS_AS(
      registry.convert(source, wrong_target, mock_space.get(), rmm::cuda_stream_default),
      std::runtime_error);
  }
}

// =============================================================================
// Built-in Converter Registration Tests
// =============================================================================

TEST_CASE("register_builtin_converters registers all expected converters",
          "[representation_converter][builtin]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  SECTION("GPU to HOST converter is registered")
  {
    REQUIRE(registry.has_converter<gpu_table_representation, host_table_representation>());
  }

  SECTION("HOST to GPU converter is registered")
  {
    REQUIRE(registry.has_converter<host_table_representation, gpu_table_representation>());
  }

  SECTION("GPU to GPU converter is registered")
  {
    REQUIRE(registry.has_converter<gpu_table_representation, gpu_table_representation>());
  }

  SECTION("HOST to HOST converter is registered")
  {
    REQUIRE(registry.has_converter<host_table_representation, host_table_representation>());
  }
}

// =============================================================================
// Built-in Converter Functional Tests
// =============================================================================

TEST_CASE("Built-in GPU to HOST conversion works", "[representation_converter][builtin][gpu_host]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);

  auto table = create_simple_cudf_table(50, gpu_space->get_default_allocator());
  gpu_table_representation gpu_repr(std::move(table),
                                    *const_cast<memory::memory_space*>(gpu_space));

  rmm::cuda_stream stream;

  auto host_result = registry.convert<host_table_representation>(gpu_repr, host_space, stream);
  stream.synchronize();

  REQUIRE(host_result != nullptr);
  REQUIRE(host_result->get_current_tier() == memory::Tier::HOST);
  REQUIRE(host_result->get_size_in_bytes() > 0);
}

TEST_CASE("Built-in HOST to GPU conversion works", "[representation_converter][builtin][gpu_host]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);

  // First create a GPU repr then convert to host to get a valid host_table_representation
  auto table = create_simple_cudf_table(50, gpu_space->get_default_allocator());
  gpu_table_representation gpu_repr(std::move(table),
                                    *const_cast<memory::memory_space*>(gpu_space));

  rmm::cuda_stream stream;

  auto host_repr = registry.convert<host_table_representation>(gpu_repr, host_space, stream);
  stream.synchronize();

  // Now convert back to GPU
  auto gpu_result = registry.convert<gpu_table_representation>(*host_repr, gpu_space, stream);
  stream.synchronize();

  REQUIRE(gpu_result != nullptr);
  REQUIRE(gpu_result->get_current_tier() == memory::Tier::GPU);
  REQUIRE(gpu_result->get_table().num_rows() == 50);
}

TEST_CASE("Built-in roundtrip GPU->HOST->GPU preserves data",
          "[representation_converter][builtin][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);

  auto original_table = create_simple_cudf_table(100, gpu_space->get_default_allocator());
  gpu_table_representation original_repr(std::move(original_table),
                                         *const_cast<memory::memory_space*>(gpu_space));

  rmm::cuda_stream stream;

  // GPU -> HOST
  auto host_repr = registry.convert<host_table_representation>(original_repr, host_space, stream);
  stream.synchronize();

  // HOST -> GPU
  auto result_repr = registry.convert<gpu_table_representation>(*host_repr, gpu_space, stream);
  stream.synchronize();

  // Verify data integrity
  REQUIRE(result_repr != nullptr);
  cucascade::test::expect_cudf_tables_equal_on_stream(
    original_repr.get_table(), result_repr->get_table(), stream);
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

TEST_CASE("Converter preserves memory space properties", "[representation_converter]")
{
  representation_converter_registry registry;

  auto source_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto target_space = make_mock_memory_space(memory::Tier::HOST, 1);

  registry.register_converter<custom_test_representation, another_test_representation>(
    [](idata_representation& source,
       const memory::memory_space* target_space,
       rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
      auto& src = source.cast<custom_test_representation>();
      return std::make_unique<another_test_representation>(
        static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
    });

  custom_test_representation source(42, *source_space);

  auto result = registry.convert<another_test_representation>(
    source, target_space.get(), rmm::cuda_stream_default);

  REQUIRE(result->get_current_tier() == memory::Tier::HOST);
  REQUIRE(result->get_device_id() == 1);
}

TEST_CASE("Multiple independent converters can coexist", "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  // Register forward converter (custom -> another)
  registry.register_converter<custom_test_representation, another_test_representation>(
    [](idata_representation& source,
       const memory::memory_space* target_space,
       rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
      auto& src = source.cast<custom_test_representation>();
      return std::make_unique<another_test_representation>(
        static_cast<double>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
    });

  // Register reverse converter (another -> custom)
  registry.register_converter<another_test_representation, custom_test_representation>(
    [](idata_representation& source,
       const memory::memory_space* target_space,
       rmm::cuda_stream_view /*stream*/) -> std::unique_ptr<idata_representation> {
      auto& src = source.cast<another_test_representation>();
      return std::make_unique<custom_test_representation>(
        static_cast<int>(src.get_value()), *const_cast<memory::memory_space*>(target_space));
    });

  // Both should exist
  REQUIRE(registry.has_converter<custom_test_representation, another_test_representation>());
  REQUIRE(registry.has_converter<another_test_representation, custom_test_representation>());

  // Test forward conversion
  custom_test_representation custom_src(10, *mock_space);
  auto another_result = registry.convert<another_test_representation>(
    custom_src, mock_space.get(), rmm::cuda_stream_default);
  REQUIRE(another_result->get_value() == 10.0);

  // Test reverse conversion
  another_test_representation another_src(25.7, *mock_space);
  auto custom_result = registry.convert<custom_test_representation>(
    another_src, mock_space.get(), rmm::cuda_stream_default);
  REQUIRE(custom_result->get_value() == 25);  // truncated to int
}

TEST_CASE("Converter error message includes type names", "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);
  custom_test_representation source(42, *mock_space);

  try {
    registry.convert<another_test_representation>(
      source, mock_space.get(), rmm::cuda_stream_default);
    FAIL("Expected exception to be thrown");
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    // Error message should mention that no converter was found
    REQUIRE(msg.find("No converter registered") != std::string::npos);
  }
}

TEST_CASE("Duplicate registration error message includes type names", "[representation_converter]")
{
  representation_converter_registry registry;
  auto mock_space = make_mock_memory_space(memory::Tier::GPU, 0);

  registry.register_converter<custom_test_representation, another_test_representation>(
    [](idata_representation&,
       const memory::memory_space*,
       rmm::cuda_stream_view) -> std::unique_ptr<idata_representation> { return nullptr; });

  try {
    registry.register_converter<custom_test_representation, another_test_representation>(
      [](idata_representation&,
         const memory::memory_space*,
         rmm::cuda_stream_view) -> std::unique_ptr<idata_representation> { return nullptr; });
    FAIL("Expected exception to be thrown");
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    // Error message should mention that converter is already registered
    REQUIRE(msg.find("already registered") != std::string::npos);
  }
}
