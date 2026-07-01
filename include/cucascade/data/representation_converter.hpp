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

#pragma once

#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>

namespace cucascade {

/**
 * @brief Type alias for a converter function.
 *
 * A converter function transforms a source idata_representation into a target
 * idata_representation of a specific type. The converter is registered for a
 * specific (source_type → target_type) pair.
 *
 * @param source The source data representation to convert from
 * @param target_memory_space The memory space where the target representation will be allocated
 * @param stream CUDA stream for asynchronous memory operations
 * @param reservation Optional caller-owned reservation on the target memory space. When non-null,
 *        the converter draws its target allocation down from this reservation instead of committing
 *        fresh capacity, preventing double-counting against the target pool. May be nullptr.
 * @return A new idata_representation of the registered target type (always unique_ptr)
 *
 * @note Because std::function cannot carry default arguments, every registered converter must
 *       accept this parameter. Converters whose target tier does not honor reservations (GPU/DISK)
 *       ignore it.
 */
using representation_converter_fn = std::function<std::unique_ptr<idata_representation>(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  memory::reservation* reservation)>;

/**
 * @brief Key for looking up converters in the registry.
 *
 * Identifies a conversion path from one idata_representation type to another.
 * The lookup is based on the (source_type, target_type) pair, NOT on memory tier.
 */
struct converter_key {
  std::type_index source_type;
  std::type_index target_type;

  bool operator==(const converter_key& other) const
  {
    return source_type == other.source_type && target_type == other.target_type;
  }
};

/**
 * @brief Hash function for converter_key.
 */
struct converter_key_hash {
  std::size_t operator()(const converter_key& key) const
  {
    auto h1 = std::hash<std::type_index>{}(key.source_type);
    auto h2 = std::hash<std::type_index>{}(key.target_type);
    return h1 ^ (h2 << 1);
  }
};

/**
 * @brief Registry for data representation converters.
 *
 * This class provides a central registry for converters between different
 * idata_representation types. Converters are registered and looked up by
 * their (source_type → target_type) pair, where both types are derived from
 * idata_representation.
 *
 * This design allows:
 * - Libraries to define their own idata_representation subclasses
 * - Libraries to register converters from/to their custom representations
 * - Runtime lookup of the appropriate converter based on concrete types
 *
 * The registry is thread-safe for concurrent registration and lookup.
 *
 * @note cuCascade core ships no built-in converters. Concrete converters (e.g. between GPU,
 * HOST, and DISK representations) are provided by the cucascade-cudf layer, which registers
 * them via register_builtin_converters() (declared in cucascade/cudf/builtin_converters.hpp).
 */
class representation_converter_registry {
 public:
  /**
   * @brief Construct an empty converter registry.
   */
  representation_converter_registry() = default;

  // Non-copyable, non-movable
  representation_converter_registry(const representation_converter_registry&)            = delete;
  representation_converter_registry& operator=(const representation_converter_registry&) = delete;
  representation_converter_registry(representation_converter_registry&&)                 = delete;
  representation_converter_registry& operator=(representation_converter_registry&&)      = delete;

  ~representation_converter_registry() = default;

  /**
   * @brief Register a converter between two representation types.
   *
   * @code
   * registry.register_converter<SourceType, TargetType>(
   *   [](idata_representation& source,
   *      const memory::memory_space* target_memory_space,
   *      rmm::cuda_stream_view stream,
   *      memory::reservation* reservation) -> std::unique_ptr<idata_representation> {
   *     auto& src = source.cast<SourceType>();
   *     // ... conversion logic ...
   *     return std::make_unique<TargetType>(...);
   *   });
   * @endcode
   *
   * @tparam SourceType The source representation type (must derive from idata_representation)
   * @tparam TargetType The target representation type (must derive from idata_representation)
   * @param converter The converter function to register
   * @throws std::runtime_error if a converter for this type pair already exists
   */
  template <typename SourceType, typename TargetType>
  void register_converter(representation_converter_fn converter)
  {
    static_assert(std::is_base_of_v<idata_representation, SourceType>,
                  "SourceType must derive from idata_representation");
    static_assert(std::is_base_of_v<idata_representation, TargetType>,
                  "TargetType must derive from idata_representation");

    converter_key key{std::type_index(typeid(SourceType)), std::type_index(typeid(TargetType))};
    register_converter_impl(key, std::move(converter));
  }

  /**
   * @brief Check if a converter exists for the given type pair.
   *
   * @tparam SourceType The source representation type
   * @tparam TargetType The target representation type
   * @return true if a converter is registered, false otherwise
   */
  template <typename SourceType, typename TargetType>
  bool has_converter() const
  {
    converter_key key{std::type_index(typeid(SourceType)), std::type_index(typeid(TargetType))};
    return has_converter_impl(key);
  }

  /**
   * @brief Check if a converter exists for the given source instance and target type.
   *
   * @tparam TargetType The target representation type
   * @param source The source data representation
   * @return true if a converter is registered, false otherwise
   */
  template <typename TargetType>
  bool has_converter_for(const idata_representation& source) const
  {
    converter_key key{std::type_index(typeid(source)), std::type_index(typeid(TargetType))};
    return has_converter_impl(key);
  }

  /**
   * @brief Convert a data representation to another type.
   *
   * @tparam TargetType The target representation type
   * @param source The source data representation
   * @param target_memory_space The target memory space for the new representation
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<TargetType> The converted representation
   * @throws std::runtime_error if no converter is registered for the type pair
   *
   * @example
   * auto result = registry.convert<my_target_representation>(source, space);
   */
  template <typename TargetType>
  std::unique_ptr<TargetType> convert(idata_representation& source,
                                      const memory::memory_space* target_memory_space,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    converter_key key{std::type_index(typeid(source)), std::type_index(typeid(TargetType))};
    auto result = convert_impl(key, source, target_memory_space, stream, nullptr);
    return std::unique_ptr<TargetType>(static_cast<TargetType*>(result.release()));
  }

  /**
   * @brief Convert a data representation, drawing the target allocation from a reservation.
   *
   * The target memory space is derived from the reservation (reservation::get_memory_space()) and
   * the reservation is threaded into the converter so the target allocation draws down the
   * reservation instead of committing fresh capacity. Use this overload whenever the caller has
   * already reserved capacity on the target space (avoids double-counting on the HOST tier).
   *
   * @tparam TargetType The target representation type
   * @param source The source data representation
   * @param reservation Caller-owned reservation on the target memory space
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<TargetType> The converted representation
   * @throws std::runtime_error if no converter is registered for the type pair
   */
  template <typename TargetType>
  std::unique_ptr<TargetType> convert(idata_representation& source,
                                      memory::reservation& reservation,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    converter_key key{std::type_index(typeid(source)), std::type_index(typeid(TargetType))};
    auto result = convert_impl(key, source, &reservation.get_memory_space(), stream, &reservation);
    return std::unique_ptr<TargetType>(static_cast<TargetType*>(result.release()));
  }

  /**
   * @brief Convert a data representation using runtime type information.
   *
   * This overload uses the source's runtime type to find the appropriate converter.
   *
   * @param source The source data representation
   * @param target_type The target representation type index
   * @param target_memory_space The target memory space for the new representation
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<idata_representation> The converted representation
   * @throws std::runtime_error if no converter is registered for the type pair
   *
   * @note This runtime-typed overload always allocates without a reservation. Callers that hold a
   *       reservation should use the templated convert<TargetType>(source, reservation, stream).
   */
  std::unique_ptr<idata_representation> convert(
    idata_representation& source,
    std::type_index target_type,
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Unregister a converter for the given type pair.
   *
   * @tparam SourceType The source representation type
   * @tparam TargetType The target representation type
   * @return true if a converter was removed, false if none existed
   */
  template <typename SourceType, typename TargetType>
  bool unregister_converter()
  {
    converter_key key{std::type_index(typeid(SourceType)), std::type_index(typeid(TargetType))};
    return unregister_converter_impl(key);
  }

  /**
   * @brief Clear all registered converters.
   *
   * This is primarily useful for testing purposes.
   */
  void clear();

 private:
  void register_converter_impl(const converter_key& key, representation_converter_fn converter);
  bool has_converter_impl(const converter_key& key) const;
  std::unique_ptr<idata_representation> convert_impl(
    const converter_key& key,
    idata_representation& source,
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream,
    memory::reservation* reservation) const;
  bool unregister_converter_impl(const converter_key& key);

  mutable std::shared_mutex _mutex;
  std::unordered_map<converter_key, representation_converter_fn, converter_key_hash> _converters;
};

}  // namespace cucascade
