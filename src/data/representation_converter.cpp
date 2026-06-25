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

#include <cucascade/data/representation_converter.hpp>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>

namespace cucascade {

// =============================================================================
// representation_converter_registry implementation
//
// cuCascade core ships no built-in converters; only the registry machinery lives here.
// Concrete GPU/HOST/DISK converters are provided by the cucascade-cudf layer
// (src/cudf/representation_converter_builtins.cpp).
// =============================================================================

void representation_converter_registry::register_converter_impl(
  const converter_key& key, representation_converter_fn converter)
{
  std::unique_lock lock(_mutex);

  if (_converters.find(key) != _converters.end()) {
    std::ostringstream oss;
    oss << "Converter already registered for source type '" << key.source_type.name()
        << "' to target type '" << key.target_type.name() << "'";
    throw std::runtime_error(oss.str());
  }

  _converters.emplace(key, std::move(converter));
}

bool representation_converter_registry::has_converter_impl(const converter_key& key) const
{
  std::shared_lock lock(_mutex);
  return _converters.find(key) != _converters.end();
}

std::unique_ptr<idata_representation> representation_converter_registry::convert_impl(
  const converter_key& key,
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  memory::reservation* reservation) const
{
  representation_converter_fn converter;
  {
    std::shared_lock lock(_mutex);

    auto it = _converters.find(key);
    if (it == _converters.end()) {
      std::ostringstream oss;
      oss << "No converter registered for source type '" << key.source_type.name()
          << "' to target type '" << key.target_type.name() << "'";
      throw std::runtime_error(oss.str());
    }

    converter = it->second;
  }

  return converter(source, target_memory_space, stream, reservation);
}

std::unique_ptr<idata_representation> representation_converter_registry::convert(
  idata_representation& source,
  std::type_index target_type,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  memory::reservation* reservation) const
{
  converter_key key{std::type_index(typeid(source)), target_type};
  return convert_impl(key, source, target_memory_space, stream, reservation);
}

bool representation_converter_registry::unregister_converter_impl(const converter_key& key)
{
  std::unique_lock lock(_mutex);
  return _converters.erase(key) > 0;
}

void representation_converter_registry::clear()
{
  std::unique_lock lock(_mutex);
  _converters.clear();
}

}  // namespace cucascade
