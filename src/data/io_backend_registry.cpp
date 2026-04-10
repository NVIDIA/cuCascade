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

#include "io_backend_internal.hpp"

#include <cucascade/data/io_backend_registry.hpp>
#include <cucascade/error.hpp>

#include <sstream>

namespace cucascade {

void io_backend_registry::register_backend(const std::string& name, io_backend_factory_fn factory)
{
  std::lock_guard<std::mutex> lock(_mutex);

  if (_factories.find(name) != _factories.end()) {
    std::ostringstream oss;
    oss << "I/O backend already registered with name '" << name << "'";
    throw std::runtime_error(oss.str());
  }

  _factories.emplace(name, std::move(factory));
}

bool io_backend_registry::has_backend(const std::string& name) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _factories.find(name) != _factories.end();
}

std::shared_ptr<idisk_io_backend> io_backend_registry::create_backend(const std::string& name) const
{
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _factories.find(name);
  if (it == _factories.end()) {
    std::ostringstream oss;
    oss << "No I/O backend registered with name '" << name << "'";
    throw std::runtime_error(oss.str());
  }

  return it->second();
}

bool io_backend_registry::unregister_backend(const std::string& name)
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _factories.erase(name) > 0;
}

void io_backend_registry::clear()
{
  std::lock_guard<std::mutex> lock(_mutex);
  _factories.clear();
  _default_name = "pipeline";
}

void io_backend_registry::set_default(const std::string& name)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_factories.find(name) == _factories.end()) {
    std::ostringstream oss;
    oss << "Cannot set default: no I/O backend registered with name '" << name << "'";
    throw std::runtime_error(oss.str());
  }
  _default_name = name;
}

std::string io_backend_registry::get_default_name() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _default_name;
}

std::shared_ptr<idisk_io_backend> io_backend_registry::create_default_backend() const
{
  return create_backend(get_default_name());
}

void register_builtin_io_backends(io_backend_registry& registry)
{
  registry.register_backend(
    "pipeline", []() -> std::shared_ptr<idisk_io_backend> { return make_pipeline_io_backend(); });
}

}  // namespace cucascade
