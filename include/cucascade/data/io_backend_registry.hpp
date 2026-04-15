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

#include <cucascade/data/disk_io_backend.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace cucascade {

/**
 * @brief Type alias for a factory function that creates an I/O backend instance.
 *
 * Each call should return a new instance. Backends may have internal state
 * (staging buffers, CUDA streams) that should not be shared across unrelated contexts.
 */
using io_backend_factory_fn = std::function<std::shared_ptr<idisk_io_backend>()>;

/**
 * @brief Registry for pluggable disk I/O backends.
 *
 * This class provides a central registry where I/O backends are registered by name
 * and created on demand via factory functions. This allows external users to implement
 * and register their own backends alongside the built-in pipeline backend.
 *
 * The registry is thread-safe for concurrent registration and lookup.
 *
 * @note Built-in backends are registered via register_builtin_io_backends().
 */
class io_backend_registry {
 public:
  /**
   * @brief Construct an empty I/O backend registry.
   */
  io_backend_registry() = default;

  // Non-copyable, non-movable
  io_backend_registry(const io_backend_registry&)            = delete;
  io_backend_registry& operator=(const io_backend_registry&) = delete;
  io_backend_registry(io_backend_registry&&)                 = delete;
  io_backend_registry& operator=(io_backend_registry&&)      = delete;

  ~io_backend_registry() = default;

  /**
   * @brief Register an I/O backend factory under the given name.
   *
   * @param name Unique name for the backend (e.g., "pipeline", "my_custom_backend").
   * @param factory Factory function that creates a new backend instance.
   * @throws std::runtime_error if a backend with this name is already registered.
   */
  void register_backend(const std::string& name, io_backend_factory_fn factory);

  /**
   * @brief Check if a backend is registered under the given name.
   *
   * @param name The backend name to look up.
   * @return true if a backend is registered, false otherwise.
   */
  [[nodiscard]] bool has_backend(const std::string& name) const;

  /**
   * @brief Create a new I/O backend instance by name.
   *
   * @param name The name of the backend to create.
   * @return A shared_ptr to the newly created backend instance.
   * @throws std::runtime_error if no backend is registered under this name.
   */
  [[nodiscard]] std::shared_ptr<idisk_io_backend> create_backend(const std::string& name) const;

  /**
   * @brief Unregister a backend by name.
   *
   * @param name The backend name to remove.
   * @return true if a backend was removed, false if none existed.
   */
  bool unregister_backend(const std::string& name);

  /**
   * @brief Clear all registered backends and reset the default name to "pipeline".
   */
  void clear();

  /**
   * @brief Set the default backend name.
   *
   * When creating disk memory spaces without specifying a backend, the default
   * is used. The named backend must already be registered.
   *
   * @param name The name of a registered backend to use as default.
   * @throws std::runtime_error if no backend is registered under this name.
   */
  void set_default(const std::string& name);

  /**
   * @brief Get the current default backend name.
   *
   * @return The name of the default backend (initially "pipeline").
   */
  [[nodiscard]] std::string get_default_name() const;

  /**
   * @brief Create a new instance of the default backend.
   *
   * Equivalent to `create_backend(get_default_name())`.
   *
   * @return A shared_ptr to the newly created default backend instance.
   * @throws std::runtime_error if the default backend is not registered.
   */
  [[nodiscard]] std::shared_ptr<idisk_io_backend> create_default_backend() const;

 private:
  mutable std::mutex _mutex;
  std::string _default_name{"pipeline"};
  std::unordered_map<std::string, io_backend_factory_fn> _factories;
};

/**
 * @brief Register the built-in I/O backends (pipeline).
 *
 * @param registry The registry to register backends with.
 */
void register_builtin_io_backends(io_backend_registry& registry);

}  // namespace cucascade
