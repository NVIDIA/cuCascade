/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/device_arena.hpp>
#include <cucascade/io/cufile/runtime.hpp>
#include <cucascade/log/logging.hpp>

#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cucascade::io::cufile {
namespace {

// Every runtime callback restores the calling thread's CUDA device, including failure paths.
class device_scope {
 public:
  explicit device_scope(int device)
  {
    _valid = cudaGetDevice(&_previous) == cudaSuccess;
    if (_valid && _previous != device) {
      _changed = cudaSetDevice(device) == cudaSuccess;
      _valid   = _changed;
    }
  }
  ~device_scope()
  {
    if (_changed) { static_cast<void>(cudaSetDevice(_previous)); }
  }
  [[nodiscard]] bool valid() const noexcept { return _valid; }

 private:
  int _previous{};
  bool _valid{};
  bool _changed{};
};

template <typename Fn, typename... Args>
bool succeeds(Fn const& fn, Args... args) noexcept
{
  try {
    return fn(args...);
  } catch (...) {
    return false;
  }
}

template <typename T, typename Fn>
std::optional<T> telemetry(Fn const& fn, int device) noexcept
{
  try {
    if (fn) { return fn(device); }
  } catch (...) {
  }
  return std::nullopt;
}

std::size_t floor_slab(std::size_t bytes) noexcept
{
  return bytes - bytes % device_arena::slab_bytes;
}

// NVML is optional at run time. Match CUDA's PCI address, not NVML's ordinal (CUDA_VISIBLE_DEVICES
// may reorder devices). Never shut down NVML or unload it underneath another process component.
std::optional<std::size_t> free_bar1_bytes(int device) noexcept
{
  struct nvml_api {
    void* library{dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE)};
    decltype(&nvmlInit_v2) init{};
    decltype(&nvmlDeviceGetHandleByPciBusId_v2) get_device{};
    decltype(&nvmlDeviceGetBAR1MemoryInfo) get_bar1{};
    bool ready{};
    nvml_api()
    {
      if (library == nullptr) { return; }
      init = reinterpret_cast<decltype(init)>(dlsym(library, "nvmlInit_v2"));
      get_device =
        reinterpret_cast<decltype(get_device)>(dlsym(library, "nvmlDeviceGetHandleByPciBusId_v2"));
      get_bar1 =
        reinterpret_cast<decltype(get_bar1)>(dlsym(library, "nvmlDeviceGetBAR1MemoryInfo"));
      ready =
        init != nullptr && get_device != nullptr && get_bar1 != nullptr && init() == NVML_SUCCESS;
    }
  };
  static nvml_api const api;
  if (!api.ready) { return std::nullopt; }
  char pci[32]{};
  nvmlDevice_t handle{};
  nvmlBAR1Memory_t info{};
  if (cudaDeviceGetPCIBusId(pci, static_cast<int>(sizeof(pci)), device) != cudaSuccess ||
      api.get_device(pci, &handle) != NVML_SUCCESS || api.get_bar1(handle, &info) != NVML_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(
    std::min<unsigned long long>(info.bar1Free, std::numeric_limits<std::size_t>::max()));
}

// cuFile exposes the configured pin limit, not external users' consumption. Account for all
// default arena registrations here; live BAR1 telemetry separately includes other consumers.
struct default_registrations {
  std::mutex mutex;
  std::unordered_map<void*, std::pair<int, std::size_t>> buffers;

  [[nodiscard]] std::size_t used(int device) const noexcept
  {
    std::size_t bytes{};
    for (auto const& [ptr, entry] : buffers) {
      static_cast<void>(ptr);
      if (entry.first == device) { bytes += entry.second; }
    }
    return bytes;
  }

  static default_registrations& instance()
  {
    static auto* const ledger = new default_registrations;
    return *ledger;
  }
};

std::optional<std::size_t> pin_limit(init_report const& report) noexcept
{
  if (!report.properties_available) { return std::nullopt; }
  auto const kib = static_cast<std::size_t>(report.properties.max_device_pinned_mem_size);
  return kib > std::numeric_limits<std::size_t>::max() / 1024
           ? std::numeric_limits<std::size_t>::max()
           : kib * 1024;
}

}  // namespace

namespace detail {

struct arena_registration {
  void* base{};
  std::size_t bytes{};
  bool active{};
};

struct arena_allocation {
  void* base{};
  std::size_t bytes{};
  std::vector<arena_registration> registrations;
};

struct arena_slab {
  void* base{};
  std::size_t allocation{};
  std::size_t allocation_offset{};
  void* registration_base{};
  std::size_t registration_offset{};
  slot_class cls{slot_class::mib1};
  std::array<bool, 16> registered{};
  std::array<bool, 16> used{};
  std::optional<arena_mount_id> owner;
  std::optional<arena_mount_id> borrower;
  unsigned live{};
  bool probe{};
  bool draining{};

  [[nodiscard]] unsigned slots() const noexcept
  {
    return static_cast<unsigned>(device_arena::slab_bytes / class_bytes(cls));
  }
  [[nodiscard]] bool has_registration() const noexcept
  {
    return std::any_of(registered.begin(), registered.end(), [](bool flag) { return flag; });
  }
};

struct arena_quota {
  arena_mount_id mount{};
  std::size_t registered{};
  std::size_t unregistered{};
  slot_class cls{slot_class::mib1};
};

struct device_arena_state {
  explicit device_arena_state(int ordinal, device_arena_hooks callbacks)
    : device(ordinal), hooks(std::move(callbacks))
  {
  }

  int device;
  device_arena_hooks hooks;
  mutable std::mutex mutex;
  std::vector<arena_allocation> allocations;
  std::vector<arena_slab> slabs;
  std::vector<arena_quota> quotas;
  device_arena_options options;
  arena_warmup_report report;
  std::size_t reserved_bytes{};
  std::size_t reclassifications{};
  bool warmed{};
  bool maintenance{};
  bool quota_dirty{};

  [[nodiscard]] bool can_reformat(arena_slab const& slab) const noexcept
  {
    return options.registration != registration_granularity::slot || !slab.has_registration();
  }

  void reformat(arena_slab& slab, slot_class cls) noexcept
  {
    ++reclassifications;
    bool const registered = slab.has_registration();
    slab.cls              = cls;
    slab.registered.fill(false);
    if (registered) { std::fill_n(slab.registered.begin(), slab.slots(), true); }
    slab.draining = false;
    slab.borrower.reset();
  }

  [[nodiscard]] arena_slab* spare(bool registered, slot_class cls) noexcept
  {
    for (auto& slab : slabs) {
      if (!slab.owner && slab.live == 0 && slab.has_registration() == registered &&
          (slab.cls == cls || can_reformat(slab))) {
        return &slab;
      }
    }
    return nullptr;
  }

  [[nodiscard]] std::size_t count(arena_mount_id mount, bool registered) const noexcept
  {
    return static_cast<std::size_t>(std::count_if(slabs.begin(), slabs.end(), [&](auto const& s) {
      return s.owner == mount && s.has_registration() == registered;
    }));
  }

  // Metadata-only: no callback, container growth, or registration on retirement/the hot path.
  void progress() noexcept
  {
    if (maintenance || !quota_dirty) { return; }
    // Two passes let one mount immediately use capacity returned by a later quota entry.
    for (unsigned pass = 0; pass < 2; ++pass) {
      for (auto const& q : quotas) {
        for (bool const registered : {true, false}) {
          auto const desired = registered ? q.registered : q.unregistered;
          auto assigned      = count(q.mount, registered);
          for (auto& slab : slabs) {
            if (slab.owner != q.mount || slab.has_registration() != registered) { continue; }
            bool const excess = assigned > desired;
            bool const wrong  = slab.cls != q.cls;
            if (!excess && !wrong) {
              slab.draining = false;  // A newer quota may have cancelled a drain.
              continue;
            }
            // One draining slab per mount across both tiers.
            bool const another_draining =
              std::any_of(slabs.begin(), slabs.end(), [&](auto const& s) {
                return &s != &slab && s.owner == q.mount && s.draining;
              });
            if (another_draining) { continue; }
            auto* replacement =
              (!excess && wrong && !can_reformat(slab)) ? spare(registered, q.cls) : nullptr;
            if (!excess && wrong && !can_reformat(slab) && replacement == nullptr) {
              slab.draining = false;  // Keep old-class capacity usable; report pending instead.
              continue;
            }
            slab.draining = true;
            if (slab.live != 0) { continue; }
            if (excess || replacement != nullptr) {
              slab.owner.reset();
              slab.draining = false;
              --assigned;
              if (replacement != nullptr) {
                ++reclassifications;
                replacement->owner = q.mount;
                if (replacement->cls != q.cls) { reformat(*replacement, q.cls); }
                ++assigned;
              }
            } else {
              reformat(slab, q.cls);
            }
          }
          while (assigned < desired) {
            auto* slab = spare(registered, q.cls);
            if (slab == nullptr) { break; }
            if (slab->cls != q.cls) { reformat(*slab, q.cls); }
            slab->owner = q.mount;
            ++assigned;
          }
        }
      }
    }
    quota_dirty = std::any_of(quotas.begin(), quotas.end(), [&](auto const& q) {
      return quota_report(q.mount).pending_slabs != 0;
    });
  }

  [[nodiscard]] arena_quota_report quota_report(arena_mount_id mount) const noexcept
  {
    arena_quota_report result;
    auto const q = std::find_if(
      quotas.begin(), quotas.end(), [&](auto const& entry) { return entry.mount == mount; });
    if (q == quotas.end()) { return result; }
    for (auto const& slab : slabs) {
      if (slab.owner != mount) { continue; }
      if (slab.has_registration()) {
        if (!slab.draining && slab.cls == q->cls) {
          result.registered_bytes +=
            static_cast<std::size_t>(
              std::count(slab.registered.begin(), slab.registered.begin() + slab.slots(), true)) *
            class_bytes(slab.cls);
        }
        ++result.registered_slabs;
      } else {
        ++result.unregistered_slabs;
      }
      if (slab.draining) { ++result.draining_slabs; }
      if (slab.cls != q->cls) { ++result.pending_slabs; }
    }
    auto const difference = [](std::size_t a, std::size_t b) { return a > b ? a - b : b - a; };
    result.pending_slabs += difference(result.registered_slabs, q->registered);
    result.pending_slabs += difference(result.unregistered_slabs, q->unregistered);
    return result;
  }
};

}  // namespace detail

device_arena_hooks device_arena_hooks::defaults()
{
  device_arena_hooks hooks;
  hooks.allocate = [](int device, std::size_t bytes) -> void* {
    device_scope scope(device);
    void* ptr{};
    if (!scope.valid() || cudaMalloc(&ptr, bytes) != cudaSuccess) { return nullptr; }
    return ptr;
  };
  hooks.deallocate = [](int device, void* ptr) {
    device_scope scope(device);
    return scope.valid() && cudaFree(ptr) == cudaSuccess;
  };
  hooks.register_buffer = [](int device, void* ptr, std::size_t bytes) {
    device_scope scope(device);
    if (!scope.valid()) { return false; }
    auto const report = runtime::instance().ensure_initialized();
    auto const* api   = runtime::instance().api();
    if (!report.driver_opened || api == nullptr) { return false; }
    auto& ledger = default_registrations::instance();
    std::lock_guard lock(ledger.mutex);
    auto const used = ledger.used(device);
    auto const cap  = pin_limit(report).value_or(std::numeric_limits<std::size_t>::max());
    if (used > cap || bytes > cap - used) { return false; }
    // Allocate bookkeeping before acquiring a registration: throwing hooks acquire nothing.
    if (!ledger.buffers.emplace(ptr, std::pair{device, bytes}).second) { return false; }
    if (api->buf_register(ptr, bytes, 0).err != CU_FILE_SUCCESS) {
      ledger.buffers.erase(ptr);
      return false;
    }
    return true;
  };
  hooks.deregister_buffer = [](int device, void* ptr) {
    device_scope scope(device);
    auto const* api = cufile_api::get();
    if (!scope.valid() || api == nullptr) { return false; }
    auto& ledger = default_registrations::instance();
    std::lock_guard lock(ledger.mutex);
    if (api->buf_deregister(ptr).err != CU_FILE_SUCCESS) { return false; }
    ledger.buffers.erase(ptr);
    return true;
  };
  hooks.memory_info = [](int device) -> std::optional<arena_memory_info> {
    device_scope scope(device);
    std::size_t free{}, total{};
    if (!scope.valid() || cudaMemGetInfo(&free, &total) != cudaSuccess) { return std::nullopt; }
    return arena_memory_info{total, free};
  };
  hooks.free_bar1        = free_bar1_bytes;
  hooks.pinning_headroom = [](int device) -> std::optional<std::size_t> {
    auto const cap = pin_limit(runtime::instance().ensure_initialized());
    if (!cap) { return std::nullopt; }
    auto& ledger = default_registrations::instance();
    std::lock_guard lock(ledger.mutex);
    auto const used = ledger.used(device);
    return *cap > used ? *cap - used : 0;
  };
  return hooks;
}

device_arena& device_arena::for_device(int device)
{
  struct registry {
    std::mutex mutex;
    std::unordered_map<int, std::unique_ptr<device_arena>> arenas;
  };
  static auto* const instance = new registry;  // Intentionally never destroyed, like scratch_pool.
  if (device < 0) { throw std::invalid_argument("device_arena device must be nonnegative"); }
  std::lock_guard lock(instance->mutex);
  auto const found = instance->arenas.find(device);
  if (found != instance->arenas.end()) { return *found->second; }
  auto arena = std::make_unique<device_arena>(device);
  auto* ptr  = arena.get();
  instance->arenas.emplace(device, std::move(arena));
  return *ptr;
}

device_arena::device_arena(int device, device_arena_hooks hooks)
{
  if (device < 0 || !hooks.allocate || !hooks.deallocate || !hooks.register_buffer ||
      !hooks.deregister_buffer ||
      static_cast<bool>(hooks.try_reserve) != static_cast<bool>(hooks.release_reservation)) {
    throw std::invalid_argument("device_arena requires a valid device and paired resource hooks");
  }
  _state = std::make_shared<detail::device_arena_state>(device, std::move(hooks));
}

device_arena::~device_arena() = default;

device_arena_options device_arena_options::from_config(async_engine_config const& config)
{
  config.validate();
  device_arena_options result;
  for (auto mib : config.arena_slot_classes) {
    auto cls = mib == 1   ? slot_class::mib1
               : mib == 2 ? slot_class::mib2
               : mib == 4 ? slot_class::mib4
                          : slot_class::mib8;
    result.slab_classes.push_back(cls);
  }
  result.initial_class = result.slab_classes.front();
  return result;
}

arena_warmup_report device_arena::bootstrap(async_engine_config const& config)
{
  return warm_up(config, device_arena_options::from_config(config));
}

arena_warmup_report device_arena::warm_up(async_engine_config const& config,
                                          device_arena_options const& options)
{
  config.validate();
  if (config.mode == engine_mode::legacy) { return {}; }
  if (class_bytes(options.initial_class) == 0 || !std::isfinite(options.bar1_fraction) ||
      options.bar1_fraction < 0 || options.bar1_fraction > 1 ||
      options.registration_first_step == 0 || options.registration_first_step % slab_bytes != 0 ||
      (options.registration != registration_granularity::arena &&
       options.registration != registration_granularity::slab &&
       options.registration != registration_granularity::slot) ||
      std::any_of(options.slab_classes.begin(), options.slab_classes.end(), [](auto cls) {
        return class_bytes(cls) == 0;
      })) {
    throw std::invalid_argument("device_arena warm-up options are invalid");
  }
  if (options.registration != registration_granularity::slot &&
      !options.allow_shared_registration) {
    throw std::invalid_argument(
      "shared arena/slab registration requires explicit qualification opt-in");
  }
  auto& s = *_state;
  std::lock_guard lock(s.mutex);
  if (s.maintenance) {
    throw std::logic_error("device_arena maintenance must finish before warm-up");
  }
  if (s.warmed) {
    if (s.options != options) {
      CUCASCADE_LOG_WARN("cuFile arena device {}: warm-up options differ; first geometry retained",
                         s.device);
    }
    return s.report;
  }

  auto const memory = telemetry<arena_memory_info>(s.hooks.memory_info, s.device);
  auto const reservable =
    memory ? memory->reservable_bytes : std::numeric_limits<std::size_t>::max();
  std::size_t desired = config.arena_bytes;
  if (desired == 0) {
    desired = memory ? std::min({1UL << 30, memory->total_bytes / 20, reservable}) : 0;
  }
  desired = floor_slab(std::min(desired, reservable));
  if (desired > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("device_arena size must fit off_t");
  }
  // Finish every potentially throwing metadata allocation before acquiring device resources.
  s.options = options;
  s.slabs.clear();
  s.allocations.clear();
  s.slabs.reserve(desired / slab_bytes);
  s.allocations.reserve(
    options.registration == registration_granularity::arena ? 1 : desired / slab_bytes);
  if (options.registration == registration_granularity::arena && desired != 0) {
    detail::arena_allocation allocation;
    allocation.registrations.reserve(desired / slab_bytes);
    s.allocations.push_back(std::move(allocation));
  } else {
    for (std::size_t i = 0; i < desired / slab_bytes; ++i) {
      detail::arena_allocation allocation;
      allocation.registrations.reserve(16);
      s.allocations.push_back(std::move(allocation));
    }
  }
  auto& report             = s.report;
  report                   = {};
  report.allocation_target = desired;
  if (s.hooks.try_reserve && desired != 0) {
    try {
      s.reserved_bytes = s.hooks.try_reserve(s.device, desired);
    } catch (...) {
      ++report.allocation_failures;
      s.reserved_bytes = 0;
    }
    desired = floor_slab(std::min(desired, s.reserved_bytes));
  }
  auto const cls_at = [&](std::size_t index) {
    return options.slab_classes.empty() ? options.initial_class
                                        : options.slab_classes[index % options.slab_classes.size()];
  };
  auto const allocate = [&](std::size_t index, std::size_t bytes) {
    void* ptr{};
    try {
      ptr = s.hooks.allocate(s.device, bytes);
    } catch (...) {
    }
    if (ptr == nullptr) {
      ++report.allocation_failures;
      return false;
    }
    s.allocations[index].base  = ptr;
    s.allocations[index].bytes = bytes;
    report.allocated_bytes += bytes;
    return true;
  };
  if (options.registration == registration_granularity::arena) {
    if (desired != 0 && allocate(0, desired)) {
      for (std::size_t i = 0; i < desired / slab_bytes; ++i) {
        detail::arena_slab slab;
        slab.base              = static_cast<std::byte*>(s.allocations[0].base) + i * slab_bytes;
        slab.allocation_offset = i * slab_bytes;
        slab.cls               = cls_at(i);
        s.slabs.push_back(slab);
      }
    }
  } else {
    for (std::size_t i = 0; i < desired / slab_bytes; ++i) {
      if (!allocate(i, slab_bytes)) { break; }
      detail::arena_slab slab;
      slab.base       = s.allocations[i].base;
      slab.allocation = i;
      slab.cls        = cls_at(i);
      s.slabs.push_back(slab);
    }
  }
  if (s.hooks.release_reservation && s.reserved_bytes > report.allocated_bytes) {
    try {
      s.hooks.release_reservation(s.device, s.reserved_bytes - report.allocated_bytes);
      s.reserved_bytes = report.allocated_bytes;
    } catch (...) {
      // Retain the charge, rather than losing track of a reservation on callback failure.
    }
  }
  auto const bar1    = telemetry<std::size_t>(s.hooks.free_bar1, s.device);
  auto const pin     = telemetry<std::size_t>(s.hooks.pinning_headroom, s.device);
  std::size_t target = std::min(report.allocated_bytes, pin.value_or(report.allocated_bytes));
  report.bar1_known  = bar1.has_value();
  if (bar1) {
    auto const available = *bar1 > options.bar1_headroom ? *bar1 - options.bar1_headroom : 0;
    // Long double avoids out-of-range rounding to SIZE_MAX+1 in a double conversion.
    auto const cap = static_cast<std::size_t>(static_cast<long double>(available) *
                                              static_cast<long double>(options.bar1_fraction));
    target         = std::min(target, cap);
  } else {
    // Unknown BAR1 is not permission to register until cuFile runs out of pinning capacity.
    target = std::min(target, options.registration_first_step);
  }
  report.registration_target = target;
  report.first_registration_step =
    bar1 ? target : std::min(target, options.registration_first_step);
  auto const register_buffer = [&](std::size_t allocation, void* ptr, std::size_t bytes) {
    if (!succeeds(s.hooks.register_buffer, s.device, ptr, bytes)) {
      ++report.registration_failures;
      return false;
    }
    s.allocations[allocation].registrations.push_back({ptr, bytes, true});
    report.registered_bytes += bytes;
    return true;
  };
  if (options.registration == registration_granularity::arena) {
    auto const limit = floor_slab(target);
    std::size_t offset{};
    auto bytes = limit;
    while (offset < limit) {
      auto* base = static_cast<std::byte*>(s.allocations[0].base) + offset;
      if (!register_buffer(0, base, bytes)) {
        // A failed bulk registration acquired nothing. Retry independent slab groups without
        // discarding earlier successes; every resulting lease retains its actual group base.
        if (bytes == slab_bytes) { break; }
        bytes = slab_bytes;
        continue;
      }
      for (std::size_t index = offset / slab_bytes; index < (offset + bytes) / slab_bytes;
           ++index) {
        auto& slab               = s.slabs[index];
        slab.registration_base   = base;
        slab.registration_offset = slab.allocation_offset - offset;
        std::fill_n(slab.registered.begin(), slab.slots(), true);
      }
      offset += bytes;
    }
  } else {
    // Control caller only; unknown BAR1 stops at the conservative first-step cap.
    // Keep every success and stop globally on the first failed registration unit.
    bool failed{};
    for (auto& slab : s.slabs) {
      if (failed) { break; }
      auto const unit =
        options.registration == registration_granularity::slot ? class_bytes(slab.cls) : slab_bytes;
      unsigned const units =
        options.registration == registration_granularity::slot ? slab.slots() : 1;
      for (unsigned slot = 0; slot < units; ++slot) {
        if (unit > target - report.registered_bytes) { break; }
        auto* ptr = static_cast<std::byte*>(slab.base) + static_cast<std::size_t>(slot) * unit;
        if (!register_buffer(slab.allocation, ptr, unit)) {
          failed = true;
          break;
        }
        if (options.registration == registration_granularity::slot) {
          slab.registered[slot] = true;
        } else {
          std::fill_n(slab.registered.begin(), slab.slots(), true);
        }
      }
    }
  }
  s.warmed      = true;
  s.quota_dirty = !s.quotas.empty();
  s.progress();
  return report;
}

std::optional<staging_lease> device_arena::try_lease(arena_mount_id mount,
                                                     std::size_t bytes,
                                                     bool need_registered) noexcept
{
  return try_lease_impl(mount, bytes, need_registered);
}

std::optional<staging_lease> device_arena::try_lease_probe(std::size_t bytes,
                                                           bool need_registered) noexcept
{
  return try_lease_impl(std::nullopt, bytes, need_registered);
}

std::optional<staging_lease> device_arena::try_lease_impl(std::optional<arena_mount_id> mount,
                                                          std::size_t bytes,
                                                          bool need_registered) noexcept
{
  if (bytes == 0 || bytes > class_bytes(slot_class::mib8)) { return std::nullopt; }
  auto& s = *_state;
  std::unique_lock lock(s.mutex, std::try_to_lock);
  if (!lock.owns_lock() || !s.warmed || s.maintenance) { return std::nullopt; }
  if (mount) {
    auto const quota = std::find_if(
      s.quotas.begin(), s.quotas.end(), [&](auto const& q) { return q.mount == *mount; });
    if (quota == s.quotas.end() || (quota->registered == 0 && quota->unregistered == 0)) {
      return std::nullopt;
    }
  }
  std::size_t best = s.slabs.size();
  unsigned best_slot{};
  bool best_owned{};
  for (std::size_t i = 0; i < s.slabs.size(); ++i) {
    auto const& slab = s.slabs[i];
    bool const owned = slab.owner == mount;
    if (slab.draining || slab.probe || class_bytes(slab.cls) < bytes ||
        (!mount && slab.live != 0) ||
        (mount && !owned && slab.live != 0 && slab.borrower != mount)) {
      continue;
    }
    for (unsigned slot = 0; slot < slab.slots(); ++slot) {
      if (slab.used[slot] || slab.registered[slot] != need_registered) { continue; }
      if (best == s.slabs.size() || (owned && !best_owned) ||
          (owned == best_owned && class_bytes(slab.cls) < class_bytes(s.slabs[best].cls))) {
        best       = i;
        best_slot  = slot;
        best_owned = owned;
      }
      break;
    }
  }
  if (best == s.slabs.size()) { return std::nullopt; }
  auto& slab           = s.slabs[best];
  slab.used[best_slot] = true;
  ++slab.live;
  slab.probe = !mount;
  if (mount && !best_owned) { slab.borrower = mount; }
  auto const slot_offset = static_cast<std::size_t>(best_slot) * class_bytes(slab.cls);
  auto* data             = static_cast<std::byte*>(slab.base) + slot_offset;
  void* base             = data;
  std::size_t offset{};
  if (need_registered && s.options.registration == registration_granularity::arena) {
    base   = slab.registration_base;
    offset = slab.registration_offset + slot_offset;
  } else if (need_registered && s.options.registration == registration_granularity::slab) {
    base   = slab.base;
    offset = slot_offset;
  }
  return staging_lease(_state,
                       best,
                       best_slot,
                       data,
                       base,
                       static_cast<off_t>(offset),
                       class_bytes(slab.cls),
                       need_registered);
}

std::uint8_t device_arena::available_class_mask(bool need_registered) const
{
  auto const& s = *_state;
  std::lock_guard lock(s.mutex);
  if (!s.warmed || s.maintenance) { return 0; }
  std::uint8_t result{};
  for (auto const& slab : s.slabs) {
    bool const has_tier = std::any_of(slab.registered.begin(),
                                      slab.registered.begin() + slab.slots(),
                                      [=](bool flag) { return flag == need_registered; });
    if (!has_tier) { continue; }
    // A partially registered slot slab cannot be reformatted, even for its unregistered tail.
    if (s.can_reformat(slab)) { return 0x0f; }
    result = static_cast<std::uint8_t>(result | (1U << static_cast<unsigned>(slab.cls)));
  }
  return result;
}

std::vector<slot_class> device_arena::available_classes(bool need_registered) const
{
  auto const mask = available_class_mask(need_registered);
  std::vector<slot_class> result;
  for (unsigned bit = 0; bit != 4; ++bit) {
    if ((mask & (1U << bit)) != 0) { result.push_back(static_cast<slot_class>(bit)); }
  }
  return result;
}

arena_quota_report device_arena::set_quota(arena_mount_id mount,
                                           std::size_t registered_slabs,
                                           std::size_t unregistered_slabs,
                                           slot_class cls)
{
  if (class_bytes(cls) == 0 || registered_slabs > std::numeric_limits<std::size_t>::max() / 4 ||
      unregistered_slabs > std::numeric_limits<std::size_t>::max() / 4) {
    throw std::invalid_argument("device_arena quota is invalid");
  }
  auto& s = *_state;
  std::lock_guard lock(s.mutex);
  if (s.maintenance) { throw std::logic_error("device_arena is closed for maintenance"); }
  auto q = std::find_if(
    s.quotas.begin(), s.quotas.end(), [&](auto const& entry) { return entry.mount == mount; });
  if (q == s.quotas.end()) {
    s.quotas.push_back({mount, registered_slabs, unregistered_slabs, cls});
  } else {
    *q = {mount, registered_slabs, unregistered_slabs, cls};
  }
  s.quota_dirty = true;
  s.progress();
  return s.quota_report(mount);
}

arena_quota_report device_arena::quota(arena_mount_id mount) const
{
  std::lock_guard lock(_state->mutex);
  return _state->quota_report(mount);
}

arena_gauges device_arena::gauges() const
{
  auto const& s = *_state;
  std::lock_guard lock(s.mutex);
  arena_gauges result;
  result.reserved_bytes    = s.reserved_bytes;
  result.reclassifications = s.reclassifications;
  for (auto const& allocation : s.allocations) {
    if (allocation.base != nullptr) { result.total_bytes += allocation.bytes; }
    for (auto const& registration : allocation.registrations) {
      if (registration.active) {
        result.registered_bytes += registration.bytes;
        ++result.registrations;
      }
    }
  }
  result.unregistered_bytes = result.total_bytes - result.registered_bytes;
  for (auto const& slab : s.slabs) {
    if (slab.base == nullptr) { continue; }
    ++result.slabs;
    auto const registrations =
      static_cast<unsigned>(std::count(slab.registered.begin(), slab.registered.end(), true));
    auto const bit = static_cast<std::uint8_t>(1U << static_cast<unsigned>(slab.cls));
    if (registrations != 0) {
      ++result.registered_slabs;
      result.registered_class_mask |= bit;
    }
    if (registrations != slab.slots()) result.unregistered_class_mask |= bit;
    if (registrations != 0 && registrations != slab.slots()) {
      ++result.partially_registered_slabs;
    }
    result.live_leases += slab.live;
    if (slab.probe) { ++result.live_probe_leases; }
    if (slab.draining) { ++result.draining_slabs; }
  }
  for (auto const& q : s.quotas) {
    auto const pending   = s.quota_report(q.mount).pending_slabs;
    result.pending_slabs = pending > std::numeric_limits<std::size_t>::max() - result.pending_slabs
                             ? std::numeric_limits<std::size_t>::max()
                             : result.pending_slabs + pending;
  }
  return result;
}

void device_arena::release_for_maintenance()
{
  auto& s = *_state;
  std::lock_guard lock(s.mutex);
  if (std::any_of(
        s.slabs.begin(), s.slabs.end(), [](auto const& slab) { return slab.live != 0; })) {
    throw std::logic_error("device_arena maintenance requires complete quiescence");
  }
  s.maintenance = true;
  // Never free an allocation while even one registration in it may still be live.
  for (auto& allocation : s.allocations) {
    for (auto& registration : allocation.registrations) {
      if (!registration.active) { continue; }
      if (!succeeds(s.hooks.deregister_buffer, s.device, registration.base)) {
        throw std::runtime_error(
          "device_arena deregistration failed; resources retained for retry");
      }
      registration.active = false;
      auto const base     = reinterpret_cast<std::uintptr_t>(registration.base);
      for (auto& slab : s.slabs) {
        for (unsigned slot = 0; slot < slab.slots(); ++slot) {
          auto const address = reinterpret_cast<std::uintptr_t>(slab.base) +
                               static_cast<std::size_t>(slot) * class_bytes(slab.cls);
          if (address >= base && address - base < registration.bytes) {
            slab.registered[slot] = false;
          }
        }
      }
    }
  }
  for (auto& allocation : s.allocations) {
    if (allocation.base == nullptr) { continue; }
    if (!succeeds(s.hooks.deallocate, s.device, allocation.base)) {
      throw std::runtime_error("device_arena deallocation failed; resources retained for retry");
    }
    auto const index = static_cast<std::size_t>(&allocation - s.allocations.data());
    for (auto& slab : s.slabs) {
      if (slab.allocation == index) { slab.base = nullptr; }
    }
    allocation.base = nullptr;
  }
  if (s.hooks.release_reservation && s.reserved_bytes != 0) {
    s.hooks.release_reservation(s.device, s.reserved_bytes);
    s.reserved_bytes = 0;
  }
  s.slabs.clear();
  s.allocations.clear();
  s.quotas.clear();
  s.report      = {};
  s.warmed      = false;
  s.maintenance = false;
  s.quota_dirty = false;
}

staging_lease::staging_lease(std::shared_ptr<detail::device_arena_state> state,
                             std::size_t slab,
                             unsigned slot,
                             void* data,
                             void* base,
                             off_t offset,
                             std::size_t capacity,
                             bool registered) noexcept
  : _state(std::move(state)),
    _slab(slab),
    _slot(slot),
    _data(data),
    _base(base),
    _offset(offset),
    _capacity(capacity),
    _registered(registered)
{
}

staging_lease::~staging_lease() { reset(); }

staging_lease::staging_lease(staging_lease&& other) noexcept { *this = std::move(other); }

staging_lease& staging_lease::operator=(staging_lease&& other) noexcept
{
  if (this != &other) {
    reset();
    _state      = std::move(other._state);
    _slab       = other._slab;
    _slot       = other._slot;
    _data       = std::exchange(other._data, nullptr);
    _base       = std::exchange(other._base, nullptr);
    _offset     = std::exchange(other._offset, 0);
    _capacity   = std::exchange(other._capacity, 0);
    _registered = std::exchange(other._registered, false);
  }
  return *this;
}

bool staging_lease::try_reset() noexcept
{
  if (!_state) return true;
  if (_state.use_count() == 1) return false;
  std::unique_lock lock(_state->mutex, std::try_to_lock);
  if (!lock) return false;
  auto& slab       = _state->slabs[_slab];
  slab.used[_slot] = false;
  --slab.live;
  if (slab.live == 0) {
    slab.borrower.reset();
    slab.probe = false;
    if (_state->quota_dirty) _state->progress();
  }
  lock.unlock();
  _state.reset();
  _data = _base = nullptr;
  _offset       = 0;
  _capacity     = 0;
  _registered   = false;
  return true;
}

void staging_lease::reset() noexcept
{
  if (!_state) { return; }
  {
    std::lock_guard lock(_state->mutex);
    auto& slab       = _state->slabs[_slab];
    slab.used[_slot] = false;
    --slab.live;
    if (slab.live == 0) {
      slab.borrower.reset();
      slab.probe = false;
      if (_state->quota_dirty) { _state->progress(); }
    }
  }
  _state.reset();
  _data       = nullptr;
  _base       = nullptr;
  _offset     = 0;
  _capacity   = 0;
  _registered = false;
}

}  // namespace cucascade::io::cufile
