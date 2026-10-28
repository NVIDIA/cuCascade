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

#include "remote_io.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade {
namespace detail {
namespace {

/// S3's part-count and part-size limits (the REST context sizes parts the same way).
constexpr std::size_t max_parts      = 10000;
constexpr std::size_t max_part_bytes = std::size_t{5} << 30;

/// Source of the zeros written into the gaps between buffers (at most DISK_FILE_ALIGNMENT - 1
/// bytes each in the batch layout; longer gaps reuse it piece by piece).
constexpr std::size_t zero_page_bytes = std::size_t{64} << 10;
alignas(4096) constexpr std::array<std::uint8_t, zero_page_bytes> zero_page{};

[[nodiscard]] std::size_t round_up(std::size_t value, std::size_t unit) noexcept
{
  return unit == 0 ? value : (value + unit - 1) / unit * unit;
}

/**
 * @brief Holds a request's future and waits for it if the holder is destroyed unconsumed (see
 *        disk_io.cpp), so no request outlives the buffers a helper was handed.
 */
class await_on_unwind {
 public:
  explicit await_on_unwind(exec::semi_future<std::size_t>&& future) noexcept
    : _future(std::move(future))
  {
  }

  ~await_on_unwind()
  {
    if (!_future.valid()) { return; }
    try {
      _future.wait();
    } catch (...) {
      // Already unwinding from another error.
    }
  }

  await_on_unwind(await_on_unwind const&)            = delete;
  await_on_unwind& operator=(await_on_unwind const&) = delete;
  await_on_unwind(await_on_unwind&&)                 = delete;
  await_on_unwind& operator=(await_on_unwind&&)      = delete;

  [[nodiscard]] std::size_t get() { return std::move(_future).get(); }

 private:
  exec::semi_future<std::size_t> _future;
};

void check_transferred(std::size_t transferred,
                       std::size_t expected,
                       char const* what,
                       std::string const& uri)
{
  if (transferred != expected) {
    throw std::runtime_error(std::string("remote tier: ") + what + " of '" + uri +
                             "' transferred " + std::to_string(transferred) + " of " +
                             std::to_string(expected) + " bytes");
  }
}

/// One piece of an object: a source range, from the caller (host or device) or from the zeros.
struct write_piece {
  std::size_t offset{0};
  std::size_t size{0};
  std::uint8_t const* src{nullptr};
  bool device{false};
};

/// @p slices plus zero pieces for every gap of [0, object_size), in offset order.
[[nodiscard]] std::vector<write_piece> cover(std::span<io::write_slice const> slices,
                                             std::size_t object_size,
                                             bool device)
{
  std::vector<write_piece> sources;
  sources.reserve(slices.size());
  for (auto const& slice : slices) {
    if (slice.size() == 0) { continue; }
    if (slice.offset() > object_size || slice.size() > object_size - slice.offset()) {
      throw std::invalid_argument("remote tier: a write slice lies past the object size");
    }
    sources.push_back(write_piece{slice.offset(), slice.size(), slice.src, device});
  }
  std::sort(sources.begin(), sources.end(), [](write_piece const& a, write_piece const& b) {
    return a.offset < b.offset;
  });
  std::vector<write_piece> pieces;
  pieces.reserve(2 * sources.size() + 1);
  std::size_t cursor = 0;
  auto zeros_until   = [&](std::size_t end) {
    while (cursor < end) {
      auto const n = std::min(end - cursor, zero_page_bytes);
      pieces.push_back(write_piece{cursor, n, zero_page.data(), false});
      cursor += n;
    }
  };
  for (auto const& source : sources) {
    if (source.offset < cursor) {
      throw std::invalid_argument("remote tier: overlapping write slices");
    }
    zeros_until(source.offset);
    pieces.push_back(source);
    cursor = source.offset + source.size;
  }
  zeros_until(object_size);
  return pieces;
}

/// Write [begin, end) of the object: the parts of @p pieces inside it, as one request.
void write_window(io::ioctx& ctx,
                  io::io_object const& object,
                  std::string const& uri,
                  std::span<write_piece const> pieces,
                  std::size_t begin,
                  std::size_t end,
                  ::cuda::stream_ref stream)
{
  std::vector<io::write_slice> host;
  std::vector<io::write_slice> device;
  // First piece ending after `begin` (pieces are contiguous and sorted).
  auto it = std::partition_point(pieces.begin(), pieces.end(), [begin](write_piece const& p) {
    return p.offset + p.size <= begin;
  });
  for (; it != pieces.end() && it->offset < end; ++it) {
    auto const lo   = std::max(it->offset, begin);
    auto const hi   = std::min(it->offset + it->size, end);
    auto const* src = it->src == zero_page.data() ? zero_page.data() : it->src + (lo - it->offset);
    (it->device ? device : host).emplace_back(lo, hi - lo, src);
  }
  await_on_unwind request{ctx.mixed_writev_async(object, host, device, stream)};
  check_transferred(request.get(), end - begin, "write", uri);
}

/// Write every byte of the object and commit it; throws on any failure (the caller destroys
/// the unflushed object, which aborts its upload).
void write_and_commit(io::ioctx& ctx,
                      io::io_object const& object,
                      std::string const& uri,
                      std::span<write_piece const> pieces,
                      std::size_t object_size,
                      std::size_t window,
                      ::cuda::stream_ref stream)
{
  for (std::size_t begin = 0; begin < object_size; begin += window) {
    write_window(ctx, object, uri, pieces, begin, std::min(object_size, begin + window), stream);
  }
  ctx.flush(object);
}

/**
 * @brief Keep @p context's reference forever without allocating: the shared_ptr is moved into
 *        storage that is never destroyed. Used where releasing it could drop the last reference
 *        on one of the context's reactor threads (which would join itself), and nothing else can
 *        hold it.
 */
void keep_alive_forever(std::shared_ptr<io::ioctx>&& context) noexcept
{
  if (context == nullptr) { return; }
  alignas(std::shared_ptr<io::ioctx>) unsigned char storage[sizeof(std::shared_ptr<io::ioctx>)];
  ::new (static_cast<void*>(storage)) std::shared_ptr<io::ioctx>(std::move(context));
}

/// Rate limit of the delete-failure warnings: the first few, then one in a thousand.
[[nodiscard]] bool should_warn_delete_failure() noexcept
{
  static std::atomic<std::uint64_t> failures{0};
  auto const n = failures.fetch_add(1, std::memory_order_relaxed);
  return n < 8 || n % 1000 == 0;
}

void warn_delete_failure(std::string const& uri, std::exception_ptr const& error) noexcept
{
  if (!should_warn_delete_failure()) { return; }
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    CUCASCADE_LOG_WARN(
      "remote tier: could not delete '{}' ({}); the bucket's lifecycle rule must collect it",
      uri,
      e.what());
  } catch (...) {
    CUCASCADE_LOG_WARN(
      "remote tier: could not delete '{}'; the bucket's lifecycle rule must collect it", uri);
  }
}

[[nodiscard]] std::size_t total_bytes(std::span<io::slice const> slices) noexcept
{
  std::size_t bytes = 0;
  for (auto const& s : slices) {
    bytes += s.size();
  }
  return bytes;
}

}  // namespace

// =============================================================================
// remote_rest_ioctx
// =============================================================================

remote_rest_ioctx::remote_rest_ioctx(std::size_t n_reactors,
                                     std::shared_ptr<io::rest::rest_reactor::reactor_context> ctx,
                                     std::size_t staging_bytes,
                                     std::size_t staging_block)
  : io::rest::rest_ioctx(n_reactors, std::move(ctx)), _budget(staging_bytes), _block(staging_block)
{
}

std::size_t remote_rest_ioctx::device_read_staging(std::size_t bytes,
                                                   std::size_t part) const noexcept
{
  if (bytes == 0) { return 0; }
  auto const block = std::max<std::size_t>(_block, 1);
  // GETs: the transport's own bound for a read of `bytes` (its effective segment bounds), plus
  // one per object part boundary crossed (a GET never spans two parts). Each allocates whole
  // blocks: less than one block of waste.
  auto const gets = io::rest::max_read_gets(_config, bytes) + (part != 0 ? bytes / part + 1 : 0);
  return bytes + gets * block;
}

std::size_t remote_rest_ioctx::part_bytes(std::size_t object_size) const noexcept
{
  auto const granule = io::write_granule();
  auto const split   = write_split_bytes();
  auto const piece   = std::max(granule, split - split % granule);
  auto part          = piece;
  if (object_size != 0) {
    auto const need = (object_size + max_parts - 1) / max_parts;
    if (need > part) { part = round_up(need, piece); }
  }
  return std::min(part, max_part_bytes);
}

// =============================================================================
// remote_object
// =============================================================================

remote_object::remote_object(std::shared_ptr<io::ioctx> context_,
                             std::shared_ptr<io::io_object> object_,
                             std::string uri_,
                             std::size_t size_) noexcept
  : context(std::move(context_)), object(std::move(object_)), uri(std::move(uri_)), size(size_)
{
}

remote_object::~remote_object() noexcept
{
  if (context == nullptr) { return; }
  // Never any I/O or blocking here: this may run on a reactor thread of `context` itself (a
  // representation released from a completion callback). The removal queue issues the delete
  // from its own thread and owns the context until the delete settled. It exists: it is created
  // before any object is written.
  if (auto* queue = removal_queue::existing(); queue != nullptr) {
    queue->submit(std::move(context), std::move(object), std::move(uri));
  } else {
    keep_alive_forever(std::move(context));
  }
}

// =============================================================================
// removal_queue
// =============================================================================

namespace {
std::atomic<removal_queue*> the_removal_queue{nullptr};
}  // namespace

removal_queue* removal_queue::existing() noexcept
{
  return the_removal_queue.load(std::memory_order_acquire);
}

removal_queue& removal_queue::instance()
{
  if (auto* queue = existing(); queue != nullptr) { return *queue; }
  // Leaked on purpose: a representation may be released during static destruction, after a
  // function-local static would be gone (the mutex below is only used before that). Normal exit
  // drains it (std::atexit, registered once it exists, so it runs before the CUDA runtime's own
  // teardown registered earlier).
  static std::mutex creation;
  std::lock_guard lock(creation);
  if (auto* queue = existing(); queue != nullptr) { return *queue; }
  auto* created = new removal_queue();
  the_removal_queue.store(created, std::memory_order_release);
  static_cast<void>(std::atexit([] {
    if (auto* queue = removal_queue::existing(); queue != nullptr) { queue->drain(); }
  }));
  return *created;
}

removal_queue::removal_queue()
  : _abandoned_error(
      std::make_exception_ptr(std::system_error(make_error_code(std::errc::operation_canceled),
                                                "remote tier: removal abandoned at shutdown"))),
    _worker([this](std::stop_token stop) { run(stop); })
{
}

void removal_queue::submit(std::shared_ptr<io::ioctx> context,
                           std::shared_ptr<io::io_object> object,
                           std::string uri) noexcept
{
  // Allocate everything before taking the context: a failure then leaves it with us, never
  // released here.
  std::list<entry> node;
  try {
    auto state = std::make_shared<entry_state>();
    node.emplace_back();
    node.back().state = std::move(state);
  } catch (...) {
    keep_alive_forever(std::move(context));
    warn_delete_failure(uri, std::current_exception());
    return;
  }
  auto& e   = node.back();
  e.key     = context.get();
  e.context = std::move(context);
  e.object  = std::move(object);
  e.uri     = std::move(uri);
  try {
    std::lock_guard lock(_mutex);
    // Counted before the splice: if the count cannot be inserted, nothing was queued.
    ++_outstanding_by_context[e.key];
    _queued.splice(_queued.end(), node);
    ++_outstanding;
  } catch (...) {
    keep_alive_forever(std::move(e.context));
    warn_delete_failure(e.uri, std::current_exception());
    return;
  }
  _wake.notify_all();
}

void removal_queue::drain() noexcept
{
  try {
    std::unique_lock lock(_mutex);
    _idle.wait(lock, [this] { return _outstanding == 0; });
  } catch (...) {
    // Nothing to wait with: give up waiting rather than terminate.
  }
}

bool removal_queue::abandoned(io::ioctx const* key) const noexcept
{
  for (auto const* scope = _abandoned; scope != nullptr; scope = scope->next) {
    if (std::find(scope->keys.begin(), scope->keys.end(), key) != scope->keys.end()) {
      return true;
    }
  }
  return false;
}

bool removal_queue::settled_for(std::span<io::ioctx const* const> keys) const noexcept
{
  return std::none_of(keys.begin(), keys.end(), [this](io::ioctx const* key) {
    return _outstanding_by_context.contains(key);
  });
}

void removal_queue::forget(std::list<entry> const& released) noexcept
{
  for (auto const& e : released) {
    --_outstanding;
    if (auto it = _outstanding_by_context.find(e.key);
        it != _outstanding_by_context.end() && --it->second == 0) {
      _outstanding_by_context.erase(it);
    }
  }
}

removal_queue::drain_outcome removal_queue::drain(
  std::span<std::shared_ptr<io::ioctx> const> contexts, std::chrono::milliseconds timeout) noexcept
{
  // 1. The keys, allocated before the lock is taken.
  std::vector<io::ioctx const*> keys;
  try {
    keys.reserve(contexts.size());
    for (auto const& context : contexts) {
      if (context != nullptr && std::find(keys.begin(), keys.end(), context.get()) == keys.end()) {
        keys.push_back(context.get());
      }
    }
  } catch (...) {
    CUCASCADE_LOG_ERROR(
      "remote tier: cannot drain the removals of {} io context(s): out of memory; they are left "
      "to the exit drain",
      contexts.size());
    return {0, true};
  }
  if (keys.empty()) { return {}; }
  std::span<io::ioctx const* const> const scope_keys{keys};
  // wait_for adds the timeout to now(): keep it far from overflowing.
  auto const wait = std::clamp(
    timeout, std::chrono::milliseconds{0}, std::chrono::milliseconds{std::chrono::hours{24 * 365}});
  // Step 5 runs between linking a scope into _abandoned and unlinking it; that window is safe
  // only because shutting a context down cannot throw.
  static_assert(noexcept(std::declval<io::ioctx&>().shutdown()));
  std::size_t n_abandoned = 0;
  try {
    std::unique_lock lock(_mutex);
    // 2. Settled in time: cancel nothing.
    if (_idle.wait_for(lock, wait, [&] { return settled_for(scope_keys); })) { return {}; }

    std::list<entry> taken;
    {
      // 3. From now on the worker issues no removal of these contexts and leaves their settled
      //    entries to us.
      abandon_scope scope{scope_keys, _abandoned};
      _abandoned = &scope;
      // 8. ... until this block is left, however it is left: unlinked with _mutex held, so the
      //    worker never walks this frame after it is gone.
      struct unlink_on_exit {
        removal_queue& queue;
        abandon_scope const& scope;
        std::unique_lock<std::mutex>& lock;
        ~unlink_on_exit()  // noexcept: if _mutex cannot be retaken, terminating beats returning
        {
          if (!lock.owns_lock()) { lock.lock(); }
          for (auto** link = &queue._abandoned; *link != nullptr; link = &(*link)->next) {
            if (*link == &scope) {
              *link = scope.next;
              break;
            }
          }
          queue._idle.notify_all();
        }
      } const unlink{*this, scope, lock};

      // 4. ... and is not issuing one right now (posting to a running context never blocks), so
      //    no removal of these contexts can be posted after they are shut down (which would run
      //    it as a blocking call on the worker).
      _idle.wait(lock, [&] {
        return _issuing == nullptr ||
               std::find(scope_keys.begin(), scope_keys.end(), _issuing) == scope_keys.end();
      });
      lock.unlock();

      // 5. Shut the contexts down: every issued removal of theirs settles (its callback takes
      //    _mutex, not held here). The caller's references keep them alive.
      for (auto const& context : contexts) {
        if (context != nullptr) { context->shutdown(); }
      }

      // 6. Take every entry of these contexts and release it here, whatever the worker is doing.
      lock.lock();
      auto take = [&](std::list<entry>& from, bool issued) {
        for (auto it = from.begin(); it != from.end();) {
          auto const next = std::next(it);
          if (std::find(scope_keys.begin(), scope_keys.end(), it->key) != scope_keys.end()) {
            // Shutdown settles every issued call; a removal still pending is released anyway
            // (its callback only holds its state).
            assert(!issued || it->state->done.load(std::memory_order_acquire));
            if (!issued || it->state->error != nullptr) { ++n_abandoned; }
            taken.splice(taken.end(), from, it);
          }
          it = next;
        }
      };
      take(_queued, /*issued=*/false);
      take(_inflight, /*issued=*/true);
      lock.unlock();
      // The contexts are shut down: dropping a reference joins nothing.
      for (auto& e : taken) {
        e.object.reset();
        e.context.reset();
      }
      lock.lock();
      forget(taken);
      _idle.notify_all();

      // 7. Entries the worker took before step 3 and is releasing now.
      _idle.wait(lock, [&] { return settled_for(scope_keys); });
    }
    lock.unlock();
    taken.clear();
  } catch (...) {
    // Waiting or locking failed (it does not with a std::mutex in practice).
    return {n_abandoned, true};
  }
  if (n_abandoned > 0) {
    try {
      CUCASCADE_LOG_WARN(
        "remote tier: {} REMOTE object removal(s) abandoned at shutdown after {} ms; the objects "
        "are left to the bucket's lifecycle rule",
        n_abandoned,
        timeout.count());
    } catch (...) {
    }
  }
  return {n_abandoned, true};
}

void removal_queue::run(std::stop_token const& stop) noexcept
{
  while (!stop.stop_requested()) {
    try {
      pass(stop);
    } catch (...) {
      // Allocation or synchronization failure: retry shortly rather than terminate.
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
  }
}

void removal_queue::issue(entry& e) noexcept
{
  try {
    auto removal = e.context->remove_io_object_async(e.uri, io::io_priority::low);
    std::move(removal).install_callback(
      [this, state = e.state](exec::try_t<void>&& outcome) noexcept {
        if (outcome.has_exception()) { state->error = std::move(outcome).exception(); }
        state->done.store(true, std::memory_order_release);
        try {
          // Taken and released so the notification cannot fall between the worker's check of
          // the predicate and its wait.
          std::lock_guard lock(_mutex);
        } catch (...) {
        }
        _wake.notify_all();
      });
  } catch (...) {
    e.state->error = std::current_exception();
    e.state->done.store(true, std::memory_order_release);
  }
}

void removal_queue::pass(std::stop_token const& stop)
{
  // A settled entry of a context a scoped drain is cancelling is left to that drain.
  auto const releasable = [this](entry const& e) {
    return e.state->done.load(std::memory_order_acquire) && !abandoned(e.key);
  };
  std::unique_lock lock(_mutex);
  // The predicate is evaluated under the lock before every wait, so a callback that settled an
  // entry before this point is never missed.
  _wake.wait(lock, stop, [&] {
    return !_queued.empty() || std::any_of(_inflight.begin(), _inflight.end(), releasable);
  });
  if (stop.stop_requested()) { return; }
  // At most the removals queued now are issued by this pass, so sustained submissions cannot
  // starve the release of settled ones.
  std::size_t to_issue = _queued.size();

  // First release what settled, before issuing anything: an issue may block (a removal through a
  // context that was shut down runs on a private ring), and must not hold released entries back.
  // Released on this thread: dropping the last reference to a context joins its reactors, which
  // must never happen on one of them.
  std::list<entry> settled;
  for (auto it = _inflight.begin(); it != _inflight.end();) {
    auto const next = std::next(it);
    if (releasable(*it)) { settled.splice(settled.end(), _inflight, it); }
    it = next;
  }
  if (!settled.empty()) {
    lock.unlock();
    for (auto& e : settled) {
      if (e.state->error != nullptr) { warn_delete_failure(e.uri, e.state->error); }
      e.object.reset();
      e.context.reset();
    }
    lock.lock();
    forget(settled);
    _idle.notify_all();
    lock.unlock();
    settled.clear();
    lock.lock();
  }

  // Then issue, one at a time; an entry is in _inflight from before it is issued.
  while (to_issue > 0 && !_queued.empty()) {
    --to_issue;
    auto const it = _queued.begin();
    _inflight.splice(_inflight.end(), _queued, it);
    entry& e = *it;
    if (abandoned(e.key)) {
      // Its context is being shut down by a scoped drain, which releases the entry.
      e.state->error = _abandoned_error;
      e.state->done.store(true, std::memory_order_release);
      continue;
    }
    // While _issuing names it, no scoped drain takes this entry or shuts its context down.
    _issuing = e.key;
    lock.unlock();
    issue(e);
    lock.lock();
    _issuing = nullptr;
    _idle.notify_all();
  }
}

// =============================================================================
// helpers
// =============================================================================

std::shared_ptr<io::ioctx> require_remote_io_context(memory::memory_space const& remote_space,
                                                     bool device)
{
  auto context = remote_space.get_io_context();
  if (context == nullptr) {
    throw std::logic_error("remote tier: the REMOTE memory space " + remote_space.to_string() +
                           " has no io context (see make_remote_io_context)");
  }
  if (!context->supports_write()) {
    throw std::logic_error("remote tier: the io context of the REMOTE memory space cannot write");
  }
  if (device && !context->supports_device_write()) {
    throw std::logic_error(
      "remote tier: the io context of the REMOTE memory space cannot write device memory");
  }
  return context;
}

std::shared_ptr<remote_object const> write_remote_object(
  std::shared_ptr<io::ioctx> context,
  std::string_view prefix,
  std::span<io::write_slice const> slices,
  std::size_t object_size,
  std::optional<::cuda::stream_ref> device_stream)
{
  // The queue that will remove the object exists before the object does (creating it may throw;
  // here, on the converter's thread, that simply fails the spill).
  static_cast<void>(removal_queue::instance());
  auto const pieces = cover(slices, object_size, device_stream.has_value());
  auto const stream = device_stream.value_or(::cuda::stream_ref{cudaStream_t{nullptr}});

  // Stay inside the staging of a context built by make_remote_io_context(): reserve what the
  // window's parts can hold, for the whole write and commit.
  auto* const owned  = dynamic_cast<remote_rest_ioctx*>(context.get());
  std::size_t window = object_size;
  exec::admission_control::slot staging;
  if (owned != nullptr && object_size != 0) {
    auto const part       = owned->part_bytes(object_size);
    auto const part_alloc = round_up(part, owned->staging_block());
    auto const parts_per_window =
      std::max<std::size_t>(1, owned->staging_bytes() / 2 / std::max<std::size_t>(part_alloc, 1));
    window                = parts_per_window * part;
    auto const parts_held = std::min(parts_per_window, (object_size + part - 1) / part);
    staging               = owned->staging_budget().acquire(parts_held * part_alloc);
  }

  io::write_open_options options;
  options.mode      = io::write_mode::create_new;
  options.size_hint = object_size;
  // A name collides only with another writer that drew the same random tag in the same run
  // sequence slot (or a leftover of a reused run id): replace it once.
  constexpr int max_attempts = 2;
  for (int attempt = 1;; ++attempt) {
    auto uri    = memory::generate_remote_object_uri(prefix);
    auto object = context->open_io_object_for_write(uri, options);
    try {
      write_and_commit(*context, *object, uri, pieces, object_size, window, stream);
    } catch (std::system_error const& error) {
      // `object` is destroyed unflushed on every exit from here: its upload is aborted.
      if (error.code() == std::errc::file_exists && attempt < max_attempts) {
        CUCASCADE_LOG_WARN("remote tier: '{}' already exists; writing the batch under a new name",
                           uri);
        continue;
      }
      throw;
    }
    return std::make_shared<remote_object const>(
      std::move(context), std::move(object), std::move(uri), object_size);
  }
}

void read_remote_object_host(remote_object const& object, std::span<io::slice const> slices)
{
  auto const expected = total_bytes(slices);
  if (expected == 0) { return; }
  // Host destinations need no staging: the bytes land in them directly.
  await_on_unwind request{object.context->host_readv_async_io(*object.object, slices)};
  check_transferred(request.get(), expected, "read", object.uri);
}

void read_remote_object_device(remote_object const& object,
                               std::span<io::slice const> slices,
                               ::cuda::stream_ref stream)
{
  auto& context = *object.context;
  if (!context.supports_device_read()) {
    throw std::logic_error("remote tier: the io context cannot read into device memory");
  }
  if (total_bytes(slices) == 0) { return; }
  std::vector<io::slice> sorted(slices.begin(), slices.end());
  std::sort(sorted.begin(), sorted.end(), [](io::slice const& a, io::slice const& b) {
    return a.offset() < b.offset();
  });

  // Device destinations are staged through pinned memory, one allocation of whole blocks per GET.
  // Without a known staging budget: one request.
  auto* const owned = dynamic_cast<remote_rest_ioctx*>(&context);
  if (owned == nullptr) {
    await_on_unwind request{context.device_readv_async_io(*object.object, sorted, stream)};
    check_transferred(request.get(), total_bytes(sorted), "read", object.uri);
    return;
  }
  // With a context built by make_remote_io_context(): windows whose staging, counted per piece
  // (see remote_rest_ioctx::device_read_staging), fits half the budget -- a wide table of many
  // small buffers costs a block per buffer, not its bytes -- each admitted for exactly that.
  auto const part   = owned->part_bytes(object.size);
  auto const budget = std::max<std::size_t>(owned->staging_bytes() / 2, 1);
  auto const cost   = [&](std::size_t bytes) { return owned->device_read_staging(bytes, part); };
  // Largest piece one window can take; larger slices are cut into pieces of it.
  std::size_t max_piece = std::max<std::size_t>(owned->staging_block(), budget / 2);
  while (max_piece > 1 && cost(max_piece) > budget) {
    max_piece /= 2;
  }

  std::vector<io::slice> window;
  std::size_t window_cost = 0;
  auto flush_window       = [&] {
    if (window.empty()) { return; }
    auto staging = owned->staging_budget().acquire(window_cost);
    // The cache-unaware primitive: spilled objects are read once.
    await_on_unwind request{context.device_readv_async_io(*object.object, window, stream)};
    check_transferred(request.get(), total_bytes(window), "read", object.uri);
    window.clear();
    window_cost = 0;
  };
  for (auto const& s : sorted) {
    for (std::size_t done = 0; done < s.size();) {
      auto const n    = std::min(max_piece, s.size() - done);
      auto const need = cost(n);
      if (window_cost + need > budget) { flush_window(); }
      window.emplace_back(s.offset() + done, n, s.dst + done);
      window_cost += need;
      done += n;
    }
  }
  flush_window();
}

}  // namespace detail
}  // namespace cucascade
