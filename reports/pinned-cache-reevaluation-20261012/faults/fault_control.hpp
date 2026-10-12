#pragma once

#include <cstddef>

enum class fault_operation : unsigned {
  host_allocate,
  event_create,
  event_record,
  event_wait,
  stream_synchronize,
  event_synchronize,
  host_free,
  event_destroy,
  count
};

extern "C" {
void pinned_fault_reset() noexcept;
void pinned_fault_arm(unsigned operation, int error, unsigned failures) noexcept;
void pinned_fault_watch(void* pointer) noexcept;
std::size_t pinned_fault_calls(unsigned operation) noexcept;
std::size_t pinned_fault_hits(unsigned operation) noexcept;
std::size_t pinned_fault_watched_frees() noexcept;
unsigned pinned_fault_last_host_flags() noexcept;
}

constexpr unsigned operation_id(fault_operation operation) noexcept
{
  return static_cast<unsigned>(operation);
}
