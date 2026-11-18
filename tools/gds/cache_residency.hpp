// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

// Diagnostic only: mincore is a point-in-time observation, not proof of direct I/O.
inline std::size_t resident_pages(int fd)
{
  struct stat st{};
  if (fstat(fd, &st)) throw std::runtime_error("residency fstat failed");
  if (!st.st_size) return 0;
  auto size = static_cast<std::size_t>(st.st_size);
  auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  void* map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) throw std::runtime_error("residency mmap failed");
  std::vector<unsigned char> pages((size + page - 1) / page);
  auto rc = mincore(map, size, pages.data());
  munmap(map, size);
  if (rc) throw std::runtime_error("mincore failed");
  std::size_t count{};
  for (auto p : pages)
    count += p & 1U;
  return count;
}
