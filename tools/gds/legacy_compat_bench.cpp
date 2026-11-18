/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Intentionally restricted to APIs present at 7641cf5. Do not enable async_engine here.
#include "cache_residency.hpp"

#include <cucascade/io/cufile/cufile_ioctx.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace cucascade::io;
using namespace cucascade::io::cufile;
constexpr std::size_t mib = 1UL << 20;
void check(cudaError_t e)
{
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct resources {
  int fd{-1};
  void* ptr{};
  cudaStream_t stream{};
  ~resources()
  {
    if (stream) static_cast<void>(cudaStreamSynchronize(stream));
    if (ptr) static_cast<void>(cudaFree(ptr));
    if (stream) static_cast<void>(cudaStreamDestroy(stream));
    if (fd >= 0) close(fd);
  }
};
std::size_t number(std::string const& value)
{
  std::size_t n{};
  auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), n);
  if (ec != std::errc{} || end != value.data() + value.size() || !n || n > (1UL << 30))
    throw std::invalid_argument("invalid positive integer");
  return n;
}
int run(int argc, char** argv)
{
  std::string file, csv, version{"new"}, workload{"sustained"}, mode{"legacy"}, arm{"legacy"},
    inode_mode{"shared"};
  std::size_t writer_threads{1};
  std::size_t size_mib{4096}, task_mib{2048}, repetitions{3}, min_seconds{1}, reactors{4};
  std::size_t device{}, iteration{1};
  bool create{}, drop{};
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "legacy_compat_bench --file PATH --csv PATH [--create-file] [--drop-cache]\n"
                   "  --workload sustained|single|spill --size-mib 4096 --task-mib 2048\n"
                   "  --repetitions 3 --min-seconds 1 --reactors 4 --device 0\n"
                   "  --version old|new --iteration N\n"
                   "  --mode legacy|async_auto (new binary only) --arm legacy|async_auto\n"
                   "  --threads 1..16 --inode-mode shared|distinct\n"
                   "CSV appends one fully verified row; no direct-path qualification.\n";
      return 0;
    }
    if (key == "--create-file") {
      create = true;
      continue;
    }
    if (key == "--drop-cache") {
      drop = true;
      continue;
    }
    if (++i == argc) throw std::invalid_argument("missing option value");
    std::string v = argv[i];
    if (key == "--file")
      file = v;
    else if (key == "--csv")
      csv = v;
    else if (key == "--version")
      version = v;
    else if (key == "--mode")
      mode = v;
    else if (key == "--arm")
      arm = v;
    else if (key == "--threads")
      writer_threads = number(v);
    else if (key == "--inode-mode")
      inode_mode = v;
    else if (key == "--workload")
      workload = v;
    else if (key == "--size-mib")
      size_mib = number(v);
    else if (key == "--task-mib")
      task_mib = number(v);
    else if (key == "--repetitions")
      repetitions = number(v);
    else if (key == "--min-seconds")
      min_seconds = number(v);
    else if (key == "--reactors")
      reactors = number(v);
    else if (key == "--iteration")
      iteration = number(v);
    else if (key == "--device") {
      auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), device);
      if (ec != std::errc{} || end != v.data() + v.size())
        throw std::invalid_argument("invalid device ordinal");
    } else
      throw std::invalid_argument("unknown option: " + key);
  }
  if (file.empty() || csv.empty() || (version != "old" && version != "new") ||
      (workload != "sustained" && workload != "single" && workload != "spill") ||
      task_mib > size_mib || size_mib > 1048576 || device > 128 || reactors > 128 ||
      writer_threads > 16 || (mode != "legacy" && mode != "async_auto") ||
      (inode_mode != "shared" && inode_mode != "distinct"))
    throw std::invalid_argument("invalid arguments (see --help)");
  check(cudaSetDevice(static_cast<int>(device)));
  resources r;
  r.fd = open(file.c_str(), O_RDWR | O_CLOEXEC | (create ? O_CREAT | O_EXCL : 0), 0600);
  if (r.fd < 0) throw std::runtime_error("open failed (creation refuses overwrite)");
  auto size       = size_mib * mib;
  auto task_bytes = task_mib * mib;
  std::vector<unsigned char> host(task_bytes);
  for (std::size_t i = 0; i < host.size(); ++i)
    host[i] = static_cast<unsigned char>((i * 17) ^ (i >> 12));
  if (create) {
    for (std::size_t offset = 0; offset < size;) {
      auto n       = std::min(task_bytes, size - offset);
      auto written = pwrite(r.fd, host.data(), n, static_cast<off_t>(offset));
      if (written <= 0) throw std::runtime_error("prepare write failed");
      offset += static_cast<std::size_t>(written);
    }
    if (fsync(r.fd)) throw std::runtime_error("prepare fsync failed");
  }
  if (std::filesystem::file_size(file) < size) throw std::runtime_error("file too small");
  check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
  check(cudaMalloc(&r.ptr, task_bytes));
  check(cudaMemcpyAsync(r.ptr, host.data(), task_bytes, cudaMemcpyHostToDevice, r.stream));
  check(cudaStreamSynchronize(r.stream));
  config cfg{};
#if defined(B10_NEW_ENGINE)
  if (mode == "async_auto") cfg.async_engine.mode = engine_mode::async_auto;
#else
  if (mode != "legacy") throw std::invalid_argument("old binary supports default mode only");
#endif
  auto shared = std::make_shared<cufile_reactor::reactor_context>(cfg);
  cufile_ioctx context(reactors, shared);
  auto object = workload == "spill"
                  ? context.open_io_object_for_write(file, {write_mode::open_existing})
                  : context.open_io_object(file);
  context.start();
  std::vector<slice> reads;
  std::vector<write_slice> writes;
  for (std::size_t offset = 0, index = 0; offset < task_bytes; ++index) {
    auto count =
      workload == "single" ? task_bytes : std::min((1 + index % 8) * mib, task_bytes - offset);
    auto file_offset = workload == "single" ? 0 : (size - offset - count);
    auto* ptr        = static_cast<unsigned char*>(r.ptr) + offset;
    reads.emplace_back(file_offset, count, ptr);
    writes.emplace_back(file_offset, count, ptr);
    offset += count;
  }
  std::vector<std::unique_ptr<resources>> workers;
  std::vector<std::shared_ptr<io_object>> worker_objects;
  for (std::size_t t = 0; t < writer_threads; ++t) {
    auto w = std::make_unique<resources>();
    check(cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking));
    check(cudaMalloc(&w->ptr, task_bytes));
    check(cudaMemcpyAsync(w->ptr, host.data(), task_bytes, cudaMemcpyHostToDevice, w->stream));
    check(cudaStreamSynchronize(w->stream));
    auto path = file;
    if (inode_mode == "distinct") {
      path += ".worker-" + std::to_string(t);
      if (create && !std::filesystem::copy_file(file, path))
        throw std::runtime_error("exclusive worker copy failed");
    }
    worker_objects.push_back(workload == "spill"
                               ? context.open_io_object_for_write(path, {write_mode::open_existing})
                               : context.open_io_object(path));
    workers.push_back(std::move(w));
  }
  auto transfer = [&] {
    std::vector<std::future<void>> futures;
    for (std::size_t t = 0; t < writer_threads; ++t)
      futures.push_back(std::async(std::launch::async, [&, t] {
        check(cudaSetDevice(static_cast<int>(device)));
        auto& w = *workers[t];
        std::vector<slice> rd;
        std::vector<write_slice> wr;
        std::size_t offset{};
        for (auto const& s : reads) {
          auto* ptr = static_cast<unsigned char*>(w.ptr) + offset;
          rd.emplace_back(s.offset(), s.size(), ptr);
          wr.emplace_back(s.offset(), s.size(), ptr);
          offset += s.size();
        }
        auto future =
          workload == "spill"
            ? context.device_writev_async(*worker_objects[t], wr, ::cuda::stream_ref{w.stream})
            : context.device_readv_async_io(*worker_objects[t], rd, ::cuda::stream_ref{w.stream});
        if (std::move(future).get() != task_bytes) throw std::runtime_error("short I/O");
        check(cudaStreamSynchronize(w.stream));
      }));
    for (auto& f : futures)
      f.get();
  };
  auto verify = [&] {
    std::vector<unsigned char> actual(task_bytes), expected(task_bytes);
    for (std::size_t t = 0; t < workers.size(); ++t) {
      check(cudaMemcpy(actual.data(), workers[t]->ptr, task_bytes, cudaMemcpyDeviceToHost));
      auto verify_fd = open(worker_objects[t]->object_path().c_str(), O_RDONLY | O_CLOEXEC);
      if (verify_fd < 0) throw std::runtime_error("verification open failed");
      resources file_owner;
      file_owner.fd = verify_fd;
      std::size_t offset{};
      for (auto const& s : reads) {
        std::size_t done{};
        while (done < s.size()) {
          auto n = pread(verify_fd,
                         expected.data() + offset + done,
                         s.size() - done,
                         static_cast<off_t>(s.offset() + done));
          if (n <= 0) throw std::runtime_error("verification pread failed");
          done += static_cast<std::size_t>(n);
        }
        offset += s.size();
      }
      if (actual != expected) throw std::runtime_error("content mismatch");
    }
  };
  transfer();
  verify();
  double seconds{};
  std::size_t passes{}, max_resident{};
  bool warned_cache{};
  do {
    if (drop) {
      for (auto const& o : worker_objects) {
        resources f;
        f.fd = open(o->object_path().c_str(), O_RDWR | O_CLOEXEC);
        if (fsync(f.fd) || posix_fadvise(f.fd, 0, 0, POSIX_FADV_DONTNEED))
          throw std::runtime_error("advisory cache drop failed");
        max_resident = std::max(max_resident, resident_pages(f.fd));
        if (!warned_cache &&
            max_resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE)) > size / 100) {
          std::cerr << "WARNING: >1% file pages resident after eviction\n";
          warned_cache = true;
        }
      }
    }
    auto start = std::chrono::steady_clock::now();
    transfer();
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++passes;
  } while (passes < repetitions || seconds < static_cast<double>(min_seconds));
  verify();
  context.shutdown();
  auto header = !std::filesystem::exists(csv) || !std::filesystem::file_size(csv);
  std::ofstream output(csv, std::ios::app);
  if (header)
    output << "step,version,workload,iteration,reactors,logical_bytes,seconds,gib_s,drop_cache,"
              "mode,data_mismatches,device,task_bytes,file_bytes,verify,arm,writer_threads,inode_"
              "mode,resident_pages\n";
  auto total = task_bytes * passes * writer_threads;
  output << std::setprecision(9) << "B-10," << version << ',' << workload << ',' << iteration << ','
         << reactors << ',' << total << ',' << seconds << ','
         << static_cast<double>(total) / (1UL << 30) / seconds << ',' << drop << ',' << mode
         << ",0," << device << ',' << task_bytes << ',' << size << ",1," << arm << ','
         << writer_threads << ',' << inode_mode << ',' << max_resident << '\n';
  if (!output) throw std::runtime_error("CSV write failed");
  return 0;
}
}  // namespace
int main(int argc, char** argv)
{
  try {
    return run(argc, argv);
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
