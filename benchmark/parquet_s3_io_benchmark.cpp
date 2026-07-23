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

// Standalone S3 parquet range-read I/O benchmark.  The S3 analogue of
// parquet_io_benchmark: it uses hybrid_scan to find the column-chunk byte
// ranges a column projection touches, then times only the reads of those
// ranges into a destination tier (host or device) — no parquet decode.
//
// Two read paths are compared:
//   rest   – the native cucascade::io REST reactor (libcurl scatter GETs
//            authorized via AWS-SDK presigned URLs).  host reads go through the
//            vector-I/O primitive (host_read_ranges_async_io); device reads
//            through device_read_async (reactor-staged: the reactor streams
//            each range network->pinned-bounce->device on its own slots).
//   kvikio – kvikIO's RemoteHandle S3 endpoint wrapped as a cudf datasource;
//            host_read_async / device_read_async issued per range.
// All async operations are collected first and synchronized at the end.
//
// Credentials / region / endpoint come from the standard AWS environment
// (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
// AWS_DEFAULT_REGION, AWS_ENDPOINT_URL) and are honored by both backends.

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>

#include <kvikio/remote_handle.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/ListObjectsV2Request.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// 4 columns used by the classic TPC-H lineitem aggregations (Q1, Q6, …).
static const std::vector<std::string> COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

enum class Backend { rest, kvikio };
enum class Dest { host, device };

static Backend parse_backend(std::string_view s)
{
  if (s == "rest") return Backend::rest;
  if (s == "kvikio") return Backend::kvikio;
  throw std::invalid_argument(std::string("unknown backend: ") + std::string(s) +
                              "  (expected: rest | kvikio)");
}

static Dest parse_dest(std::string_view s)
{
  if (s == "host") return Dest::host;
  if (s == "device") return Dest::device;
  throw std::invalid_argument(std::string("unknown dest: ") + std::string(s) +
                              "  (expected: host | device)");
}

static void usage(char const* prog)
{
  std::cerr
    << "usage: " << prog
    << " <bucket> <prefix> <rest|kvikio> <host|device> <num_rows> [n_threads] [key=value ...]\n"
    << "  bucket     – S3 bucket name (no scheme)\n"
    << "  prefix     – key prefix to list ('-' for the whole bucket); only\n"
    << "               keys ending in .parquet are read\n"
    << "  rest       – native REST reactor (host=vector-io, device=host->device)\n"
    << "  kvikio     – kvikIO RemoteHandle per-range host_read/device_read_async\n"
    << "  host|device– destination tier the ranges are read into\n"
    << "  num_rows   – rows to read (0 = all)\n"
    << "  n_threads  – reader threads / REST reactors (default 2)\n"
    << "\n"
    << "REST config overrides (rest only; key=value, any order after num_rows):\n"
    << "  n_threads=<N>           reader threads / REST reactors\n"
    << "  max_connections=<N>     max concurrent in-flight easy handles per reactor (def 16)\n"
    << "  chunk_size=<bytes>      target max bytes per ranged GET (def 8388608)\n"
    << "  max_n_chunks=<N>        max buffers fused into one scatter GET (def 16)\n"
    << "  max_read_split=<N>      parallel GETs a contiguous host read splits into (def 16)\n"
    << "\n"
    << "credentials/region/endpoint via the AWS environment: AWS_ACCESS_KEY_ID,\n"
    << "AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL\n";
}

// Apply a "key=value" REST config override.  n_threads lives outside the
// rest::config, so it is threaded through separately.  Returns false for an
// unknown key so the caller can report it and exit.
static bool apply_override(std::string const& key,
                           std::string const& val,
                           cucascade::io::rest::config& cfg,
                           size_t& n_threads)
{
  auto ull = [&val] { return static_cast<std::size_t>(std::stoull(val)); };
  if (key == "n_threads") {
    n_threads = ull();
  } else if (key == "max_connections") {
    cfg.max_connections = ull();
  } else if (key == "chunk_size") {
    cfg.chunk_size = ull();
  } else if (key == "max_n_chunks") {
    cfg.max_n_chunks = ull();
  } else if (key == "max_read_split") {
    cfg.max_read_split = ull();
  } else {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// AWS SDK helpers
// ---------------------------------------------------------------------------

/// RAII around Aws::InitAPI / ShutdownAPI.
struct aws_api {
  Aws::SDKOptions options;
  aws_api() { Aws::InitAPI(options); }
  ~aws_api() { Aws::ShutdownAPI(options); }
  aws_api(aws_api const&)            = delete;
  aws_api& operator=(aws_api const&) = delete;
};

/// Client configuration from the AWS environment.  AWS_ENDPOINT_URL (e.g. a
/// MinIO endpoint) switches the client to endpoint-override + path-style
/// addressing, matching what kvikIO does with the same variable.
static std::shared_ptr<Aws::S3::S3Client> make_s3_client()
{
  Aws::Client::ClientConfiguration cfg;
  if (char const* region = std::getenv("AWS_DEFAULT_REGION"); region != nullptr) {
    cfg.region = region;
  }
  bool use_virtual_addressing = true;
  if (char const* ep = std::getenv("AWS_ENDPOINT_URL"); ep != nullptr) {
    std::string endpoint{ep};
    if (endpoint.rfind("http://", 0) == 0) {
      cfg.scheme = Aws::Http::Scheme::HTTP;
      endpoint   = endpoint.substr(std::strlen("http://"));
    } else if (endpoint.rfind("https://", 0) == 0) {
      cfg.scheme = Aws::Http::Scheme::HTTPS;
      endpoint   = endpoint.substr(std::strlen("https://"));
    }
    cfg.endpointOverride   = endpoint;
    use_virtual_addressing = false;  // custom endpoints (MinIO, …) are path-style
  }
  return std::make_shared<Aws::S3::S3Client>(
    cfg, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, use_virtual_addressing);
}

/// List every key under @p prefix ending in ".parquet", sorted.
static std::vector<std::string> list_parquet_objects(Aws::S3::S3Client& client,
                                                     std::string const& bucket,
                                                     std::string const& prefix)
{
  std::vector<std::string> keys;
  Aws::S3::Model::ListObjectsV2Request req;
  req.SetBucket(bucket.c_str());
  if (!prefix.empty()) { req.SetPrefix(prefix.c_str()); }

  bool more = true;
  while (more) {
    auto outcome = client.ListObjectsV2(req);
    if (!outcome.IsSuccess()) {
      throw std::runtime_error("ListObjectsV2 failed for bucket '" + bucket +
                               "': " + outcome.GetError().GetMessage());
    }
    auto const& result = outcome.GetResult();
    for (auto const& obj : result.GetContents()) {
      std::string key = obj.GetKey();
      if (key.size() >= 8 && key.compare(key.size() - 8, 8, ".parquet") == 0) {
        keys.push_back(std::move(key));
      }
    }
    more = result.GetIsTruncated();
    if (more) { req.SetContinuationToken(result.GetNextContinuationToken()); }
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

/// cucascade::io::rest::request_authorizer backed by the AWS SDK presigner.
/// Presigned URLs carry the auth in the query string, so no headers are
/// returned; the REST reactor appends its Range header without invalidating
/// the signature.
class awssdk_presigned_authorizer final : public cucascade::io::rest::request_authorizer {
 public:
  explicit awssdk_presigned_authorizer(std::shared_ptr<Aws::S3::S3Client> client)
    : _client(std::move(client))
  {
  }

  [[nodiscard]] cucascade::io::rest::authorized_request authorize(
    cucascade::io::rest::object_ref const& obj,
    cucascade::io::rest::request_method method,
    std::chrono::seconds timeout) override
  {
    auto const http_method = method == cucascade::io::rest::request_method::GET
                               ? Aws::Http::HttpMethod::HTTP_GET
                               : Aws::Http::HttpMethod::HTTP_HEAD;
    long long const ttl    = timeout.count() > 0 ? timeout.count() : 60;
    auto url               = _client->GeneratePresignedUrl(
      obj.bucket.c_str(), obj.key.c_str(), http_method, static_cast<uint64_t>(ttl));
    return {std::string(url.c_str(), url.size()), {}};
  }

 private:
  std::shared_ptr<Aws::S3::S3Client> _client;
};

// ---------------------------------------------------------------------------
// kvikIO adapter: cudf::io::datasource over kvikio::RemoteHandle
// ---------------------------------------------------------------------------

class kvikio_s3_datasource final : public cudf::io::datasource {
 public:
  explicit kvikio_s3_datasource(std::string const& s3_url)
    : _handle(kvikio::RemoteHandle::open(to_https_url(s3_url), kvikio::RemoteEndpointType::S3))
  {
  }

  [[nodiscard]] size_t size() const override { return _handle.nbytes(); }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    size_t const n = clamp(offset, size);
    if (n == 0) { return 0; }
    return _handle.read(dst, n, offset);
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    size_t const n = clamp(offset, size);
    std::vector<uint8_t> data(n);
    if (n != 0) { _handle.read(data.data(), n, offset); }
    return buffer::create(std::move(data));
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  size_t device_read(size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view) override
  {
    size_t const n = clamp(offset, size);
    if (n == 0) { return 0; }
    return _handle.read(dst, n, offset);
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view) override
  {
    size_t const n = clamp(offset, size);
    return _handle.pread(dst, n, offset);
  }

  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    size_t const n = clamp(offset, size);
    rmm::device_buffer out(n, stream);
    if (n != 0) { _handle.read(out.data(), n, offset); }
    return buffer::create(std::move(out));
  }

 private:
  // kvikIO's RemoteHandle::open rejects the "s3://" scheme; convert it to the
  // virtual-hosted "https://<bucket>.s3.<region>.amazonaws.com/<object>" form
  // (region/endpoint from AWS_DEFAULT_REGION / AWS_ENDPOINT_URL) that the S3
  // endpoint expects.
  [[nodiscard]] static std::string to_https_url(std::string const& s3_url)
  {
    auto [bucket, object] = kvikio::S3Endpoint::parse_s3_url(s3_url);
    return kvikio::S3Endpoint::url_from_bucket_and_object(
      std::move(bucket), std::move(object), std::nullopt, std::nullopt);
  }

  [[nodiscard]] size_t clamp(size_t offset, size_t size) const
  {
    size_t const fsize = _handle.nbytes();
    return offset < fsize ? std::min(size, fsize - offset) : 0;
  }

  kvikio::RemoteHandle _handle;
};

// ---------------------------------------------------------------------------

// One coalesced column-chunk byte range in a specific object.
struct coalesced_range {
  size_t obj_idx;
  size_t offset;
  size_t size;
};

int main(int argc, char** argv)
{
  if (argc < 6) {
    usage(argv[0]);
    return 1;
  }

  std::string const bucket = argv[1];
  std::string prefix       = argv[2];
  if (prefix == "-") { prefix.clear(); }

  Backend backend;
  Dest dest;
  try {
    backend = parse_backend(argv[3]);
    dest    = parse_dest(argv[4]);
  } catch (std::invalid_argument const& e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  long long num_rows_arg = std::stoll(argv[5]);
  if (num_rows_arg < 0) {
    std::cerr << "num_rows must be >= 0\n";
    return 1;
  }
  size_t num_rows = static_cast<size_t>(num_rows_arg);  // 0 means all

  // Everything after num_rows is either a bare positional n_threads (kept for
  // backward compatibility) or a "key=value" REST config override, any order.
  size_t n_threads = 2;
  cucascade::io::rest::config rest_cfg;
  try {
    for (int i = 6; i < argc; ++i) {
      std::string const tok = argv[i];
      auto const eq         = tok.find('=');
      if (eq == std::string::npos) {
        long long const v = std::stoll(tok);
        if (v <= 0) {
          std::cerr << "n_threads must be > 0\n";
          return 1;
        }
        n_threads = static_cast<size_t>(v);
      } else if (!apply_override(tok.substr(0, eq), tok.substr(eq + 1), rest_cfg, n_threads)) {
        std::cerr << "unknown config key: " << tok.substr(0, eq) << "\n";
        usage(argv[0]);
        return 1;
      }
    }
  } catch (std::exception const& e) {
    std::cerr << "bad argument value: " << e.what() << "\n";
    return 1;
  }

  aws_api api;
  auto s3_client = make_s3_client();

  auto keys = list_parquet_objects(*s3_client, bucket, prefix);
  if (keys.empty()) {
    std::cerr << "no .parquet objects under s3://" << bucket << "/" << prefix << "\n";
    return 1;
  }

  std::cout << "Backend: " << argv[3] << "\n"
            << "Dest   : " << argv[4] << "\n"
            << "Bucket : " << bucket << "\n"
            << "Objects: " << keys.size() << "\n";
  for (auto const& k : keys)
    std::cout << "  s3://" << bucket << "/" << k << "\n";
  std::cout << "Rows   : " << (num_rows == 0 ? "all" : std::to_string(num_rows)) << "\n"
            << "Threads: " << n_threads << "\n"
            << "Columns: ";
  for (auto const& c : COLUMNS)
    std::cout << c << "  ";
  std::cout << "\n\n";

  cudaFree(nullptr);

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(::cuda::mr::any_resource<::cuda::mr::device_accessible>{
    rmm::device_async_resource_ref{async_mr}});

  auto time_ms = [](auto fn) -> double {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
      .count();
  };

  // -- Backend / staging setup (untimed) --------------------------------------

  // Host staging pool for the REST reactor's bounce slots (rest path only, but
  // cheap to keep unconditional).  device_read_async draws one 1 MiB bounce
  // block per in-flight connection, so the pool must cover the worst case of
  // n_threads (reactors) * max_connections concurrent GETs, else it OOMs.
  constexpr size_t CHUNKS_PER_SLAB =
    cucascade::memory::fixed_size_host_memory_resource::default_pool_size;
  size_t const default_blocks = 20 * CHUNKS_PER_SLAB;
  size_t const needed_blocks  = n_threads * rest_cfg.max_connections + CHUNKS_PER_SLAB;
  size_t const pool_blocks    = std::max(default_blocks, needed_blocks);
  size_t const n_slabs        = (pool_blocks + CHUNKS_PER_SLAB - 1) / CHUNKS_PER_SLAB;
  size_t const POOL_CAPACITY  = n_slabs * CHUNKS_PER_SLAB * (1 << 20);

  cucascade::memory::numa_region_pinned_host_memory_resource upstream(0, /*make_portable=*/true);
  cucascade::memory::fixed_size_host_memory_resource host_mr(0,                // device_id
                                                             upstream,         // upstream allocator
                                                             POOL_CAPACITY,    // mem_limit
                                                             POOL_CAPACITY,    // capacity
                                                             1 << 20,          // block_size = 1 MiB
                                                             CHUNKS_PER_SLAB,  // pool_size
                                                             1);               // initial_pools

  // rest path: one io_object per object on the REST ioctx.
  std::shared_ptr<cucascade::io::ioctx> io_ctx;
  std::vector<std::shared_ptr<cucascade::io::io_object>> io_objects;
  // kvikio path: one datasource per object.
  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;

  if (backend == Backend::rest) {
    rest_cfg.bounce_block_size = host_mr.get_block_size();
    std::cout << "REST   : n_reactors=" << n_threads
              << "  max_connections=" << rest_cfg.max_connections
              << "  chunk_size=" << rest_cfg.chunk_size
              << "  max_n_chunks=" << rest_cfg.max_n_chunks
              << "  max_read_split=" << rest_cfg.max_read_split << "\n";
    auto authorizer = std::make_shared<awssdk_presigned_authorizer>(s3_client);
    auto rest_ctx   = std::make_shared<cucascade::io::rest::rest_reactor::reactor_context>(
      std::move(rest_cfg), std::move(authorizer), &host_mr);
    io_ctx = std::make_shared<cucascade::io::rest::rest_ioctx>(n_threads, std::move(rest_ctx));
    io_ctx->start();
    io_objects.reserve(keys.size());
    for (auto const& key : keys)
      io_objects.push_back(io_ctx->open_io_object("s3://" + bucket + "/" + key));
  } else {
    datasources.reserve(keys.size());
    for (auto const& key : keys)
      datasources.push_back(std::make_unique<kvikio_s3_datasource>("s3://" + bucket + "/" + key));
  }

  auto object_size = [&](size_t obj_idx) -> size_t {
    return backend == Backend::rest ? io_objects[obj_idx]->size() : datasources[obj_idx]->size();
  };
  auto host_read_footer = [&](size_t obj_idx, size_t offset, size_t size, uint8_t* dst) {
    if (backend == Backend::rest) {
      io_ctx->host_read(*io_objects[obj_idx], offset, size, dst);
    } else {
      datasources[obj_idx]->host_read(offset, size, dst);
    }
  };

  auto scan_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();

  // Per-object (untimed): probe the footer through the backend's own reads, then
  // run hybrid_scan to collect the byte ranges the selected columns touch.  The
  // num_rows budget is consumed in object order.
  std::vector<coalesced_range> ranges;
  size_t total_range_bytes = 0;
  int64_t accumulated_rows = 0;

  for (size_t obj_idx = 0; obj_idx < keys.size(); ++obj_idx) {
    std::vector<uint8_t> footer_buf;
    {
      auto file_size = object_size(obj_idx);
      cudf::io::parquet::file_ender_s ender{};
      host_read_footer(
        obj_idx, file_size - sizeof(ender), sizeof(ender), reinterpret_cast<uint8_t*>(&ender));
      footer_buf.resize(ender.footer_len);
      host_read_footer(
        obj_idx, file_size - sizeof(ender) - ender.footer_len, ender.footer_len, footer_buf.data());
    }

    cudf::io::parquet::experimental::hybrid_scan_reader scanner{
      cudf::host_span<uint8_t const>{footer_buf.data(), footer_buf.size()}, scan_opts};
    auto file_metadata = scanner.parquet_metadata();

    std::vector<cudf::size_type> selected_row_groups;
    if (num_rows == 0) {
      selected_row_groups = scanner.all_row_groups(scan_opts);
    } else if (accumulated_rows < static_cast<int64_t>(num_rows)) {
      for (cudf::size_type i = 0;
           i < static_cast<cudf::size_type>(file_metadata.row_groups.size()) &&
           accumulated_rows < static_cast<int64_t>(num_rows);
           ++i) {
        selected_row_groups.push_back(i);
        accumulated_rows += file_metadata.row_groups[static_cast<size_t>(i)].num_rows;
      }
    }

    auto byte_ranges = scanner.all_column_chunks_byte_ranges(selected_row_groups, scan_opts);
    std::sort(byte_ranges.begin(), byte_ranges.end(), [](auto const& a, auto const& b) {
      return a.offset() < b.offset();
    });
    std::vector<cudf::io::text::byte_range_info> coalesced;
    coalesced.reserve(byte_ranges.size());
    for (auto const& br : byte_ranges) {
      if (br.size() <= 0) continue;
      if (!coalesced.empty()) {
        auto& back       = coalesced.back();
        int64_t back_end = back.offset() + back.size();
        if (br.offset() <= back_end) {
          int64_t br_end = br.offset() + br.size();
          if (br_end > back_end)
            back = cudf::io::text::byte_range_info{back.offset(), br_end - back.offset()};
          continue;
        }
      }
      coalesced.push_back(br);
    }

    for (auto const& br : coalesced) {
      ranges.push_back(
        coalesced_range{obj_idx, static_cast<size_t>(br.offset()), static_cast<size_t>(br.size())});
      total_range_bytes += static_cast<size_t>(br.size());
    }
  }

  if (ranges.empty()) {
    std::cerr << "hybrid scan produced no byte ranges\n";
    return 1;
  }

  std::cout << "Hybrid scan: " << ranges.size() << " byte range(s), " << std::fixed
            << std::setprecision(2) << static_cast<double>(total_range_bytes) / (1024.0 * 1024.0)
            << " MiB total\n\n";

  // Per-range destination buffers (untimed).  host → pinned; device → rmm.
  // Both device paths (REST reactor-staged device_read_async and kvikIO
  // device_read_async) manage their own host bounce slots internally, so no
  // caller-supplied staging buffer is needed.
  rmm::cuda_stream alloc_stream;
  std::vector<uint8_t*> dsts(ranges.size(), nullptr);
  std::vector<void*> host_bufs;              // owned pinned allocations (host dest only)
  std::vector<rmm::device_buffer> dev_bufs;  // owned device allocations
  bool const need_stage = false;

  if (dest == Dest::host) {
    host_bufs.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
      void* p = upstream.allocate_sync(ranges[i].size);
      host_bufs.push_back(p);
      dsts[i] = static_cast<uint8_t*>(p);
    }
  } else {
    dev_bufs.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
      dev_bufs.emplace_back(ranges[i].size, alloc_stream.view());
      dsts[i] = static_cast<uint8_t*>(dev_bufs[i].data());
    }
    alloc_stream.synchronize();
    if (need_stage) {
      host_bufs.reserve(ranges.size());
      for (size_t i = 0; i < ranges.size(); ++i)
        host_bufs.push_back(upstream.allocate_sync(ranges[i].size));
    }
  }

  // Settle delay so the reactors finish spinning up before the timed run.
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));

  size_t const n_threads_eff = std::min(n_threads, ranges.size());
  std::cout << "Read   : " << n_threads_eff << " worker thread(s)\n\n";

  // -- Timed run --------------------------------------------------------------
  //
  // Partition the coalesced ranges into contiguous [lo, hi) slices, one per
  // worker.  Each worker issues its reads (collecting futures) then blocks on
  // them at the end, so the region measures overlapped in-flight I/O.

  double ms = time_ms([&] {
    std::vector<std::thread> workers;
    workers.reserve(n_threads_eff);
    size_t const base = ranges.size() / n_threads_eff;
    size_t const rem  = ranges.size() % n_threads_eff;

    size_t lo = 0;
    for (size_t t = 0; t < n_threads_eff; ++t) {
      size_t const count = base + (t < rem ? 1 : 0);
      size_t const hi    = lo + count;
      workers.emplace_back([&, lo, hi] {
        rmm::cuda_stream stream;  // per-worker stream for device reads

        if (backend == Backend::rest && dest == Dest::host) {
          // Vector I/O: one host_read_ranges_async_io per object, over that
          // object's segments.  Segment vectors must outlive the futures.
          std::vector<std::vector<cucascade::io::io_object_segment>> seg_sets;
          std::vector<cucascade::exec::semi_future<size_t>> futs;
          size_t cur_obj = SIZE_MAX;
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            if (r.obj_idx != cur_obj) {
              seg_sets.emplace_back();
              cur_obj = r.obj_idx;
            }
            seg_sets.back().push_back(cucascade::io::io_object_segment{r.offset, r.size, dsts[i]});
          }
          size_t set = 0;
          cur_obj    = SIZE_MAX;
          for (size_t i = lo; i < hi; ++i) {
            if (ranges[i].obj_idx != cur_obj) {
              cur_obj   = ranges[i].obj_idx;
              auto& seg = seg_sets[set++];
              futs.push_back(io_ctx->host_read_ranges_async_io(
                *io_objects[cur_obj], std::span<cucascade::io::io_object_segment>(seg)));
            }
          }
          for (auto& f : futs)
            std::move(f).get();
        } else if (backend == Backend::rest && dest == Dest::device) {
          // Reactor-staged device read: one device_read_async per range (the
          // reactor manages its own pinned bounce slots).  Mirrors kvikIO's
          // per-range device_read_async for an apples-to-apples comparison.
          std::vector<cucascade::exec::semi_future<size_t>> futs;
          futs.reserve(hi - lo);
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            futs.push_back(io_ctx->device_read_async(
              *io_objects[r.obj_idx], r.offset, r.size, dsts[i], stream.view()));
          }
          for (auto& f : futs)
            std::move(f).get();
          stream.synchronize();
        } else if (backend == Backend::kvikio && dest == Dest::host) {
          std::vector<std::future<size_t>> futs;
          futs.reserve(hi - lo);
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            futs.push_back(datasources[r.obj_idx]->host_read_async(r.offset, r.size, dsts[i]));
          }
          for (auto& f : futs)
            f.get();
        } else {  // kvikio + device
          std::vector<std::future<size_t>> futs;
          futs.reserve(hi - lo);
          for (size_t i = lo; i < hi; ++i) {
            auto const& r = ranges[i];
            futs.push_back(
              datasources[r.obj_idx]->device_read_async(r.offset, r.size, dsts[i], stream.view()));
          }
          for (auto& f : futs)
            f.get();
          stream.synchronize();
        }
      });
      lo = hi;
    }
    for (auto& w : workers)
      w.join();
  });

  double const gib   = static_cast<double>(total_range_bytes) / (1024.0 * 1024.0 * 1024.0);
  double const gib_s = gib / (ms / 1000.0);
  std::cout << std::fixed << std::setprecision(1) << ms << " ms  " << std::setprecision(2) << gib_s
            << " GiB/s\n";

  for (size_t i = 0; i < host_bufs.size(); ++i)
    upstream.deallocate_sync(host_bufs[i], ranges[i].size);

  if (io_ctx) { io_ctx->shutdown(); }
  return 0;
}
