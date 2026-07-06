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

// Standalone S3 parquet read benchmark: the cucascade::io REST datasource
// (libcurl scatter GETs authorized via AWS-SDK presigned URLs) vs kvikIO's
// remote S3 handle wrapped as a cudf datasource.
//
// The AWS SDK is used for (a) listing the parquet objects under the given
// bucket/prefix and (b) presigning each GET/HEAD the REST reactor issues.
// It is a dependency of THIS benchmark only (CUCASCADE_BUILD_S3_BENCHMARK),
// never of the cucascade libraries.
//
// Credentials / region / endpoint come from the standard AWS environment
// (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
// AWS_DEFAULT_REGION, AWS_ENDPOINT_URL) and are honored by both backends.

#include <cucascade/io/datasource.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/s3/s3_request_authorizer.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <kvikio/remote_handle.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/ListObjectsV2Request.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// 4 columns used by the classic TPC-H lineitem aggregations (Q1, Q6, …).
static const std::vector<std::string> COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

enum class DataSource { rest, kvikio };

static DataSource parse_source(std::string_view s)
{
  if (s == "rest") return DataSource::rest;
  if (s == "kvikio") return DataSource::kvikio;
  throw std::invalid_argument(std::string("unknown datasource: ") + std::string(s) +
                              "  (expected: rest | kvikio)");
}

static void usage(char const* prog)
{
  std::cerr << "usage: " << prog << " <bucket> <prefix> <rest|kvikio> <num_rows> [n_reactors]\n"
            << "  bucket     – S3 bucket name (no scheme)\n"
            << "  prefix     – key prefix to list ('-' for the whole bucket); only\n"
            << "               keys ending in .parquet are read\n"
            << "  rest       – cucascade::io REST reactor (AWS-SDK presigned URLs)\n"
            << "  kvikio     – kvikIO RemoteHandle S3 endpoint\n"
            << "  num_rows   – rows to read (0 = all)\n"
            << "  n_reactors – REST reactor threads (default 2; rest only)\n"
            << "\n"
            << "credentials/region/endpoint via the AWS environment: AWS_ACCESS_KEY_ID,\n"
            << "AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL\n";
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

/// cucascade::io::s3::s3_request_authorizer backed by the AWS SDK presigner.
/// Presigned URLs carry the auth in the query string, so no headers are
/// returned; the REST reactor appends its Range header without invalidating
/// the signature.
class awssdk_presigned_authorizer final : public cucascade::io::s3::s3_request_authorizer {
 public:
  explicit awssdk_presigned_authorizer(std::shared_ptr<Aws::S3::S3Client> client)
    : _client(std::move(client))
  {
  }

  [[nodiscard]] cucascade::io::s3::s3_authorized_request authorize(
    cucascade::io::s3::s3_object_ref const& obj,
    cucascade::io::s3::s3_request_method method,
    std::chrono::seconds timeout) override
  {
    auto const http_method = method == cucascade::io::s3::s3_request_method::GET
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
    : _handle(kvikio::RemoteHandle::open(s3_url))
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
    // RemoteHandle::read accepts device memory (staged through kvikIO's
    // internal bounce buffer).
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
  [[nodiscard]] size_t clamp(size_t offset, size_t size) const
  {
    size_t const fsize = _handle.nbytes();
    return offset < fsize ? std::min(size, fsize - offset) : 0;
  }

  kvikio::RemoteHandle _handle;
};

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
  if (argc != 5 && argc != 6) {
    usage(argv[0]);
    return 1;
  }

  std::string const bucket = argv[1];
  std::string prefix       = argv[2];
  if (prefix == "-") { prefix.clear(); }

  DataSource source;
  try {
    source = parse_source(argv[3]);
  } catch (std::invalid_argument const& e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  long long num_rows_arg = std::stoll(argv[4]);
  if (num_rows_arg < 0) {
    std::cerr << "num_rows must be >= 0\n";
    return 1;
  }
  size_t num_rows = static_cast<size_t>(num_rows_arg);  // 0 means all

  size_t n_reactors = 2;
  if (argc == 6) {
    long long n_reactors_arg = std::stoll(argv[5]);
    if (n_reactors_arg <= 0) {
      std::cerr << "n_reactors must be > 0\n";
      return 1;
    }
    n_reactors = static_cast<size_t>(n_reactors_arg);
  }

  aws_api api;
  auto s3_client = make_s3_client();

  auto keys = list_parquet_objects(*s3_client, bucket, prefix);
  if (keys.empty()) {
    std::cerr << "no .parquet objects under s3://" << bucket << "/" << prefix << "\n";
    return 1;
  }

  std::cout << "Source : " << argv[3] << "\n"
            << "Bucket : " << bucket << "\n"
            << "Objects: " << keys.size() << "\n";
  for (auto const& k : keys)
    std::cout << "  s3://" << bucket << "/" << k << "\n";
  std::cout << "Rows   : " << (num_rows == 0 ? "all" : std::to_string(num_rows)) << "\n"
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

  // Build the per-backend datasources up front (both do one HEAD per object
  // at open) — connection setup and metadata are outside the timed region.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.reserve(keys.size());

  // Host staging pool for the REST reactor's bounce slots (rest path only,
  // but the pool is cheap to keep unconditional and simplifies lifetimes).
  constexpr uint32_t POOL_MAX_SLABS = 20;
  constexpr size_t CHUNKS_PER_SLAB =
    cucascade::memory::fixed_size_host_memory_resource::default_pool_size;
  constexpr size_t POOL_CAPACITY =
    static_cast<size_t>(POOL_MAX_SLABS) * CHUNKS_PER_SLAB * (1 << 20);

  cucascade::memory::numa_region_pinned_host_memory_resource upstream(0, /*make_portable=*/true);
  cucascade::memory::fixed_size_host_memory_resource host_mr(0,                // device_id
                                                             upstream,         // upstream allocator
                                                             POOL_CAPACITY,    // mem_limit
                                                             POOL_CAPACITY,    // capacity
                                                             1 << 20,          // block_size = 1 MiB
                                                             CHUNKS_PER_SLAB,  // pool_size
                                                             1);               // initial_pools

  std::shared_ptr<cucascade::io::rest::rest_ioctx> io_ctx;  // rest path only
  if (source == DataSource::rest) {
    cucascade::io::rest::config rest_cfg;
    rest_cfg.bounce_block_size = host_mr.get_block_size();

    auto authorizer = std::make_shared<awssdk_presigned_authorizer>(s3_client);
    auto rest_ctx   = std::make_shared<cucascade::io::rest::rest_reactor::reactor_context>(
      std::move(rest_cfg), std::move(authorizer), &host_mr);
    io_ctx = std::make_shared<cucascade::io::rest::rest_ioctx>(n_reactors, std::move(rest_ctx));
    io_ctx->start();

    for (auto const& key : keys) {
      sources.push_back(io_ctx->open_datasource("s3://" + bucket + "/" + key));
    }
  } else {
    for (auto const& key : keys) {
      sources.push_back(std::make_unique<kvikio_s3_datasource>("s3://" + bucket + "/" + key));
    }
  }

  auto scan_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();

  // Per-file: probe the footer through the backend's own datasource (host
  // reads; untimed), then run hybrid_scan to collect the byte ranges the
  // selected columns touch.  The num_rows budget is consumed in file order.
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  metadatas.reserve(keys.size());
  size_t total_range_bytes = 0;
  size_t total_row_groups  = 0;
  size_t total_ranges      = 0;
  int64_t accumulated_rows = 0;

  for (auto const& src : sources) {
    std::vector<uint8_t> footer_buf;
    {
      auto& ds       = *src;
      auto file_size = ds.size();
      cudf::io::parquet::file_ender_s ender{};
      ds.host_read(file_size - sizeof(ender), sizeof(ender), reinterpret_cast<uint8_t*>(&ender));
      footer_buf.resize(ender.footer_len);
      ds.host_read(
        file_size - sizeof(ender) - ender.footer_len, ender.footer_len, footer_buf.data());
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

    for (auto const& br : coalesced)
      total_range_bytes += static_cast<size_t>(br.size());
    total_row_groups += selected_row_groups.size();
    total_ranges += coalesced.size();

    metadatas.push_back(std::move(file_metadata));
  }

  std::cout << "Hybrid scan: " << total_row_groups << " row group(s), " << total_ranges
            << " byte range(s), " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_range_bytes) / (1024.0 * 1024.0) << " MiB total\n\n";

  auto read_opts_builder = cudf::io::parquet_reader_options::builder().column_names(COLUMNS);
  if (num_rows > 0) read_opts_builder.num_rows(static_cast<int64_t>(num_rows));
  auto read_opts = read_opts_builder.build();

  // cudaMemcpyBatchAsync rejects the legacy / per-thread default streams with
  // cudaMemLocation hints — pass an explicit stream so all H2D copies inside
  // the datasource and cudf share the same one.
  rmm::cuda_stream stream;

  // Timed run.
  double ms = time_ms([&] {
    auto tbl =
      cudf::io::read_parquet(std::move(sources), std::move(metadatas), read_opts, stream.view());
  });

  std::cout << std::fixed << std::setprecision(1) << ms << " ms\n";
  return 0;
}
