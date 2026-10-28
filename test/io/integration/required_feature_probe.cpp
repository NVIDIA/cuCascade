// SPDX-License-Identifier: Apache-2.0
// Test-only public-API driver. No redirect implementation lives in this probe.
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string json(std::string const& value)
{
  std::string out{"\""};
  char const* hex = "0123456789abcdef";
  for (auto c : value) {
    auto const b = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (b < 32) {
      out += "\\u00";
      out += hex[b >> 4];
      out += hex[b & 15];
    } else
      out += c;
  }
  return out + '"';
}
void require(bool condition, char const* message)
{
  if (!condition) throw std::runtime_error(message);
}
std::string env(char const* name)
{
  auto p = std::getenv(name);
  require(p != nullptr && *p != '\0', "missing fixture credential");
  return p;
}
}  // namespace

int main(int argc, char** argv)
{
  char const* stage = "setup";
  try {
    require(argc == 8, "usage: probe ENDPOINT URI FILE ETAG MODE COUNT CA_FILE_OR_DASH");
    std::string mode{argv[5]};
    auto count = static_cast<std::size_t>(std::stoul(argv[6]));
    require(count > 0 && count <= 256, "count outside 1..256");
    require(
      mode == "serial" || mode == "queue" || mode == "footer" || mode == "head" || mode == "list",
      "unknown mode");
    cucascade::io::rest::config cfg{};
    cfg.request_timeout_s       = 5;
    cfg.stall_time_s            = 5;
    cfg.max_retry_attempts      = 0;
    cfg.max_auth_retry_attempts = 0;
    cfg.max_connections         = mode == "queue" ? 4 : 1;
    cfg.footer_probe_bytes      = 512;
    if (std::string(argv[7]) != "-") cfg.ca_bundle_path = argv[7];
    cucascade::io::rest::s3::static_credentials creds{};
    creds.access_key_id     = env("CUCASCADE_TEST_ACCESS_KEY");
    creds.secret_access_key = env("CUCASCADE_TEST_SECRET_KEY");
    auto auth               = std::make_shared<cucascade::io::rest::s3::sigv4_header_authorizer>(
      std::move(creds), "us-east-1", argv[1]);
    auto reactor =
      std::make_shared<cucascade::io::rest::rest_reactor::reactor_context>(cfg, auth, nullptr);
    auto ctx = std::make_shared<cucascade::io::rest::rest_ioctx>(1, reactor);
    if (mode == "list") {
      stage             = "list";
      std::size_t pages = 0;
      ctx->list_objects_paged(argv[2], "", 2, [&](auto const& page) {
        ++pages;
        std::cout << "{\"event\":\"page\",\"index\":" << pages
                  << ",\"truncated\":" << (page.is_truncated ? "true" : "false")
                  << ",\"next\":" << json(page.next_continuation_token) << ",\"entries\":[";
        bool first = true;
        for (auto const& entry : page.entries) {
          if (!first) std::cout << ',';
          first = false;
          std::cout << "{\"key\":" << json(entry.key) << ",\"size\":" << entry.size << '}';
        }
        std::cout << "]}" << std::endl;
        return true;
      });
      std::cout << "{\"event\":\"done\",\"status\":\"PASS\",\"pages\":" << pages
                << ",\"api_exposes_etag\":false}" << std::endl;
      return 0;
    }
    std::ifstream input(argv[3], std::ios::binary);
    require(input.good(), "cannot open fixture bytes");
    std::vector<std::uint8_t> expected{std::istreambuf_iterator<char>(input), {}};
    require(expected.size() >= count * 37 + 31, "fixture too small");
    stage = "open";
    auto object =
      ctx->open_io_object(argv[2],
                          mode == "footer" ? cucascade::io::open_hint::parquet_footer_probe
                                           : cucascade::io::open_hint::generic);
    require(object->size() == expected.size(), "size mismatch");
    require(object->validation_tag() == argv[4], "ETag mismatch");
    if (mode == "footer") {
      auto rest = std::dynamic_pointer_cast<cucascade::io::rest::rest_io_object>(object);
      require(rest && rest->stash() && rest->stash()->size() == 512, "missing suffix stash");
      require(rest->stash_window_lo() == expected.size() - 512, "suffix offset mismatch");
      require(std::equal(rest->stash()->begin(), rest->stash()->end(), expected.end() - 512),
              "suffix bytes mismatch");
    }
    if (mode == "head" || mode == "footer") {
      std::cout << "{\"event\":\"done\",\"status\":\"PASS\"}" << std::endl;
      return 0;
    }
    stage = "read";
    ctx->start();
    using future_type =
      decltype(ctx->host_read_async(*object, 0, 31, static_cast<std::uint8_t*>(nullptr)));
    std::vector<std::vector<std::uint8_t>> buffers(count, std::vector<std::uint8_t>(47, 0xa5));
    std::vector<future_type> pending;
    pending.reserve(count);
    bool all_ok  = true;
    auto collect = [&](std::size_t i, future_type&& future) {
      try {
        auto n = std::move(future).get();
        require(n == 31, "returned byte count mismatch");
        require(std::equal(buffers[i].begin(),
                           buffers[i].begin() + 31,
                           expected.begin() + static_cast<std::ptrdiff_t>(i * 37)),
                "payload mismatch");
        require(
          std::all_of(buffers[i].begin() + 31, buffers[i].end(), [](auto b) { return b == 0xa5; }),
          "buffer guard changed");
        std::cout << "{\"event\":\"result\",\"id\":" << i << ",\"offset\":" << i * 37
                  << ",\"length\":31,\"status\":\"PASS\"}" << std::endl;
      } catch (std::exception const& error) {
        all_ok = false;
        std::cout << "{\"event\":\"result\",\"id\":" << i << ",\"offset\":" << i * 37
                  << ",\"length\":31,\"status\":\"FAIL\",\"error\":" << json(error.what()) << '}'
                  << std::endl;
      }
    };
    // Every allocation and the object outlive ALL submitted futures, including errors.
    for (std::size_t i = 0; i < count; ++i) {
      try {
        pending.emplace_back(ctx->host_read_async(*object, i * 37, 31, buffers[i].data()));
      } catch (...) {
        for (std::size_t j = 0; j < pending.size(); ++j)
          if (mode == "queue") collect(j, std::move(pending[j]));
        ctx->shutdown();
        throw;
      }
      if (mode == "serial") collect(i, std::move(pending.back()));
    }
    if (mode == "queue") {
      // Runner releases server's payload gate only after observing this line.
      std::cout << "{\"event\":\"submitted\",\"count\":" << pending.size() << "}" << std::endl;
      for (std::size_t i = 0; i < pending.size(); ++i)
        collect(i, std::move(pending[i]));
    }
    ctx->shutdown();
    std::cout << "{\"event\":\"done\",\"status\":" << json(all_ok ? "PASS" : "FAIL")
              << ",\"count\":" << count << '}' << std::endl;
    return all_ok ? 0 : 1;
  } catch (std::exception const& error) {
    std::cout << "{\"event\":\"error\",\"stage\":" << json(stage)
              << ",\"error\":" << json(error.what()) << '}' << std::endl;
    return 1;
  }
}
