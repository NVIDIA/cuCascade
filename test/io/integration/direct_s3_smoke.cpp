// SPDX-License-Identifier: Apache-2.0
// Local integration probe: all object reads use cuCascade's public REST API.
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, char const* message)
{
  if (!condition) throw std::runtime_error(message);
}

std::string required_env(char const* name)
{
  auto const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') throw std::runtime_error(std::string("missing ") + name);
  return value;
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    require(argc == 6, "usage: direct_s3_smoke ENDPOINT S3_URI FILE ETAG MODE(read|404|403)");
    cucascade::io::rest::config cfg{};
    cfg.request_timeout_s       = 5;
    cfg.stall_time_s            = 5;
    cfg.max_retry_attempts      = 2;
    cfg.max_auth_retry_attempts = 1;
    cfg.max_connections         = 2;
    cfg.footer_probe_bytes      = 512;
    cfg.retry_backoff_base      = std::chrono::milliseconds{10};
    cfg.retry_jitter            = std::chrono::milliseconds{0};
    cfg.honor_retry_after       = false;
    cucascade::io::rest::s3::static_credentials creds{};
    creds.access_key_id     = required_env("CUCASCADE_TEST_ACCESS_KEY");
    creds.secret_access_key = required_env("CUCASCADE_TEST_SECRET_KEY");
    auto auth               = std::make_shared<cucascade::io::rest::s3::sigv4_header_authorizer>(
      std::move(creds), "us-east-1", argv[1]);
    auto reactor_context =
      std::make_shared<cucascade::io::rest::rest_reactor::reactor_context>(cfg, auth, nullptr);
    auto ctx = std::make_shared<cucascade::io::rest::rest_ioctx>(1, reactor_context);
    std::string const mode{argv[5]};
    if (mode == "404" || mode == "403") {
      try {
        (void)ctx->open_io_object(argv[2]);
      } catch (std::exception const& error) {
        require(std::string(error.what()).find("HTTP " + mode + " for ") != std::string::npos,
                "failure did not contain the expected HTTP status");
        std::cout << "{\"status\":\"PASS\",\"expected_http\":" << mode << "}\n";
        return 0;
      }
      throw std::runtime_error("expected HTTP error but object opened successfully");
    }
    require(mode == "read", "unknown mode");
    std::ifstream input(argv[3], std::ios::binary);
    require(input.good(), "cannot open expected bytes");
    std::vector<std::uint8_t> expected{std::istreambuf_iterator<char>(input), {}};
    auto object = ctx->open_io_object(argv[2]);
    require(object->size() == expected.size(), "HEAD size mismatch");
    require(object->validation_tag() == argv[4], "HEAD ETag mismatch");
    ctx->start();
    std::size_t checks{2};
    auto check_read = [&](std::size_t offset, std::size_t length, bool asynchronous) {
      std::vector<std::uint8_t> bytes(length + 16, 0xa5);
      // Buffer and object remain alive until the public future completes.
      auto const n =
        asynchronous ? std::move(ctx->host_read_async(*object, offset, length, bytes.data())).get()
                     : ctx->host_read(*object, offset, length, bytes.data());
      auto const want = offset < expected.size() ? std::min(length, expected.size() - offset) : 0;
      require(n == want, "read count mismatch");
      for (std::size_t i = 0; i < n; ++i)
        require(bytes[i] == expected[offset + i], "read bytes mismatch");
      for (std::size_t i = n; i < bytes.size(); ++i)
        require(bytes[i] == 0xa5, "read wrote past returned byte count");
      ++checks;
    };
    check_read(0, expected.size(), false);
    for (std::size_t offset = 0; offset < expected.size(); offset += 65536)
      check_read(offset, std::min(std::size_t{65536}, expected.size() - offset), true);
    check_read(0, 17, true);
    check_read(expected.size() / 2, 31, false);
    check_read(expected.empty() ? 0 : expected.size() - 1, 17, true);
    check_read(expected.size(), 16, false);
    check_read(expected.size() + 7, 16, true);
    check_read(0, 0, false);
    auto footer = std::dynamic_pointer_cast<cucascade::io::rest::rest_io_object>(
      ctx->open_io_object(argv[2], cucascade::io::open_hint::parquet_footer_probe));
    require(footer != nullptr && footer->size() == expected.size(), "footer size mismatch");
    require(footer->validation_tag() == argv[4], "footer ETag mismatch");
    if (!expected.empty()) {
      auto const& stash = footer->stash();
      auto const n      = std::min(expected.size(), std::size_t{512});
      require(stash != nullptr && stash->size() == n, "suffix GET did not provide expected stash");
      require(footer->stash_window_lo() == expected.size() - n, "suffix offset mismatch");
      for (std::size_t i = 0; i < n; ++i)
        require((*stash)[i] == expected[expected.size() - n + i], "suffix bytes mismatch");
    }
    ++checks;
    ctx->shutdown();
    std::cout << "{\"status\":\"PASS\",\"bytes\":" << expected.size() << ",\"checks\":" << checks
              << ",\"etag\":" << std::quoted(std::string(object->validation_tag())) << "}\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "direct_s3_smoke: " << error.what() << '\n';
    return 1;
  }
}
