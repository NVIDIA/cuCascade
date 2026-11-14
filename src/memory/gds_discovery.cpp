/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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

#include "detail/gds_setup.hpp"

#include <cucascade/memory/gds_topology.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cucascade::memory {

namespace {

namespace fs = std::filesystem;

/// cuFile's default configuration file.
constexpr char const* default_cufile_json = "/etc/cufile.json";
/// A configuration file larger than this is not a cuFile configuration file.
constexpr std::uintmax_t max_cufile_json_bytes = 4UL << 20;

// ---------------------------------------------------------------------------
// A small JSON reader: enough for cufile.json, which is JSON plus comments.
// ---------------------------------------------------------------------------

struct json_value;
struct json_member;
using json_array  = std::vector<json_value>;
using json_object = std::vector<json_member>;

struct json_value {
  enum class kind : std::uint8_t { null, boolean, number, string, array, object };
  kind type{kind::null};
  bool boolean{false};
  double number{0.0};
  std::string text;
  json_array elements;
  json_object members;
};

struct json_member {
  std::string key;
  json_value value;
};

class json_reader {
 public:
  explicit json_reader(std::string_view text) : _text(text) {}

  /// Parse the whole text as one value; false with @c error() set on failure.
  bool parse(json_value& out)
  {
    skip_space();
    if (!value(out, 0)) { return false; }
    skip_space();
    if (!_error.empty()) { return false; }
    if (_pos != _text.size()) { return fail("unexpected trailing text"); }
    return true;
  }

  [[nodiscard]] std::string const& error() const noexcept { return _error; }

 private:
  static constexpr int max_depth = 64;

  bool fail(std::string_view what)
  {
    if (_error.empty()) {
      // Report a 1-based line and column, which is what a person editing the file needs.
      std::size_t line = 1;
      std::size_t col  = 1;
      for (std::size_t i = 0; i < _pos && i < _text.size(); ++i) {
        if (_text[i] == '\n') {
          ++line;
          col = 1;
        } else {
          ++col;
        }
      }
      _error =
        std::string{what} + " at line " + std::to_string(line) + ", column " + std::to_string(col);
    }
    return false;
  }

  [[nodiscard]] char peek() const noexcept { return _pos < _text.size() ? _text[_pos] : '\0'; }

  void skip_space()
  {
    while (_pos < _text.size()) {
      char const c = _text[_pos];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++_pos;
        continue;
      }
      if (c == '/' && _pos + 1 < _text.size() && _text[_pos + 1] == '/') {
        auto const end = _text.find('\n', _pos);
        _pos           = end == std::string_view::npos ? _text.size() : end;
        continue;
      }
      if (c == '/' && _pos + 1 < _text.size() && _text[_pos + 1] == '*') {
        auto const end = _text.find("*/", _pos + 2);
        if (end == std::string_view::npos) {
          fail("unterminated comment");
          _pos = _text.size();
          return;
        }
        _pos = end + 2;
        continue;
      }
      return;
    }
  }

  bool value(json_value& out, int depth)
  {
    if (!_error.empty()) { return false; }
    if (depth > max_depth) { return fail("nesting too deep"); }
    switch (peek()) {
      case '{': return object(out, depth);
      case '[': return array(out, depth);
      case '"': out.type = json_value::kind::string; return string(out.text);
      case 't':
        out.type    = json_value::kind::boolean;
        out.boolean = true;
        return word("true");
      case 'f':
        out.type    = json_value::kind::boolean;
        out.boolean = false;
        return word("false");
      case 'n': out.type = json_value::kind::null; return word("null");
      case '\0':
        if (_pos >= _text.size()) { return fail("unexpected end of text"); }
        break;
      default: break;
    }
    char const c = peek();
    if (c == '-' || (c >= '0' && c <= '9')) { return number(out); }
    return fail("unexpected character");
  }

  bool word(std::string_view expected)
  {
    if (_text.substr(_pos, expected.size()) != expected) { return fail("invalid literal"); }
    _pos += expected.size();
    return true;
  }

  bool object(json_value& out, int depth)
  {
    out.type = json_value::kind::object;
    ++_pos;  // '{'
    while (true) {
      skip_space();
      if (!_error.empty()) { return false; }
      if (peek() == '}') {  // empty object, or a trailing comma before the brace
        ++_pos;
        return true;
      }
      if (peek() != '"') { return fail("expected a quoted key"); }
      json_member member;
      if (!string(member.key)) { return false; }
      skip_space();
      if (peek() != ':') { return fail("expected ':'"); }
      ++_pos;
      skip_space();
      if (!value(member.value, depth + 1)) { return false; }
      out.members.push_back(std::move(member));
      skip_space();
      if (!_error.empty()) { return false; }
      if (peek() == ',') {
        ++_pos;
        continue;
      }
      if (peek() == '}') {
        ++_pos;
        return true;
      }
      return fail("expected ',' or '}'");
    }
  }

  bool array(json_value& out, int depth)
  {
    out.type = json_value::kind::array;
    ++_pos;  // '['
    while (true) {
      skip_space();
      if (!_error.empty()) { return false; }
      if (peek() == ']') {  // empty array, or a trailing comma before the bracket
        ++_pos;
        return true;
      }
      json_value element;
      if (!value(element, depth + 1)) { return false; }
      out.elements.push_back(std::move(element));
      skip_space();
      if (!_error.empty()) { return false; }
      if (peek() == ',') {
        ++_pos;
        continue;
      }
      if (peek() == ']') {
        ++_pos;
        return true;
      }
      return fail("expected ',' or ']'");
    }
  }

  static void append_utf8(std::string& out, unsigned code)
  {
    if (code < 0x80U) {
      out += static_cast<char>(code);
    } else if (code < 0x800U) {
      out += static_cast<char>(0xC0U | (code >> 6U));
      out += static_cast<char>(0x80U | (code & 0x3FU));
    } else {
      out += static_cast<char>(0xE0U | (code >> 12U));
      out += static_cast<char>(0x80U | ((code >> 6U) & 0x3FU));
      out += static_cast<char>(0x80U | (code & 0x3FU));
    }
  }

  bool string(std::string& out)
  {
    ++_pos;  // opening quote
    while (_pos < _text.size()) {
      char const c = _text[_pos++];
      if (c == '"') { return true; }
      if (static_cast<unsigned char>(c) < 0x20U) { return fail("control character in a string"); }
      if (c != '\\') {
        out += c;
        continue;
      }
      if (_pos >= _text.size()) { break; }
      char const escaped = _text[_pos++];
      switch (escaped) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (_text.size() - _pos < 4) { return fail("truncated \\u escape"); }
          unsigned code        = 0;
          auto const* first    = _text.data() + _pos;
          auto const [end, ec] = std::from_chars(first, first + 4, code, 16);
          if (ec != std::errc{} || end != first + 4) { return fail("invalid \\u escape"); }
          _pos += 4;
          // Surrogate pairs never occur in cufile.json; keep a replacement character.
          append_utf8(out, code >= 0xD800U && code <= 0xDFFFU ? 0xFFFDU : code);
          break;
        }
        default: return fail("invalid escape");
      }
    }
    return fail("unterminated string");
  }

  bool number(json_value& out)
  {
    auto const start = _pos;
    if (peek() == '-') { ++_pos; }
    auto digits = [this] {
      std::size_t count = 0;
      while (_pos < _text.size() && _text[_pos] >= '0' && _text[_pos] <= '9') {
        ++_pos;
        ++count;
      }
      return count;
    };
    if (digits() == 0) { return fail("invalid number"); }
    if (peek() == '.') {
      ++_pos;
      if (digits() == 0) { return fail("invalid number"); }
    }
    if (peek() == 'e' || peek() == 'E') {
      ++_pos;
      if (peek() == '+' || peek() == '-') { ++_pos; }
      if (digits() == 0) { return fail("invalid number"); }
    }
    double parsed        = 0.0;
    auto const* first    = _text.data() + start;
    auto const* last     = _text.data() + _pos;
    auto const [end, ec] = std::from_chars(first, last, parsed);
    if (ec != std::errc{} || end != last) { return fail("number out of range"); }
    out.type   = json_value::kind::number;
    out.number = parsed;
    return true;
  }

  std::string_view _text;
  std::size_t _pos{0};
  std::string _error;
};

/// The value of @p key in @p object (the last one when a key repeats, as most
/// JSON readers do), or null.
[[nodiscard]] json_value const* member(json_value const* object, std::string_view key) noexcept
{
  if (object == nullptr || object->type != json_value::kind::object) { return nullptr; }
  json_value const* found = nullptr;
  for (auto const& entry : object->members) {
    if (entry.key == key) { found = &entry.value; }
  }
  return found;
}

[[nodiscard]] std::optional<std::size_t> as_size(json_value const* value) noexcept
{
  // A non-negative integer that a double represents exactly.
  constexpr double max_exact = 9007199254740992.0;  // 2^53
  if (value == nullptr || value->type != json_value::kind::number) { return std::nullopt; }
  if (!(value->number >= 0.0) || value->number > max_exact ||
      std::floor(value->number) != value->number) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(value->number);
}

[[nodiscard]] std::optional<bool> as_bool(json_value const* value) noexcept
{
  if (value == nullptr || value->type != json_value::kind::boolean) { return std::nullopt; }
  return value->boolean;
}

[[nodiscard]] std::optional<std::string> as_string(json_value const* value)
{
  if (value == nullptr || value->type != json_value::kind::string) { return std::nullopt; }
  return value->text;
}

[[nodiscard]] std::vector<std::string> as_strings(json_value const* value)
{
  std::vector<std::string> out;
  if (value == nullptr || value->type != json_value::kind::array) { return out; }
  for (auto const& element : value->elements) {
    if (element.type == json_value::kind::string) { out.push_back(element.text); }
  }
  return out;
}

void extract_settings(json_value const& root, cufile_json_settings& out)
{
  auto const* properties            = member(&root, "properties");
  out.max_direct_io_size_kb         = as_size(member(properties, "max_direct_io_size_kb"));
  out.max_device_cache_size_kb      = as_size(member(properties, "max_device_cache_size_kb"));
  out.per_buffer_cache_size_kb      = as_size(member(properties, "per_buffer_cache_size_kb"));
  out.max_device_pinned_mem_size_kb = as_size(member(properties, "max_device_pinned_mem_size_kb"));
  out.use_poll_mode                 = as_bool(member(properties, "use_poll_mode"));
  out.poll_mode_max_size_kb         = as_size(member(properties, "poll_mode_max_size_kb"));
  out.allow_compat_mode             = as_bool(member(properties, "allow_compat_mode"));
  out.use_pci_p2pdma                = as_bool(member(properties, "use_pci_p2pdma"));
  out.rdma_dev_addr_list            = as_strings(member(properties, "rdma_dev_addr_list"));
  out.rdma_load_balancing_policy    = as_string(member(properties, "rdma_load_balancing_policy"));
  out.rdma_dynamic_routing          = as_bool(member(properties, "rdma_dynamic_routing"));

  auto const* execution        = member(&root, "execution");
  out.parallel_io              = as_bool(member(execution, "parallel_io"));
  out.max_io_threads           = as_size(member(execution, "max_io_threads"));
  out.max_io_queue_depth       = as_size(member(execution, "max_io_queue_depth"));
  out.min_io_threshold_size_kb = as_size(member(execution, "min_io_threshold_size_kb"));
  out.max_request_parallelism  = as_size(member(execution, "max_request_parallelism"));

  auto const* filesystems     = member(&root, "fs");
  auto const* gpfs            = member(filesystems, "gpfs");
  out.gpfs_gds_write_support  = as_bool(member(gpfs, "gds_write_support"));
  out.gpfs_gds_async_support  = as_bool(member(gpfs, "gds_async_support"));
  out.lustre_posix_gds_min_kb = as_size(member(member(filesystems, "lustre"), "posix_gds_min_kb"));
  out.beegfs_posix_gds_min_kb = as_size(member(member(filesystems, "beegfs"), "posix_gds_min_kb"));

  auto const* denylist   = member(&root, "denylist");
  out.denied_drivers     = as_strings(member(denylist, "drivers"));
  out.denied_devices     = as_strings(member(denylist, "devices"));
  out.denied_mounts      = as_strings(member(denylist, "mounts"));
  out.denied_filesystems = as_strings(member(denylist, "filesystems"));

  out.cufile_stats  = as_size(member(member(&root, "profile"), "cufile_stats"));
  out.logging_level = as_string(member(member(&root, "logging"), "level"));
}

[[nodiscard]] std::string trim(std::string_view text)
{
  auto const first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) { return {}; }
  auto const last = text.find_last_not_of(" \t\r\n");
  return std::string{text.substr(first, last - first + 1)};
}

/// Read a small text file whole; false when it cannot be opened or read.
[[nodiscard]] bool read_small_file(fs::path const& path, std::string& out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { return false; }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  if (in.bad()) { return false; }
  out = std::move(buffer).str();
  return true;
}

/// Merge @p from into @p into the way cuFile layers its configuration files:
/// objects merge member by member (recursively), anything else is replaced.
void merge_into(json_value& into, json_value const& from)
{
  if (into.type != json_value::kind::object || from.type != json_value::kind::object) {
    into = from;
    return;
  }
  for (auto const& member : from.members) {
    json_value* found = nullptr;
    for (auto& existing : into.members) {
      if (existing.key == member.key) found = &existing.value;  // the last one wins
    }
    if (found != nullptr) {
      merge_into(*found, member.value);
    } else {
      into.members.push_back(member);
    }
  }
}

/// Parse @p text into @p root (a JSON object); false with @p error set otherwise.
[[nodiscard]] bool parse_tree(std::string_view text, json_value& root, std::string& error)
{
  json_reader reader{text};
  if (!reader.parse(root)) {
    error = reader.error();
    return false;
  }
  if (root.type != json_value::kind::object) {
    error = "the top level is not a JSON object";
    return false;
  }
  return true;
}

/// What happened to one configuration file.
enum class file_outcome : std::uint8_t { missing, parsed, failed };

/// Read and parse @p path into @p root; a missing file leaves @p root alone.
[[nodiscard]] file_outcome load_tree(std::string const& path, json_value& root, std::string& error)
{
  std::error_code ec;
  auto const size = fs::file_size(path, ec);
  if (ec) {
    if (ec == std::errc::no_such_file_or_directory) return file_outcome::missing;
    error = path + ": cannot stat: " + ec.message();
    return file_outcome::failed;
  }
  if (size > max_cufile_json_bytes) {
    error = path + ": larger than " + std::to_string(max_cufile_json_bytes) + " bytes";
    return file_outcome::failed;
  }
  std::string text;
  if (!read_small_file(path, text)) {
    error = path + ": cannot be read";
    return file_outcome::failed;
  }
  json_value tree;
  std::string why;
  if (!parse_tree(text, tree, why)) {
    error = path + ": not valid JSON: " + why;
    return file_outcome::failed;
  }
  merge_into(root, tree);
  return file_outcome::parsed;
}

[[nodiscard]] std::optional<std::size_t> parse_size_value(std::string_view text) noexcept
{
  std::size_t value    = 0;
  auto const* first    = text.data();
  auto const* last     = text.data() + text.size();
  auto const [end, ec] = std::from_chars(first, last, value);
  if (ec != std::errc{} || end == first) { return std::nullopt; }
  return value;
}

}  // namespace

namespace detail {

bool parse_cufile_json(std::string_view text,
                       cufile_json_settings& out,
                       std::string& error) noexcept
{
  try {
    json_value root;
    if (!parse_tree(text, root, error)) return false;
    cufile_json_settings settings;
    extract_settings(root, settings);
    out = std::move(settings);
    return true;
  } catch (std::exception const& e) {
    error = e.what();
  } catch (...) {
    error = "unknown error";
  }
  return false;
}

std::string parse_nvidia_fs_version(std::string_view text)
{
  auto const newline = text.find('\n');
  auto line          = text.substr(0, newline);
  if (auto const colon = line.find(':'); colon != std::string_view::npos) {
    line = line.substr(colon + 1);
  }
  return trim(line);
}

void parse_nvidia_fs_stats(std::string_view text, nvidia_fs_info& out)
{
  std::istringstream lines{std::string{text}};
  std::string line;
  auto after = [](std::string const& from, std::string_view key) -> std::optional<std::string> {
    auto const at = from.find(key);
    if (at == std::string::npos) { return std::nullopt; }
    return trim(std::string_view{from}.substr(at + key.size()));
  };
  while (std::getline(lines, line)) {
    if (auto value = after(line, "GDS Version:"); value && line.starts_with("GDS Version")) {
      out.gds_version = *value;
      continue;
    }
    if (auto value = after(line, "NVFS Driver(version:"); value) {
      if (out.version.empty()) {
        auto const close = value->find(')');
        out.version      = trim(std::string_view{*value}.substr(0, close));
      }
      continue;
    }
    if (auto value = after(line, "PeerDirect Supported:"); value) {
      out.peer_direct_supported = value->starts_with("True") || value->starts_with("true");
      continue;
    }
    if (auto value = after(line, "IO stats:"); value && line.starts_with("IO stats")) {
      out.io_stats_enabled = value->starts_with("Enabled") || value->starts_with("enabled");
      continue;
    }
    // "GPU 0000:be:00.0  uuid:... : Registered_MiB=0 Cache_MiB=0 max_pinned_MiB=1
    //  cross_root_port(%)=0"
    if (line.starts_with("GPU ")) {
      std::istringstream fields{line};
      std::string token;
      nvidia_fs_gpu_stats gpu;
      fields >> token >> gpu.pci_bus_id;
      while (fields >> token) {
        auto const equals = token.find('=');
        if (equals == std::string::npos) { continue; }
        auto const key   = std::string_view{token}.substr(0, equals);
        auto const value = parse_size_value(std::string_view{token}.substr(equals + 1));
        if (!value) { continue; }
        if (key == "Registered_MiB") {
          gpu.registered_mib = *value;
        } else if (key == "Cache_MiB") {
          gpu.cache_mib = *value;
        } else if (key == "max_pinned_MiB") {
          gpu.max_pinned_mib = *value;
        } else if (key == "cross_root_port(%)") {
          gpu.cross_root_port_pct = *value;
        }
      }
      if (!gpu.pci_bus_id.empty()) { out.gpus.push_back(std::move(gpu)); }
    }
  }
}

std::vector<std::string> live_modules(std::string_view proc_modules)
{
  // Rows are "<name> <size> <refcount> <deps> <state> <address>".
  std::vector<std::string> live;
  std::istringstream lines{std::string{proc_modules}};
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream fields{line};
    std::string name, size, refcount, deps, state;
    if ((fields >> name >> size >> refcount >> deps >> state) && state == "Live") {
      live.push_back(std::move(name));
    }
  }
  return live;
}

bool force_compat_value(char const* value) noexcept
{
  if (value == nullptr) { return false; }
  constexpr char upper[] = "TRUE";
  for (int i = 0; i < 4; ++i) {
    // A terminator never matches a letter, so short values stop here.
    if (value[i] != upper[i] && value[i] != upper[i] + ('a' - 'A')) { return false; }
  }
  return true;
}

}  // namespace detail

cufile_config_file read_cufile_config(std::string const& path, std::string const& override_path)
{
  cufile_config_file file;
  try {
    file.path          = path;
    file.override_path = override_path;
    file.examined      = true;
    json_value root;
    root.type = json_value::kind::object;
    std::string error;
    bool failed       = false;
    std::size_t found = 0;
    for (auto const& name : {path, override_path}) {
      if (name.empty()) continue;
      std::string why;
      switch (load_tree(name, root, why)) {
        case file_outcome::missing: break;
        case file_outcome::parsed: ++found; break;
        case file_outcome::failed:
          ++found;
          failed = true;
          error += (error.empty() ? "" : "; ") + why;
          break;
      }
    }
    file.found  = found != 0;
    file.parsed = !failed;
    if (failed) {
      file.error = error;
    } else {
      extract_settings(root, file.settings);
      if (!file.found) file.error = "not found (cuFile uses its built-in defaults)";
    }
  } catch (std::exception const& e) {
    file.parsed = false;
    file.error  = e.what();
  } catch (...) {
    file.parsed = false;
    file.error  = "unknown error";
  }
  return file;
}

gds_setup_info discover_gds_setup(discovery_roots const& roots,
                                  std::string_view cufile_json_path,
                                  std::optional<std::string> const& override_json_path)
{
  gds_setup_info info;
  try {
    std::string text;
    if (read_small_file(roots.procfs_root / "proc" / "modules", text)) {
      auto const modules = detail::live_modules(text);
      auto live          = [&modules](std::initializer_list<std::string_view> names) {
        return std::any_of(modules.begin(), modules.end(), [&names](std::string const& module) {
          return std::find(names.begin(), names.end(), module) != names.end();
        });
      };
      info.nvidia_fs.loaded     = live({"nvidia_fs"});
      info.peer_memory_loaded   = live({"nvidia_peermem", "nv_peer_mem"});
      info.rdma_core_loaded     = live({"ib_core"});
      info.gpfs_client_loaded   = live({"mmfs26", "mmfslinux"});
      info.lustre_client_loaded = live({"lustre"});
      info.beegfs_client_loaded = live({"beegfs"});
      info.wekafs_client_loaded = live({"wekafsio"});
      info.nfs_rdma_loaded      = live({"rpcrdma"});
    }

    // The module's own version attribute is the cleanest source; the driver's
    // procfs files are the fallback (and carry the GDS-wide details).
    if (read_small_file(roots.sysfs_root / "sys" / "module" / "nvidia_fs" / "version", text)) {
      info.nvidia_fs.version = detail::parse_nvidia_fs_version(text);
    }
    auto const nvfs_proc = roots.procfs_root / "proc" / "driver" / "nvidia-fs";
    if (info.nvidia_fs.version.empty() && read_small_file(nvfs_proc / "version", text)) {
      info.nvidia_fs.version = detail::parse_nvidia_fs_version(text);
    }
    if (read_small_file(nvfs_proc / "stats", text)) {
      detail::parse_nvidia_fs_stats(text, info.nvidia_fs);
    }

    info.force_compat_env = detail::force_compat_value(std::getenv("CUFILE_FORCE_COMPAT_MODE"));

    std::string const path =
      cufile_json_path.empty() ? std::string{default_cufile_json} : std::string{cufile_json_path};
    std::string override_path;
    bool from_environment = false;
    if (override_json_path) {
      override_path = *override_json_path;
    } else if (char const* from_env = ::secure_getenv("CUFILE_ENV_PATH_JSON");
               from_env != nullptr && *from_env != '\0') {
      // cuFile reads the variable with secure_getenv, so this does too.
      override_path    = from_env;
      from_environment = true;
    }
    info.cufile_json                  = read_cufile_config(path, override_path);
    info.cufile_json.from_environment = from_environment;
  } catch (...) {
    // Discovery is best effort: keep whatever was learned.
  }
  return info;
}

}  // namespace cucascade::memory
