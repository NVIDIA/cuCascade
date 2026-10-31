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

#pragma once

#include "io/rest/common.hpp"
#include "io/rest/errc.hpp"

#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

/**
 * @file multipart.hpp
 * @brief S3 writes without a transport: the multipart protocol helpers (ported from gpuio's
 *        `src/s3/write.cpp` and the ETag helpers of `src/s3/s3_client.cpp`) and @ref
 *        cucascade::io::rest::s3::upload_state, the thread-safe state machine of one object
 *        being written (docs/io/rest-uring-design.md section 5) that the REST reactor drives.
 *
 * Protocol (S3 API): CreateMultipartUpload = `POST <key>?uploads` (the UploadId comes back in
 * an `InitiateMultipartUploadResult`), UploadPart = `PUT <key>?partNumber=N&uploadId=U` (the
 * part's ETag comes back in the `ETag` header), CompleteMultipartUpload = `POST
 * <key>?uploadId=U` with the ETags of every part in part-number order (the object's ETag comes
 * back in a `CompleteMultipartUploadResult`), AbortMultipartUpload = `DELETE <key>?uploadId=U`.
 * An object of at most one part that was never uploaded is written with one `PUT <key>`.
 * Bodies are signed `UNSIGNED-PAYLOAD` (gpuio S3); no checksums are sent.
 *
 * Nothing here performs I/O, allocates pinned memory or touches CUDA.
 */

namespace cucascade::io::rest::s3 {

//===----------------------------------------------------------------------===//
// Limits and constants (gpuio src/s3/write.cpp:47-57, remote_request.hpp:234-235)
//===----------------------------------------------------------------------===//

inline constexpr unsigned max_part_number     = 10000;    ///< S3 multipart limit (part numbers)
inline constexpr std::uint64_t min_part_bytes = 5 * MiB;  ///< S3: every part but the last
inline constexpr std::uint64_t max_part_bytes = 5 * GiB;  ///< S3: one part
inline constexpr std::uint64_t max_put_bytes  = 5 * GiB;  ///< S3: one single PUT
/// One SENDMSG window: the request head plus at most this many body iovecs (gpuio W1/W2).
inline constexpr std::size_t max_put_sources = 255;
/// Create / Complete response window (gpuio T18b): a 1024-byte key XML-escaped (<= ~5 KiB) plus
/// the Location. A larger CreateMultipartUpload answer fails (no UploadId); a larger Complete
/// answer succeeded and takes its ETag from a HEAD.
inline constexpr std::size_t xml_rx_bytes = 8 * KiB;
/// CompleteMultipartUpload attempts when S3 answers `200 OK` with an `<Error>` body (the S3
/// documentation says to retry; gpuio W2).
inline constexpr unsigned complete_attempts = 3;
/// ETag storage per part (gpuio `part_slot::etag`); a longer part ETag is `http_bad_message`.
inline constexpr std::size_t max_etag_bytes = 80;
/// The CreateMultipartUpload sub-resource query (SigV4 canonical form: "uploads=").
inline constexpr std::string_view uploads_query = "uploads";

//===----------------------------------------------------------------------===//
// Tiny XML (namespace-free S3 documents; cold paths)
//===----------------------------------------------------------------------===//

/// @return the text of the first `<tag>...</tag>` in @p doc (not unescaped), or empty. Tags
/// longer than 60 characters are not searched.
[[nodiscard]] std::string_view xml_text(std::string_view doc, std::string_view tag) noexcept;

/// @return @p s with the five predefined XML entities (and `&#34;`) decoded.
[[nodiscard]] std::string xml_unescape(std::string_view s);

/// Append @p s to @p out with `"`, `&`, `<`, `>` escaped.
void xml_escape_append(std::string& out, std::string_view s);

//===----------------------------------------------------------------------===//
// S3 error documents
//===----------------------------------------------------------------------===//

/// `<Code>` and `<Message>` of an S3 `<Error>` document (whitespace-trimmed, not unescaped).
struct s3_error_info {
  std::string_view code;
  std::string_view message;
};

/// @return the code and message when @p body is an S3 error document (contains `<Error>`),
/// as views into @p body.
/// @note The `200 OK` + `<Error>` interpretation below takes the code from the HTTP codec's
/// `http::extract_error_code` (src/io/rest/http/classify.hpp), which fills a `response_meta`
/// and has no `<Message>`; this helper is the bare-body variant with the message, for logs and
/// tests.
[[nodiscard]] std::optional<s3_error_info> parse_s3_error(std::string_view body) noexcept;

/// @brief Map an S3 `<Code>` carried by a `200 OK` body to an error (gpuio `body_error`).
/// @param[out] retriable set for `InternalError`, `ServiceUnavailable` and `SlowDown`.
/// @return `errc::throttled` (SlowDown), HTTP 500 / 503 / 404 (`NoSuchUpload`, `NoSuchKey`) /
/// 403 (`AccessDenied`), else HTTP 400 (`make_http_error`).
[[nodiscard]] std::error_code s3_code_error(std::string_view code, bool& retriable) noexcept;

/// Interpretation of a CreateMultipartUpload / CompleteMultipartUpload `2xx` body.
struct xml_outcome {
  std::error_code ec;      ///< failure (the request fails; not retried unless @c retriable)
  bool retriable{false};   ///< a `200 OK` + `<Error>` that S3 says to retry
  std::string value;       ///< Create: the UploadId; Complete: the object ETag (may be empty)
  std::string error_code;  ///< the S3 `<Code>` when the body was an error document
};

/**
 * @brief CreateMultipartUpload answered 2xx: the `InitiateMultipartUploadResult` UploadId.
 * @param body the captured body (at most @ref xml_rx_bytes).
 * @param overflow the body did not fit the window: there is no UploadId (`http_bad_message`).
 * @param header_error_code an S3 error code already known for the response (empty: none).
 * @return the UploadId (unescaped) in @c value; an `<Error>` body is mapped by @ref
 *         s3_code_error; a body without `<UploadId>` is `http_bad_message`.
 */
[[nodiscard]] xml_outcome interpret_create_response(std::string_view body,
                                                    bool overflow,
                                                    std::string_view header_error_code = {});

/**
 * @brief CompleteMultipartUpload answered 2xx: success, or the "200 OK with an S3 `<Error>`"
 *        case (S3 sends the status before it finished the assembly).
 * @param overflow the body did not fit the window: success without an ETag (take it from a
 *        HEAD; gpuio T18b).  The engine re-issues an overflowing capture with a larger window
 *        instead, so it always passes false.
 * @return the object ETag (unescaped; empty when the `CompleteMultipartUploadResult` did not
 *         carry one) in @c value, or the mapped error with @c retriable.  A 2xx body that is
 *         neither an `<Error>` nor a `CompleteMultipartUploadResult` (a proxy page, keep-alive
 *         whitespace only) is `http_bad_message`, not retried.
 */
[[nodiscard]] xml_outcome interpret_complete_response(std::string_view body,
                                                      bool overflow,
                                                      std::string_view header_error_code = {});

//===----------------------------------------------------------------------===//
// Request building
//===----------------------------------------------------------------------===//

/// @return `"uploadId=<id, RFC 3986-encoded with '/' encoded>"` (Complete / Abort query).
[[nodiscard]] std::string upload_id_query(std::string_view upload_id);

/// @return `"partNumber=<n>&uploadId=<encoded id>"` (UploadPart query, SigV4 key order).
[[nodiscard]] std::string part_query(unsigned part_number, std::string_view upload_id);

/// One `<Part>` of a CompleteMultipartUpload request.
struct part_etag {
  unsigned part_number{0};
  std::string_view etag;  ///< as received (quotes kept); escaped into the XML
};

/// @return the CompleteMultipartUpload request body for @p parts (in the given order, which must
/// be ascending part numbers): `<CompleteMultipartUpload xmlns=...><Part><PartNumber>N
/// </PartNumber><ETag>E</ETag></Part>...</CompleteMultipartUpload>`.
[[nodiscard]] std::string complete_multipart_xml(std::span<part_etag const> parts);

//===----------------------------------------------------------------------===//
// ETag helpers (gpuio src/s3/s3_client.cpp:147-166)
//===----------------------------------------------------------------------===//

/// @return the multipart part count of an ETag `"<md5>-N"` (quotes kept), or nullopt.
[[nodiscard]] std::optional<unsigned> etag_part_count(std::string_view etag) noexcept;

/// @return a part size consistent with @p size bytes in @p parts parts: the smallest
/// power-of-two MiB in [ceil(size / parts), floor((size - 1) / (parts - 1))] when one exists
/// (8/16/64 MiB tool defaults), else ceil(size / parts) rounded up to a MiB; 0 for one part. The
/// ETag gives the count only; the estimate only steers alignment and the per-part caps, never
/// correctness.
[[nodiscard]] std::uint64_t estimate_part_size(std::uint64_t size, unsigned parts) noexcept;

//===----------------------------------------------------------------------===//
// upload_state
//===----------------------------------------------------------------------===//

/// S3 limits an @ref upload_state enforces (tests may lower @c min_part_bytes).
struct upload_limits {
  std::uint64_t min_part_bytes{s3::min_part_bytes};  ///< every part but the last
  std::uint64_t max_part_bytes{s3::max_part_bytes};  ///< largest part
  unsigned max_parts{max_part_number};               ///< part numbers 1..max_parts
};

/**
 * @brief The part size of a new upload (docs/io/rest-uring-design.md section 5).
 *
 * @p configured (`write_part_size`) unless @p size_hint needs more than `limits.max_parts`
 * parts of it; then the smallest multiple of @p granule that fits @p size_hint in
 * `limits.max_parts` parts, clamped to `limits.max_part_bytes`.
 */
[[nodiscard]] std::uint64_t choose_part_size(std::uint64_t configured,
                                             std::uint64_t size_hint,
                                             std::uint64_t granule,
                                             upload_limits const& limits = {}) noexcept;

/// Memory a write slice comes from.
enum class write_source : std::uint8_t { host, device };

/**
 * @brief The pinned assembly buffer of one part (opaque to @ref upload_state).
 *
 * The reactor fills @c iov with pinned blocks that cover the part's bytes [0, part size) in
 * order and keeps them alive through @c owner (e.g. the host-MR block handle).
 */
struct assembly_buffer {
  std::vector<iovec> iov;
  std::shared_ptr<void> owner;
  /// @return the sum of the iovec lengths.
  [[nodiscard]] std::uint64_t capacity() const noexcept;
};

/// How a write slice is served (@ref upload_state::classify_write).
struct write_plan {
  enum class kind : std::uint8_t {
    direct,    ///< host slice covering the whole part: UploadPart from the caller's buffer
    assemble,  ///< copy (memcpy / D2H) into the part's assembly buffer
  };
  kind what{kind::assemble};
  unsigned part{0};                 ///< 1-based part number
  file_range part_range{};          ///< object bytes of the part: [(part-1)P, part*P)
  std::uint64_t offset_in_part{0};  ///< where the slice lands in the part
  /// assemble: the part's buffer; null: allocate one of the part size and @ref
  /// upload_state::attach_buffer it before copying.
  std::shared_ptr<assembly_buffer const> buffer;
};

/// Who started an upload: a write (its future reports the result) or a flush.
enum class upload_origin : std::uint8_t { write, flush };

/**
 * @brief One part upload the caller performs and settles with exactly one @ref
 *        upload_state::finish_upload (passing this object back).
 */
struct part_upload {
  unsigned part{0};    ///< 1-based part number
  file_range range{};  ///< object bytes of the body: [(part-1)P, (part-1)P + size)
  /// The body: the first `size()` bytes of the buffer's iovecs. Null: the caller's host slice
  /// (direct upload) or, with @c zeros, `size()` zero bytes.
  std::shared_ptr<assembly_buffer const> buffer;
  bool zeros{false};  ///< the part was never written: the body is zeros
  /// Part-relative ranges of @c buffer no write covered: zero them before sending (a gap reads
  /// back as zeros). Empty for full parts.
  std::vector<file_range> zero_fill;
  upload_origin origin{upload_origin::write};
  std::uint64_t generation{0};  ///< content generation at the start (internal)

  /// @return the body size.
  [[nodiscard]] std::uint64_t size() const noexcept { return range.size; }
};

/// Opaque handle of a waiting operation (e.g. the reactor's op slot or user_data).
using waiter_id = std::uint64_t;

/// @ref upload_state::need_upload_id's answer.
struct upload_id_status {
  enum class kind : std::uint8_t {
    ready,   ///< the UploadId exists (@c upload_id, @c id_query)
    create,  ///< the caller sends CreateMultipartUpload, then set_upload_id / upload_id_failed
    wait,    ///< a creation is in flight; the waiter is returned by its settlement
  };
  kind what{kind::wait};
  std::string_view upload_id;  ///< ready: stable for the lifetime of the state
  std::string_view id_query;   ///< ready: "uploadId=<encoded>", stable likewise
};

/// Waiters handed back by an operation: resume them (success) or fail them with @c ec.
struct waiters_to_resume {
  std::error_code ec;
  std::vector<waiter_id> waiters;
};

/// Result of the operations that settle a write or an upload.
struct settle_result {
  /// The error the settled write (or flush step) reports; success otherwise.
  std::error_code ec;
  /// Flushes that waited for the writes and uploads in flight to drain: call plan_flush again.
  std::vector<waiter_id> flush_waiters;
};

/// @ref upload_state::note_assembled's answer.
struct assembled_result {
  std::error_code ec;  ///< the write fails with this error (its bytes were not taken)
  /// Engaged exactly once per fill: this write completed the part's coverage and now owns its
  /// upload (get the UploadId, send it, then finish_upload; the write completes then).
  std::optional<part_upload> upload;
  std::vector<waiter_id> flush_waiters;  ///< as settle_result::flush_waiters
};

/// @ref upload_state::plan_flush's answer.
struct flush_plan {
  enum class kind : std::uint8_t {
    committed,   ///< already committed: the flush is a no-op success
    join,        ///< another flush is running; its complete_* returns this waiter
    wait,        ///< writes / uploads in flight; a settle returns this waiter: plan again
    single_put,  ///< one PUT of `uploads[0]` (part 1, real size), then complete_*
    multipart,   ///< upload `uploads`, then complete_xml() + Complete, then complete_*
  };
  kind what{kind::wait};
  std::uint64_t object_size{0};
  unsigned parts{0};                 ///< N: the object's part count
  bool if_none_match{false};         ///< create_new: `If-None-Match: *` on the PUT / Complete
  bool need_upload_id{false};        ///< multipart without an UploadId yet: need_upload_id first
  std::vector<part_upload> uploads;  ///< claimed: settle each with finish_upload
  /// single_put only: an UploadId that exists but carries nothing of the object; abort it
  /// (best effort) once the PUT committed.
  std::optional<std::string> orphan_upload_id;
};

/// @ref upload_state::begin_abort's answer.
struct abort_result {
  std::optional<std::string> upload_id;  ///< send AbortMultipartUpload for it (best effort)
  std::vector<waiter_id> waiters;        ///< every waiter: fail it with `bad_file_descriptor`
};

/// Lifecycle of one part (@ref upload_state::info).
enum class part_phase : std::uint8_t { empty, filling, uploading, uploaded };

/// Snapshot of one part (tests and diagnostics).
struct part_info {
  part_phase phase{part_phase::empty};
  std::uint64_t covered_bytes{0};  ///< bytes of the part covered by assembled writes
  bool has_buffer{false};
  bool reproducible{true};  ///< the part's content is its buffer + zeros (partial writes allowed)
  bool dirty{false};        ///< written since its last successful upload
  unsigned outstanding{0};  ///< uploads claimed and not finished
  std::uint64_t uploaded_size{0};
  std::string etag;  ///< of the last successful upload
};

/**
 * @brief State machine of one S3 object being written (docs/io/rest-uring-design.md section 5).
 *
 * Transport-agnostic and thread-safe (one mutex: every reactor of the ioctx may drive the same
 * object). Part size P; part `k + 1` holds the object bytes `[kP, (k+1)P)` (1-based part
 * numbers, 1..`max_parts`); a write past `max_parts × P` is `file_too_large` (EFBIG). The
 * caller (the reactor) owns the transport, the pinned buffers and the waiters; the state
 * decides and accounts.
 *
 * @par Write path
 * - `classify_write(offset, size, source)` for a slice inside one part (templated_ioctx cuts
 *   write slices at multiples of `write_split_bytes == P`):
 *   - `direct`: a host slice covering the whole part of a part without an assembly buffer.
 *     Get the UploadId (`need_upload_id`), then `begin_upload(part)`, send UploadPart from the
 *     caller's buffer, `finish_upload`. The write completes when S3 answered.
 *   - `assemble`: attach a buffer when `plan.buffer` is null (`attach_buffer`; the first
 *     attached buffer wins), copy the slice into it, then `note_assembled(part, range)`. The
 *     write completes once copied, unless `note_assembled` hands it the part's upload (the
 *     write that completes a part's coverage uploads it and completes when S3 answered).
 *   - Every successful `classify_write` is settled exactly once: `assemble` by
 *     `note_assembled` or `abandon_write`; `direct` by `begin_upload` (which hands the
 *     accounting to the upload) or `abandon_write`. Every `part_upload` handed out (by
 *     `begin_upload`, `note_assembled`, `plan_flush`) is settled by exactly one
 *     `finish_upload`.
 * - A part that was uploaded from its buffer by a write is released (buffered memory stays
 *   bounded by the parts being filled); its bytes then exist only in S3, so a later partial
 *   write into it is `not_supported` (rewrite the whole part from host memory instead: a direct
 *   upload replaces it; S3 keeps the last upload of a part number). Parts uploaded by a flush
 *   keep their buffer until the commit, so writes after a failed flush still merge.
 *
 * @par Upload id
 * Created lazily by the first upload that needs it: `need_upload_id(waiter)` answers `ready`,
 * `create` (the caller sends CreateMultipartUpload — never retried after a lost response — then
 * `set_upload_id` or `upload_id_failed`, which hand back the waiters) or `wait`.
 *
 * @par Flush (= commit)
 * `plan_flush(object_size, waiter)` waits (`wait`) until every write and upload in flight
 * settled; new writes are refused meanwhile (`device_or_resource_busy`: writes concurrent with
 * a flush are not supported on an object store). Then it plans: `single_put` when the object
 * fits in part 1 and no UploadId exists (also for an empty object); else `multipart`: every
 * part 1..N whose current content is not in S3 yet (the last part with its real size; gaps
 * zero-filled through `zero_fill`; never-written parts as `zeros`). After the uploads, build
 * the request with `complete_xml()`, send Complete (with `If-None-Match: *` when `create_new`;
 * a `200 OK` + `<Error>` is retried up to @ref complete_attempts times), then
 * `complete_succeeded(etag)` (committed: flush again is a no-op, writes fail with
 * `bad_file_descriptor`, buffers are released) or `complete_failed(ec)` (the upload stays
 * usable: write or flush again). A second flush while one runs joins it.
 *
 * @par Errors
 * Sticky first error: a failure on the write path (a write-started upload, a copy reported
 * through `abandon_write`, an UploadId creation that failed waiting writes, or `fail(ec)`)
 * poisons the upload: every later write and flush fails with that first error, so an object
 * whose content a failed write left unknown is never committed. Failures of a flush (its part
 * uploads, the UploadId creation it triggered, Complete) are not sticky.
 *
 * @par Destruction
 * Destroying a state that has an UploadId and was neither committed nor aborted through
 * `begin_abort` calls the abort hook with the UploadId (AbortMultipartUpload, best effort).
 */
class upload_state {
 public:
  /// Configuration of one upload.
  struct options {
    std::uint64_t part_size{16 * MiB};  ///< P (`write_part_size`, or choose_part_size)
    bool create_new{false};             ///< `If-None-Match: *` on the final PUT / Complete
    upload_limits limits{};
  };

  /// @throw std::invalid_argument when P is outside [limits.min_part_bytes,
  ///        limits.max_part_bytes] or `limits.max_parts` is 0 or above @ref max_part_number.
  explicit upload_state(options const& opts);
  ~upload_state();

  upload_state(upload_state const&)            = delete;
  upload_state& operator=(upload_state const&) = delete;
  upload_state(upload_state&&)                 = delete;
  upload_state& operator=(upload_state&&)      = delete;

  // -- configuration ---------------------------------------------------------

  [[nodiscard]] std::uint64_t part_size() const noexcept { return _part_size; }
  [[nodiscard]] bool create_new() const noexcept { return _create_new; }
  [[nodiscard]] upload_limits const& limits() const noexcept { return _limits; }
  /// @return `max_parts × P`: writes ending past it are `file_too_large`.
  [[nodiscard]] std::uint64_t max_object_size() const noexcept;
  /// @return the 1-based part number holding @p offset.
  [[nodiscard]] unsigned part_of(std::uint64_t offset) const noexcept;
  /// @return the object bytes of @p part: [(part-1)P, part*P).
  [[nodiscard]] file_range part_range(unsigned part) const noexcept;

  // -- write path ------------------------------------------------------------

  /**
   * @brief Decide how the write slice [offset, offset + size) is served.
   * @return the plan; `bad_file_descriptor` after commit or abort; the sticky error;
   *         `device_or_resource_busy` while a flush waits or runs; `invalid_argument` for an
   *         empty slice or one that crosses a part boundary; `file_too_large` past
   *         max_object_size(); `not_supported` for a partial write into a part whose bytes are
   *         only in S3.
   */
  [[nodiscard]] result<write_plan> classify_write(std::uint64_t offset,
                                                  std::uint64_t size,
                                                  write_source source);

  /**
   * @brief Attach @p buffer (capacity >= P) to @p part for an `assemble` write.
   * @return the part's buffer: @p buffer, or the one another writer attached first (then drop
   *         yours). `invalid_argument` for a too small or empty buffer; `bad_file_descriptor` /
   *         the sticky error. An error leaves the write classified: settle it with
   *         abandon_write.
   */
  [[nodiscard]] result<std::shared_ptr<assembly_buffer const>> attach_buffer(
    unsigned part, std::shared_ptr<assembly_buffer> buffer);

  /**
   * @brief An `assemble` write copied @p range (object coordinates, inside @p part) into the
   *        part's buffer: settles the write's accounting and records the coverage.
   * @return see @ref assembled_result: the upload when this write completed the part.
   */
  [[nodiscard]] assembled_result note_assembled(unsigned part, file_range range);

  /**
   * @brief A classified write ends without its bytes reaching the state (copy failed,
   *        cancelled, no UploadId, ...). @p ec becomes the sticky error.
   */
  [[nodiscard]] settle_result abandon_write(unsigned part, std::error_code ec);

  /**
   * @brief A `direct` write starts its UploadPart (call it once the UploadId is ready).
   * @return the upload to send from the caller's buffer (settle with finish_upload);
   *         `bad_file_descriptor` / the sticky error, or `invalid_argument` without an UploadId.
   *         An error leaves the write classified: settle it with abandon_write.
   */
  [[nodiscard]] result<part_upload> begin_upload(unsigned part);

  /// @brief An upload succeeded with the part ETag @p etag (an empty or longer than @ref
  /// max_etag_bytes ETag is `http_bad_message`, handled as a failure).
  [[nodiscard]] settle_result finish_upload(part_upload const& upload, std::string_view etag);
  /// @brief An upload failed with @p ec (sticky when the upload was started by a write).
  [[nodiscard]] settle_result finish_upload(part_upload const& upload, std::error_code ec);

  // -- upload id -------------------------------------------------------------

  /// @return ready / create / wait (see @ref upload_id_status); `bad_file_descriptor` after
  /// commit or abort; the sticky error.
  [[nodiscard]] result<upload_id_status> need_upload_id(waiter_id waiter);
  /// @brief CreateMultipartUpload answered @p upload_id (empty: `http_bad_message`, handled as
  /// upload_id_failed). @return the waiters to resume (or to fail, when @c ec is set).
  [[nodiscard]] waiters_to_resume set_upload_id(std::string upload_id);
  /// @brief CreateMultipartUpload failed. @return the waiters to fail with @p ec.
  [[nodiscard]] waiters_to_resume upload_id_failed(std::error_code ec);
  /// @return the UploadId (empty when none); stable once set.
  [[nodiscard]] std::string_view upload_id() const;

  // -- flush -----------------------------------------------------------------

  /**
   * @brief Plan the commit of an object of @p object_size bytes (`io_object::size()`; the
   *        largest end the state saw is used when larger).
   * @return the plan (see @ref flush_plan); `bad_file_descriptor` after abort; the sticky
   *         error.
   */
  [[nodiscard]] result<flush_plan> plan_flush(std::uint64_t object_size, waiter_id waiter);

  /// @return the CompleteMultipartUpload body of the running multipart flush, once every part
  /// 1..N is uploaded with its ETag; `invalid_argument` otherwise.
  [[nodiscard]] result<std::string> complete_xml() const;

  /// @brief The final PUT / Complete succeeded: committed with the object ETag @p etag.
  /// @return the joined flushes to resolve with success.
  [[nodiscard]] std::vector<waiter_id> complete_succeeded(std::string etag);
  /// @brief The running flush failed (a part upload, the UploadId, the PUT or Complete): the
  /// upload stays usable. @return the joined flushes to fail with @p ec.
  [[nodiscard]] std::vector<waiter_id> complete_failed(std::error_code ec);

  // -- lifecycle ---------------------------------------------------------------

  /// @brief Abort: the state stops accepting writes and flushes (`bad_file_descriptor`).
  /// No effect after commit.
  [[nodiscard]] abort_result begin_abort();
  /// @brief Record @p ec as the sticky error (first error wins).
  void fail(std::error_code ec);
  /// @brief Called from the destructor with the UploadId when the upload was neither committed
  /// nor aborted through begin_abort (best-effort AbortMultipartUpload). Exceptions are
  /// swallowed.
  void set_abort_hook(std::function<void(std::string const& upload_id)> hook);

  [[nodiscard]] bool committed() const;
  [[nodiscard]] bool aborted() const;
  /// @return the committed object's ETag (empty before the commit).
  [[nodiscard]] std::string committed_etag() const;
  /// @return the sticky error (success when none).
  [[nodiscard]] std::error_code first_error() const;
  /// @return the largest object end any settled write reached.
  [[nodiscard]] std::uint64_t highest_end() const;
  /// @return writes classified and not yet settled plus uploads not yet finished.
  [[nodiscard]] std::size_t in_flight() const;
  /// @return a snapshot of @p part.
  [[nodiscard]] part_info info(unsigned part) const;

 private:
  /// Covered sub-ranges of one part (sorted, disjoint, non-adjacent; part-relative). The first
  /// few runs live inline, so a sequentially written part allocates nothing (gpuio W2: no
  /// allocation per part on the reactor).
  class coverage_set {
   public:
    void add(std::uint64_t begin, std::uint64_t end);
    [[nodiscard]] bool covers(std::uint64_t begin, std::uint64_t end) const noexcept;
    [[nodiscard]] std::uint64_t bytes() const noexcept;
    /// Append the uncovered ranges of [0, end) to @p out.
    void holes(std::uint64_t end, std::vector<file_range>& out) const;
    void clear() noexcept;

   private:
    static constexpr std::size_t inline_runs = 4;
    [[nodiscard]] std::span<file_range const> runs() const noexcept;
    void assign(std::span<file_range const> runs);
    std::array<file_range, inline_runs> _inline {};
    std::size_t _n_inline{0};
    std::vector<file_range> _spill;  ///< used instead of _inline once more runs are needed
    bool _spilled{false};
  };

  struct part_slot {
    std::shared_ptr<assembly_buffer> buffer;
    coverage_set coverage;
    std::uint64_t generation{0};     ///< bumped by every assembled write
    std::uint64_t uploaded_size{0};  ///< size of the last successful upload
    unsigned outstanding{0};         ///< uploads handed out and not finished
    bool reproducible{true};         ///< content == buffer + zeros
    bool dirty{false};               ///< changed since the last successful upload
    std::uint16_t etag_len{0};
    std::array<char, max_etag_bytes> etag{};
  };
  static constexpr unsigned slots_per_block = 128;  ///< one allocation per 128 part numbers
  using slot_block                          = std::array<part_slot, slots_per_block>;

  enum class phase : std::uint8_t { open, flush_waiting, flushing, committed, aborted };

  [[nodiscard]] part_slot* slot(unsigned part, bool create);
  [[nodiscard]] part_slot const* slot(unsigned part) const noexcept;
  [[nodiscard]] std::error_code closed_or_failed() const noexcept;
  [[nodiscard]] std::uint64_t part_len(unsigned part, std::uint64_t object_size) const noexcept;
  [[nodiscard]] part_phase phase_of(part_slot const& s) const noexcept;
  void make_sticky(std::error_code ec) noexcept;
  void release_buffers() noexcept;
  /// One unit of in-flight work settled: hand back the waiting flushes when none is left.
  void settle_one(std::vector<waiter_id>& flush_waiters);
  [[nodiscard]] settle_result finish(part_upload const& up,
                                     std::string_view etag,
                                     std::error_code ec);

  std::uint64_t const _part_size;
  bool const _create_new;
  upload_limits const _limits;

  mutable std::mutex _mutex;
  phase _phase{phase::open};
  std::error_code _sticky;
  std::vector<std::unique_ptr<slot_block>> _blocks;
  std::size_t _writers{0};      ///< classified writes not settled
  std::size_t _outstanding{0};  ///< uploads handed out and not finished
  std::uint64_t _highest_end{0};
  std::string _upload_id;
  std::string _id_query;
  bool _creating_id{false};
  bool _abort_sent{false};
  std::vector<waiter_id> _id_waiters;
  std::vector<waiter_id> _plan_waiters;  ///< flushes waiting for the in-flight work to drain
  std::vector<waiter_id> _join_waiters;  ///< flushes joined to the running one
  unsigned _flush_parts{0};              ///< N of the running flush
  std::uint64_t _flush_size{0};          ///< object size of the running flush
  bool _flush_multipart{false};
  std::string _committed_etag;
  std::function<void(std::string const&)> _abort_hook;
};

}  // namespace cucascade::io::rest::s3
