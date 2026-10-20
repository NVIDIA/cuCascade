# Tests of the test/io/rest/http module; included by test/CMakeLists.txt into
# cucascade_io_tests. gpuio's alloc_counter.cpp / perf_alloc_test.cpp are not
# ported (they replace the global operator new of the whole binary).
target_sources(
  cucascade_io_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/http_test_support.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_beast_vectors.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_classify.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_content_range.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_iovec_sink.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_request_head.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_s3_responses.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_http_split_fuzz.cpp)
