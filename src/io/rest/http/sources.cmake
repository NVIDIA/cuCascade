# Sources of the src/io/rest/http module; included by src/io/CMakeLists.txt.
# HTTP/1.1 codec ported from gpuio (Beast-derived parser, classify): pure CPU,
# no I/O, no allocation.
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/chunked_decoder.cpp
          ${CMAKE_CURRENT_LIST_DIR}/classify.cpp
          ${CMAKE_CURRENT_LIST_DIR}/content_range.cpp
          ${CMAKE_CURRENT_LIST_DIR}/iovec_sink.cpp
          ${CMAKE_CURRENT_LIST_DIR}/request_head.cpp
          ${CMAKE_CURRENT_LIST_DIR}/response_parser.cpp)
