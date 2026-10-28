# Sources of the src/io/rest/net module; included by src/io/CMakeLists.txt. The
# io_uring host reactor (uring_loop), DNS, the connection pool's non-template
# parts and the user-space TLS shell (docs/io/rest-uring-design.md sections 3,
# 4).
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dns_resolver.cpp
          ${CMAKE_CURRENT_LIST_DIR}/pool_support.cpp
          ${CMAKE_CURRENT_LIST_DIR}/tls_stream.cpp
          ${CMAKE_CURRENT_LIST_DIR}/uring_loop.cpp)
