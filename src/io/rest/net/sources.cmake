# Sources of the src/io/rest/net module; included by src/io/CMakeLists.txt. The
# io_uring HTTP/1.1 transport, the DNS resolver thread and the user-space TLS
# stream.
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/dns_resolver.cpp
          ${CMAKE_CURRENT_LIST_DIR}/tls_stream.cpp
          ${CMAKE_CURRENT_LIST_DIR}/uring_transport.cpp)
