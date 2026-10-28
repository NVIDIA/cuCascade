# Tests of the test/io/rest/net module; included by test/CMakeLists.txt into
# cucascade_io_tests.
target_sources(
  cucascade_io_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/net_test_support.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_net_dns.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_net_pool.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_net_streams.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_net_tls.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_net_uring_loop.cpp)
