# Tests of the test/io/tls module; included by test/CMakeLists.txt into
# cucascade_io_tests.
target_sources(
  cucascade_io_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_tls_record_layer.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_tls_provider.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_tls_session.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_ktls.cpp)
