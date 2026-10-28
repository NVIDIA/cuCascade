# Tests of the src/io/rest/s3 module (SigV4 write signing, multipart protocol,
# upload state machine); included by test/CMakeLists.txt into
# cucascade_io_tests. test_sigv4.cpp, test_sigv4_authorizer.cpp and
# test_static_credentials.cpp are still listed there directly.
target_sources(
  cucascade_io_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_sigv4_write.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_multipart.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_upload_state.cpp)
