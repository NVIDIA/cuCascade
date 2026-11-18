# Sources of the src/io/tls module; included by src/io/CMakeLists.txt.
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tls.cpp
          ${CMAKE_CURRENT_LIST_DIR}/key_schedule.cpp
          ${CMAKE_CURRENT_LIST_DIR}/evp_aead.cpp
          ${CMAKE_CURRENT_LIST_DIR}/record_opener.cpp
          ${CMAKE_CURRENT_LIST_DIR}/ktls.cpp
          ${CMAKE_CURRENT_LIST_DIR}/openssl_context.cpp
          ${CMAKE_CURRENT_LIST_DIR}/openssl_session.cpp
          ${CMAKE_CURRENT_LIST_DIR}/openssl_session_cache.cpp
          ${CMAKE_CURRENT_LIST_DIR}/openssl_trust_store.cpp)
