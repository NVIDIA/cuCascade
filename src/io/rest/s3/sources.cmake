# Sources of the src/io/rest/s3 module (S3 multipart protocol and upload state
# machine); included by src/io/CMakeLists.txt. sigv4.cpp, sigv4_authorizer.cpp
# and list_parser.cpp are still listed there directly.
target_sources(cucascade_io_objects
               PRIVATE ${CMAKE_CURRENT_LIST_DIR}/multipart.cpp)
