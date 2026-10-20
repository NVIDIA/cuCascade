# Sources of the src/io/rest/cloud module; included by src/io/CMakeLists.txt.
# EC2 detection (DMI) and the IMDSv2 link seed of the remote controller (gpuio
# src/auth, link seed only): blocking, helper thread only, never on a reactor.
target_sources(
  cucascade_io_objects PRIVATE ${CMAKE_CURRENT_LIST_DIR}/http_client.cpp
                               ${CMAKE_CURRENT_LIST_DIR}/imds.cpp)
