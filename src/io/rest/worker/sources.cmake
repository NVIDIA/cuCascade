# Sources of the REST worker (docs/io/rest-uring-design.md section 3); included
# by src/io/CMakeLists.txt.
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_tracker.cpp
          ${CMAKE_CURRENT_LIST_DIR}/intake.cpp
          ${CMAKE_CURRENT_LIST_DIR}/services.cpp)
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/engine.cpp
          ${CMAKE_CURRENT_LIST_DIR}/engine_write.cpp
          ${CMAKE_CURRENT_LIST_DIR}/upload.cpp)
