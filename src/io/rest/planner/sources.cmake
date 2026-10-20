# Sources of the src/io/rest/planner module; included by src/io/CMakeLists.txt.
# Remote planner (gpuio src/planner): segment sizing, coalescing, part
# alignment, fairness, admission, retries, deadlines, estimator / controller,
# EC2 link table. Pure: no I/O.
target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/coalesce.cpp
          ${CMAKE_CURRENT_LIST_DIR}/remote_admission.cpp
          ${CMAKE_CURRENT_LIST_DIR}/remote_estimator.cpp
          ${CMAKE_CURRENT_LIST_DIR}/remote_fairness.cpp
          ${CMAKE_CURRENT_LIST_DIR}/remote_link_table.cpp
          ${CMAKE_CURRENT_LIST_DIR}/remote_retry.cpp
          ${CMAKE_CURRENT_LIST_DIR}/remote_segment.cpp)
