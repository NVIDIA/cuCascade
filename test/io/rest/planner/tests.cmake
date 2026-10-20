# Tests of the test/io/rest/planner module (and of src/io/rest/cloud); included
# by test/CMakeLists.txt into cucascade_io_tests. gpuio's
# remote_alloc_counter.cpp / remote_alloc_test.cpp (and the "allocates nothing"
# controller case) are not ported: they replace the global operator new of the
# whole binary.
target_sources(
  cucascade_io_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_admission.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_coalesce.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_controller.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_imds.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_remote_cut.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_remote_estimator.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_remote_fairness.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_remote_retry.cpp
          ${CMAKE_CURRENT_LIST_DIR}/test_remote_segment.cpp)
