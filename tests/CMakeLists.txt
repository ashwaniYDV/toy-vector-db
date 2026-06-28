include(FetchContent)

# Prefer a system/brew GoogleTest if present; otherwise fetch a pinned release.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  FIND_PACKAGE_ARGS NAMES GTest
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(toyvdb_tests
  test_distance.cpp
  test_vector_store.cpp
  test_metadata_filter.cpp
  test_flat_index.cpp
  test_hnsw.cpp
  test_storage_seam.cpp
  test_persistence.cpp
  test_parallel.cpp
  test_rag.cpp
  test_edgecases.cpp
)
target_link_libraries(toyvdb_tests PRIVATE toyvdb::toyvdb GTest::gtest_main)
target_compile_options(toyvdb_tests PRIVATE -Wall -Wextra)

include(GoogleTest)
gtest_discover_tests(toyvdb_tests)
