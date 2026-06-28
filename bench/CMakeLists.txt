add_executable(recall_harness recall_harness.cpp)
target_link_libraries(recall_harness PRIVATE toyvdb::toyvdb)
target_compile_options(recall_harness PRIVATE -Wall -Wextra)
