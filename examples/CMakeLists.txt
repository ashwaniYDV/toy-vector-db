add_executable(rag_demo rag_demo.cpp)
target_link_libraries(rag_demo PRIVATE toyvdb::toyvdb)
target_compile_options(rag_demo PRIVATE -Wall -Wextra)
