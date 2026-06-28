cmake_minimum_required(VERSION 3.24)
project(toy_vector_db
        VERSION 0.1.0
        DESCRIPTION "In-memory vector database + RAG engine in modern C++20 (educational)"
        LANGUAGES CXX)

# ---- Standard -------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

export(PACKAGE ${PROJECT_NAME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TOYVDB_BUILD_TESTS "Build unit tests" ON)
option(TOYVDB_BUILD_BENCH "Build benchmark harness" ON)
option(TOYVDB_BUILD_EXAMPLES "Build example executables" ON)
option(TOYVDB_BUILD_SERVER "Build the REST API server" ON)
option(TOYVDB_WERROR "Treat warnings as errors for project targets" ON)

# Warning flags applied only to OUR targets (not third-party like GoogleTest).
set(TOYVDB_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
if(TOYVDB_WERROR)
  list(APPEND TOYVDB_WARNINGS -Werror)
endif()

# ---- Core library ---------------------------------------------------------
add_library(toyvdb STATIC
  src/vector_store.cpp
  src/flat_index.cpp
  src/hnsw_index.cpp
  src/op_codec.cpp
  src/log_store.cpp
  src/blob_store.cpp
  src/engine.cpp
  src/rag.cpp
)
add_library(toyvdb::toyvdb ALIAS toyvdb)
target_include_directories(toyvdb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(toyvdb PUBLIC cxx_std_20)
target_compile_options(toyvdb PRIVATE ${TOYVDB_WARNINGS})

# ---- Tests ----------------------------------------------------------------
if(TOYVDB_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Benchmarks -----------------------------------------------------------
if(TOYVDB_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# ---- Examples -------------------------------------------------------------
if(TOYVDB_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---- REST server ----------------------------------------------------------
if(TOYVDB_BUILD_SERVER)
  add_subdirectory(server)
endif()

message(STATUS "toy-vector-db ${PROJECT_VERSION} | compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} | build: ${CMAKE_BUILD_TYPE}")
