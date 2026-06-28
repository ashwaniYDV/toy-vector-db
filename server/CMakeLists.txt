include(FetchContent)

# Header-only HTTP server + JSON. Use the packaged JSON release (tiny) instead of a
# full git clone (the json repo bundles large test data). Prefer system installs if present.
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.16.3
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS NAMES httplib
)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  FIND_PACKAGE_ARGS NAMES nlohmann_json
)
FetchContent_MakeAvailable(httplib nlohmann_json)

add_executable(toyvdb_server server.cpp)
target_link_libraries(toyvdb_server PRIVATE
  toyvdb::toyvdb httplib::httplib nlohmann_json::nlohmann_json)
target_compile_options(toyvdb_server PRIVATE -Wall -Wextra)

# Absolute path to the static UI directory, baked in so the server finds it
# regardless of the working directory it is launched from.
target_compile_definitions(toyvdb_server PRIVATE
  TOYVDB_STATIC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/static")
