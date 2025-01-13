cmake_minimum_required(VERSION 3.21)
project(pieces)

set(CMAKE_CXX_STANDARD 23)

add_compile_options(-fuse-ld=lld)
add_compile_options(-Wall -Wextra -Werror)
add_compile_options(-Wno-parentheses)
add_compile_options(-Wno-missing-field-initializers)
add_compile_options(-Wno-unused-function)
add_compile_options(-Wno-unused-label)
add_compile_options(-Wno-unused-parameter)
add_compile_options(-Wno-unknown-pragmas)
add_compile_options(-Wno-sign-compare)
add_compile_options(-Wno-unused-command-line-argument)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(piece wasm.cpp Piece.cpp Shape.cpp)
target_link_libraries(piece PRIVATE embind)
