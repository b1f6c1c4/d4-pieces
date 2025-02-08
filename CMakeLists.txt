cmake_minimum_required(VERSION 3.21)
if(EMSCRIPTEN)
    project(pieces)
else()
    project(pieces LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 23)

add_compile_options(-fuse-ld=lld)
add_compile_options(-Wall -Wextra)
add_compile_options(-Werror)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:all-warnings>)
add_compile_options(-Wno-parentheses)
add_compile_options(-Wno-missing-field-initializers)
add_compile_options(-Wno-unused-function)
add_compile_options(-Wno-unused-label)
add_compile_options(-Wno-unused-parameter)
add_compile_options(-Wno-unknown-pragmas)
add_compile_options(-Wno-sign-compare)
add_compile_options(-Wno-unused-command-line-argument)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

if(EMSCRIPTEN)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_executable(pieces src/wasm.cpp src/Piece.cpp src/Shape.cpp src/known.cpp src/Group.cpp)
    target_link_options(pieces PRIVATE -sWASM_BIGINT)
    target_link_options(pieces PRIVATE -sASSERTIONS)
    target_link_options(pieces PRIVATE -sALLOW_MEMORY_GROWTH=1)
    target_link_options(pieces PRIVATE -sENVIRONMENT=web)
    target_link_options(pieces PRIVATE -sMODULARIZE=1)
    target_link_options(pieces PRIVATE -sEXPORT_NAME=Pieces)
    target_link_libraries(pieces PRIVATE embind)

    add_executable(profile src/main.cpp src/Piece.cpp src/Shape.cpp src/Group.cpp)
    target_compile_options(profile PRIVATE -g)
    target_link_options(profile PRIVATE -sWASM_BIGINT)
    target_link_options(profile PRIVATE --profiling)
    target_link_options(profile PRIVATE -sWASM=1)
    target_link_options(profile PRIVATE -sSTANDALONE_WASM=1)
    target_link_options(profile PRIVATE -sENVIRONMENT=shell)
    target_link_options(profile PRIVATE -sMODULARIZE=0)
else()
    # set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_executable(search
        src/search.cpp
        src/searcher.cpp
        src/searcher_cuda.cu
        src/frow.cu
        src/frow.cpp
        src/kernel.cu
        src/device.cu
        src/record.cu
        src/region.cu
        src/sorter.cpp
        src/naming.cpp
        src/Piece.cpp src/Shape.cpp src/Group.cpp
        src/known1.cpp src/known2.cpp)
    target_compile_definitions(search PRIVATE SORTER=1 SORTER_N=127)
    target_link_libraries(search PRIVATE boost_thread)
    target_link_libraries(search PRIVATE tbb)
    target_link_libraries(search PRIVATE mimalloc)
    target_link_libraries(search PRIVATE cuda)
    target_compile_options(search PRIVATE -O3 -march=native)
    target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
    target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>)
    target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(search PROPERTIES CUDA_ARCHITECTURES "86")

    add_executable(enum src/enum.cpp src/Shape.cpp src/Group.cpp)
    target_compile_options(enum PRIVATE -g -O3)

    add_executable(kernel
        bmark/kernel.cu
        src/frow.cpp
        src/frow.cu
        src/kernel.cu
        src/kernel_impl.cu
        src/known1.cpp
        src/known2.cpp
        src/record.cu
        src/naming.cpp
        src/Shape.cpp)
    target_compile_definitions(kernel PUBLIC BMARK)
    target_compile_options(kernel PUBLIC -g -O3)
    target_link_libraries(kernel PRIVATE cuda)
    target_link_libraries(kernel PRIVATE curand)
    target_compile_options(kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>)
    set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(kernel PROPERTIES CUDA_ARCHITECTURES "86")

    add_library(sorter OBJECT bmark/sorter.cpp)
    target_compile_definitions(sorter INTERFACE BMARK)
    target_compile_options(sorter PUBLIC -g -O3)
    target_link_libraries(sorter PUBLIC boost_thread)
    target_link_libraries(sorter PUBLIC mimalloc)
    link_libraries(sorter)
    foreach(N RANGE 4)
        add_executable(sorter-${N} src/sorter.cpp)
        target_compile_definitions(sorter-${N} PRIVATE SORTER=${N} SORTER_NPARF)
        add_executable(sorter-${N}p src/sorter.cpp)
        target_compile_definitions(sorter-${N}p PRIVATE SORTER=${N})
        add_executable(sorter-${N}x src/sorter.cpp)
        target_compile_definitions(sorter-${N}x PRIVATE SORTER=${N} SORTER_NPARF SORTER_256)
        add_executable(sorter-${N}px src/sorter.cpp)
        target_compile_definitions(sorter-${N}px PRIVATE SORTER=${N} SORTER_256)
    endforeach()
endif()
