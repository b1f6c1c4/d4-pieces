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

if(EMSCRIPTEN)
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
    add_executable(search
        src/search.cpp
        src/naming.cpp
        src/Piece.cpp src/Shape.cpp src/Group.cpp
        src/known1.cpp src/known2.cpp)
    target_link_libraries(search PRIVATE boost_thread)
    target_link_libraries(search PRIVATE mimalloc)
    target_compile_options(search PRIVATE -g -O3)
    target_compile_options(search PRIVATE
        -march=znver2
        -mmmx
        -mpopcnt
        -msse
        -msse2
        -msse3
        -mssse3
        -msse4.1
        -msse4.2
        -mavx
        -mavx2
        -msse4a
        -mno-fma4
        -mno-xop
        -mfma
        -mno-avx512f
        -mbmi
        -mbmi2
        -maes
        -mpclmul
        -mno-avx512vl
        -mno-avx512bw
        -mno-avx512dq
        -mno-avx512cd
        -mno-avx512vbmi
        -mno-avx512ifma
        -mno-avx512vpopcntdq
        -mno-avx512vbmi2
        -mno-gfni
        -mno-vpclmulqdq
        -mno-avx512vnni
        -mno-avx512bitalg
        -mno-avx512bf16
        -mno-avx512vp2intersect
        -mno-3dnow
        -madx
        -mabm
        -mno-cldemote
        -mclflushopt
        -mclwb
        -mclzero
        -mcx16
        -mno-enqcmd
        -mf16c
        -mfsgsbase
        -mfxsr
        -mno-hle
        -msahf
        -mno-lwp
        -mlzcnt
        -mmovbe
        -mno-movdir64b
        -mno-movdiri
        -mmwaitx
        -mno-pconfig
        -mno-pku
        -mprfchw
        -mno-ptwrite
        -mrdpid
        -mrdrnd
        -mrdseed
        -mno-rtm
        -mno-serialize
        -mno-sgx
        -msha
        -mno-shstk
        -mno-tbm
        -mno-tsxldtrk
        -mno-vaes
        -mno-waitpkg
        -mwbnoinvd
        -mxsave
        -mxsavec
        -mxsaveopt
        -mxsaves
        -mno-amx-tile
        -mno-amx-int8
        -mno-amx-bf16
        -mno-uintr
        -mno-hreset
        -mno-kl
        -mno-widekl
        -mno-avxvnni
        -mno-avx512fp16
        -mno-avxifma
        -mno-avxvnniint8
        -mno-avxneconvert
        -mno-cmpccxadd
        -mno-amx-fp16
        -mno-prefetchi
        -mno-raoint
        -mno-amx-complex
        -mno-avxvnniint16
        -mno-sm3
        -mno-sha512
        -mno-sm4
        -mno-apxf
        -mno-usermsr
        # --param l1-cache-size=32
        # --param l1-cache-line-size=64
        # --param l2-cache-size=512
        -mtune=znver2)

    add_executable(enum src/enum.cpp src/Shape.cpp src/Group.cpp)
    target_compile_options(enum PRIVATE -g -O3)
endif()
