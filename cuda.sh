#!/usr/bin/bash

set -euxo pipefail

cd "$(dirname "$(realpath "$0")")"

OPT=-O3
G="-lineinfo --split-compile=0"
LTO=TRUE
ELF=

ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        -B)
            shift
            rm -rf build-x86
            ;;
        -G0)
            shift
            OPT=-O0
            G=-G
            LTO=FALSE
            ELF=-ltbb
            ;;
        -G3)
            shift
            G=-G
            LTO=FALSE
            ;;
        -fno-lto)
            shift
            LTO=FALSE
            ;;
        --)
            shift
            break
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

export CC=gcc-13
export CXX=g++-13
cmake -S . -B build-x86 \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION="$LTO" \
        -DCMAKE_CXX_FLAGS="-g $OPT -march=native" \
        -DCMAKE_EXE_LINKER_FLAGS="$ELF" \
        -DCMAKE_CUDA_FLAGS="-g --keep-dir \${CMAKE_BINARY_DIR}/keep.dir/ $OPT $G --extra-device-vectorization -restrict ${ARGS[*]} -gen-opt-lto" \
        -G Ninja
exec cmake --build build-x86 -- "$@"
