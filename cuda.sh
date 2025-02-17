#!/usr/bin/bash

set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

CUDA_OPT=-O3
CXX_OPT=-O3
G=-lineinfo
LTO=FALSE
ELF=

ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        -B)
            shift
            rm -rf build-x86
            ;;
        -O0)
            shift
            CXX_OPT=-O0
            ELF=-ltbb
            ;;
        -G0)
            shift
            CUDA_OPT=-O0
            G=-G
            LTO=FALSE
            CXX_OPT=-O0
            ELF=-ltbb
            ;;
        -G3)
            shift
            G=-G
            LTO=FALSE
            ;;
        -flto)
            shift
            LTO=TRUE
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

set -x
cmake -S . -B build-x86 \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION="$LTO" \
        -DCMAKE_CXX_FLAGS="-g $CXX_OPT -march=native" \
        -DCMAKE_EXE_LINKER_FLAGS="$ELF" \
        -DCMAKE_CUDA_FLAGS="-g $CUDA_OPT $G --restrict --extra-device-vectorization --device-entity-has-hidden-visibility=true --static-global-template-stub=true ${ARGS[*]}" \
        -G Ninja
exec cmake --build build-x86 -- "$@"
