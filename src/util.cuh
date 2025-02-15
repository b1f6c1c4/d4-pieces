#pragma once

#include <cuda.h>
#include <format>
#include <iostream>
#include "util.hpp"

/**
 * 128 resident grids / device (Concurrent Kernel Execution)
 * 2147483647*65535*65535 blocks / grid
 * 1024*1024*64 <= 1024 threads / block
 * 32 threads / warp
 * 16 blocks / SM
 * 48 warps / SM
 * 1536 threads / SM
 * 65536 regs / SM
 * 255 regs / threads
 * 64KiB constant memory (8KiB cache)
 */

#define CYC_CHUNK (512ull * 1048576ull / sizeof(RX))

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        THROW("CUDA: {}: {} @ {}:{}\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
    }
}

inline void chk_impl(CUresult code, const char *file, int line) {
    const char *pn = "???", *ps = "???";
    cuGetErrorName(code, &pn);
    cuGetErrorString(code, &ps);
    if (code != CUDA_SUCCESS) {
        THROW("CUDA Driver: {}: {} @ {}:{}\n", pn, ps, file, line);
    }
}

#ifdef CURAND_H_
inline void chk_impl(curandStatus_t code, const char *file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        THROW("curand: {} @ {}:{}\n", (int)code, file, line);
    }
}
#endif
