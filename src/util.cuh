#pragma once

#include <cuda.h>
#include <format>
#include <iostream>
#include <stdexcept>

/**
 * 128 resident grids / device (Concurrent Kernel Execution)
 * 2147483647*65535*65535 blocks / grid
 * 1024*1024*64 <= 1024 threads / block
 * 32 threads / warp
 * 16 blocks / SM
 * 48 threads / warp
 * 1536 threads / SM
 * 65536 regs / SM
 * 255 regs / threads
 * 64KiB constant memory (8KiB cache)
 */

#define CYC_CHUNK (512ull * 1048576ull / sizeof(RX))

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error{
            std::format("CUDA: {}: {} @ {}:{}\n",
                    cudaGetErrorName(code), cudaGetErrorString(code),
                    file, line) };
    }
}

inline void chk_impl(CUresult code, const char *file, int line) {
    const char *pn = "???", *ps = "???";
    cuGetErrorName(code, &pn);
    cuGetErrorString(code, &ps);
    if (code != CUDA_SUCCESS) {
        throw std::runtime_error{
            std::format("CUDA Driver: {}: {} @ {}:{}\n", pn, ps, file, line) };
    }
}
