#pragma once

#include <cstdint>
#ifdef __CUDA_ARCH__
#include <cuda.h>
#endif

// g_nme and g_sym must be set
// required before calling Searcher::Searcher
void compute_frow_on_cpu();
void transfer_frow_to_gpu();
void show_gpu_devices();

struct frow32_t;
struct frow_t {
    uint64_t shape;
    union {
        uint32_t nm0123;
        uint8_t nm[4];
    };

    operator frow32_t() const;
};

struct frow32_t {
    uint32_t shapeL;
    uint32_t shapeH;
    uint32_t nm0123;

    operator frow_t() const;
};

struct frow_info_t {
    frow_t *data;
    frow32_t *data32;
    uint32_t sz[6];
};

// defined in frow.cu
extern int n_devices;
extern frow_info_t h_frowInfoL[16], h_frowInfoR[16]; // defined in frow.cpp
extern frow32_t *d_frowDataL[128][16], *d_frowDataR[128][16];
#ifdef __CUDA_ARCH__
extern CUcontext cuda_ctxs[128];
#endif

#ifdef __CUDA_ARCH__
__host__ __device__ __forceinline__
#else
inline
#endif
frow_t::operator frow32_t() const {
    return frow32_t{ (uint32_t)shape, (uint32_t)(shape >> 32), nm0123 };
}

#ifdef __CUDA_ARCH__
__host__ __device__ __forceinline__
#else
inline
#endif
frow32_t::operator frow_t() const {
    return frow_t{ ((uint64_t)shapeH) << 32 | shapeL, nm0123 };
}
