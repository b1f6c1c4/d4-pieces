#include "kernel.h"
#include "util.cuh"
#include "record.cuh"
#include "sn.cuh"

#include <cuda_pipeline.h>

// without __forceinline__ nvcc will encounter silent ICE and produce weird
// result, including CUDA_EXCEPTION_4 or CUDA_EXCEPTION_9.
template <unsigned H>
__device__ __forceinline__
static void impl(R r, frow_t fL, frow_t fR, K_PARAMS_OUT, uint8_t ea) {
    auto cfg = parse_R<H>(r, ea);
    if (fL.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fR.shape & ~cfg.empty_area) [[unlikely]] return;
    d_push(cfg.nm_cnt, cfg.ex, fL.nm0123);
    d_push(cfg.nm_cnt, cfg.ex, fR.nm0123);
    d_sn(cfg.nm_cnt, cfg.ex);
    if (!d_uniq_chk(cfg.nm_cnt, cfg.ex)) [[unlikely]] return;
    cfg.empty_area &= ~fL.shape;
    cfg.empty_area &= ~fR.shape;
    auto ocfg = assemble_R<H - 1>(cfg);
    auto out = __nv_atomic_fetch_add(n_outs, 1,
            __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
#ifndef BMARK
spin:
    auto nrc = __nv_atomic_load_n(n_reader_chunk,
            __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM);
    if (out >= (nrc + n_chunks - 1u) * CYC_CHUNK) {
        __nanosleep(1000000);
        goto spin;
    }
#endif
    ring_buffer[out % (n_chunks * CYC_CHUNK)] = ocfg; // slice
#ifndef BMARK
    if (out && out % CYC_CHUNK == 0) {
        auto tgt = out / CYC_CHUNK;
        auto src = tgt - 1;
        while (!__nv_atomic_compare_exchange_n(
                    n_writer_chunk,
                    &src, tgt, /* ignored */ true,
                    __NV_ATOMIC_RELEASE, __NV_ATOMIC_RELAXED,
                    __NV_THREAD_SCOPE_SYSTEM)) {
            if (src >= tgt) __builtin_unreachable();
            src = tgt - 1;
            __nanosleep(1000000);
        }
    }
#endif
}

// N is always coalesced
// Y is never coalesced
// X is coalesced and cached at shmem
template <unsigned H, bool Reverse>
__global__
__launch_bounds__(768, 2)
void row_search(unsigned shmem_len, K_PARAMS) {
    extern __shared__ frow32_t shmem[/* shmem_len * 3 */];

    uint32_t f0Xsz, f0Ysz;
    const frow32_t *f0X, *f0Y;
    if constexpr (!Reverse) {
        f0Xsz = f0Rsz, f0Ysz = f0Lsz;
        f0X = f0R, f0Y = f0L;
    } else {
        f0Xsz = f0Lsz, f0Ysz = f0Rsz;
        f0X = f0L, f0Y = f0R;
    }

    // some strides / sizes
    __builtin_assume(blockDim.x % warpSize == 0);
    auto wpb = static_cast<uint64_t>(blockDim.x) / warpSize;
    auto wpg = static_cast<uint64_t>(gridDim.x) * wpb;
    auto wpn = (n_cfgs + warpSize - 1) / warpSize;
    auto iterations = (wpn * f0Ysz + wpg - 1) / wpg;

    for (auto f0Xoffset = 0u; f0Xoffset < f0Xsz; f0Xoffset += shmem_len) {
        // every block, do the same: fill shmem upto shmem_len
        auto n_shmem = min(shmem_len, f0Xsz - f0Xoffset);
        auto dst = reinterpret_cast<uint32_t *>(shmem);
        auto src = reinterpret_cast<const uint32_t *>(f0X + f0Xoffset);
        for (auto i = threadIdx.x; i < 3 * n_shmem; i += blockDim.x) {
            __pipeline_memcpy_async(dst + i, src + i, 4, 0);
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // each warp, inspect the (warpIdx + k * wpg)-th warp
        auto warpIdx = threadIdx.x / warpSize + blockIdx.x * wpb;
        for (auto k = 0ull; k < iterations; k++) {
            auto w = warpIdx + k * wpg;
            if (w / wpn >= f0Ysz)
                break;
            frow32_t fY32 = f0Y[w / wpn];
            frow_t fY = fY32;
            if (threadIdx.x % warpSize + w % wpn >= n_cfgs) [[unlikely]]
                continue;

            auto r = cfgs[threadIdx.x % warpSize + w % wpn];
            for (auto i = 0ull; i < n_shmem; i++) {
                frow_t fX = shmem[i];
                if (fX.shape & fY.shape) [[unlikely]]
                    continue;
                impl<H>(r, Reverse ? fX : fY, Reverse ? fY : fX,
                    ring_buffer, n_outs, n_chunks,
                    n_reader_chunk, n_writer_chunk, ea);
            }
        }
    }
}

template <unsigned H, int>
__global__
void simple_row_search(unsigned, K_PARAMS) {
    auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_cfgs * f0Lsz * f0Rsz) [[unlikely]] return;
    auto r = cfgs[idx / f0Rsz / f0Lsz];
    frow_t fL = f0L[idx / f0Rsz % f0Lsz];
    frow_t fR = f0R[idx % f0Rsz];
    if (fL.shape & fR.shape) [[unlikely]] return;
    impl<H>(r, fL, fR, ring_buffer, n_outs, n_chunks,
            n_reader_chunk, n_writer_chunk, ea);
}

template __global__ void simple_row_search<8, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<7, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<6, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<5, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<4, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<3, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<2, 0>(unsigned, K_PARAMS);
template __global__ void simple_row_search<1, 0>(unsigned, K_PARAMS);
template __global__ void row_search<8, true>(unsigned, K_PARAMS);
template __global__ void row_search<7, true>(unsigned, K_PARAMS);
template __global__ void row_search<6, true>(unsigned, K_PARAMS);
template __global__ void row_search<5, true>(unsigned, K_PARAMS);
template __global__ void row_search<4, true>(unsigned, K_PARAMS);
template __global__ void row_search<3, true>(unsigned, K_PARAMS);
template __global__ void row_search<2, true>(unsigned, K_PARAMS);
template __global__ void row_search<1, true>(unsigned, K_PARAMS);
template __global__ void row_search<8, false>(unsigned, K_PARAMS);
template __global__ void row_search<7, false>(unsigned, K_PARAMS);
template __global__ void row_search<6, false>(unsigned, K_PARAMS);
template __global__ void row_search<5, false>(unsigned, K_PARAMS);
template __global__ void row_search<4, false>(unsigned, K_PARAMS);
template __global__ void row_search<3, false>(unsigned, K_PARAMS);
template __global__ void row_search<2, false>(unsigned, K_PARAMS);
template __global__ void row_search<1, false>(unsigned, K_PARAMS);
