#pragma once

#include "kernel.h"
#include "util.cuh"
#include "record.cuh"
#include "sn.cuh"

#include <cuda_pipeline.h>

// without __forceinline__ nvcc will encounter silent ICE and produce weird
// result, including CUDA_EXCEPTION_4 or CUDA_EXCEPTION_9.
template <unsigned H>
__device__ __forceinline__
static void impl(RCfg cfg, frow_t fL, frow_t fR, K_PARAMS_OUT) {
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

__device__ __forceinline__
void cache_frow(frow32_t *dst, const frow32_t *src, unsigned sz) {
    auto d = reinterpret_cast<uint32_t *>(dst);
    auto s = reinterpret_cast<const uint32_t *>(src);
    for (auto i = threadIdx.x; i < sizeof(frow32_t) / sizeof(uint32_t) * sz; i += blockDim.x) {
        __pipeline_memcpy_async(d + i, s + i, 4, 0);
        // d[i] = s[i];
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
}

// N is always coalesced
// L is tiled/coalesced and cached at shmem
// R is tiled/coalesced and cached at shmem
//
// Reverse == false:
// the (j + i*nR + k*nL*nR)-th block shall process:
//   f0L[i*Ltile, (i+1)*Ltile) *
//   f0R[j*Rtile, (j+1)*Rtile) *
//   cfgs[k*blockDim.x, (k+1)*blockDim.x)
//
// Reverse == true:
// the (i + j*nL + k*nL*nR)-th block shall process:
//   f0L[i*Ltile, (i+1)*Ltile) *
//   f0R[j*Rtile, (j+1)*Rtile) *
//   cfgs[k*blockDim.x, (k+1)*blockDim.x)
//
template <unsigned H, int W, bool Reverse>
__global__
__launch_bounds__(W, 1536 / W)
void tiled_row_search(unsigned Ltile, unsigned Rtile, K_PARAMS) {

    auto tpb = static_cast<uint64_t>(blockDim.x);
    auto nC = (n_cfgs + tpb - 1) / tpb;
    auto nL = (f0Lsz + Ltile - 1) / Ltile;
    auto nR = (f0Rsz + Rtile - 1) / Rtile;
    if (blockIdx.x >= nC * nL * nR) // this block shouldn't even exist!
        return;

#ifdef BMARK
    long long perf_lr{}, perf_n{}, perf_tile{}, perf_comp{};
#define BEFORE(X) perf_ ## X -= clock64()
#define AFTER(X)  perf_ ## X += clock64()
#else
#define BEFORE(X)
#define AFTER(X)
#endif

    unsigned i, j, k;
    if constexpr (!Reverse) {
        j = blockIdx.x % nR;
        i = blockIdx.x / nR % nL;
        k = blockIdx.x / nR / nL;
    } else {
        i = blockIdx.x % nL;
        j = blockIdx.x / nL % nR;
        k = blockIdx.x / nL / nR;
    }
    auto Lsz = min(Ltile, f0Lsz - i * Ltile);
    auto Rsz = min(Rtile, f0Rsz - j * Rtile);

    extern __shared__ frow32_t shmem[/* Ltile + Rtile */];
    auto *Lcache = shmem;
    auto *Rcache = shmem + Ltile;

    BEFORE(lr);
    cache_frow(Lcache, f0L.data32 + i * Ltile, Lsz);
    cache_frow(Rcache, f0R.data32 + j * Rtile, Rsz);
    __syncthreads();
    AFTER(lr);

    auto idx = threadIdx.x + k * tpb;
    if (idx < n_cfgs) {
        BEFORE(n);
        auto cfg = parse_R<H>(cfgs[idx], ea);
        AFTER(n);

        BEFORE(tile);
        // profiling showed that ALWAYS put (actual) f0R as outer loop in a tile
        for (auto r = 0u; r < Rsz; r++) {
            frow_t fR = Rcache[r];
            if (fR.shape & ~cfg.empty_area) [[unlikely]] continue;

            for (auto l = 0u; l < Lsz; l++) {
                frow_t fL = Lcache[l];
                if (fL.shape & ~cfg.empty_area || fR.shape & fL.shape) [[likely]]
                    continue;

                BEFORE(comp);
                impl<H>(cfg, fL, fR,
                        ring_buffer, n_outs, n_chunks,
                        n_reader_chunk, n_writer_chunk);
                AFTER(comp);
            }
        }
        AFTER(tile);
    }

#ifdef BMARK
    __nv_atomic_fetch_add(&perf[0], perf_lr,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_add(&perf[1], perf_n,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_add(&perf[2], perf_tile,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_add(&perf[3], perf_comp,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#endif
}

template <unsigned H, int Coalesced>
__global__
__launch_bounds__(768, 2)
void linear_row_search(unsigned, unsigned, K_PARAMS) {
    static_assert(-1 <= Coalesced && Coalesced <= +1,
            "Coalesced must be -1, 0, +1");

#ifdef BMARK
    long long perf_lr{}, perf_n{}, perf_tile{}, perf_comp{};
#define BEFORE(X) perf_ ## X -= clock64()
#define AFTER(X)  perf_ ## X += clock64()
#else
#define BEFORE(X)
#define AFTER(X)
#endif

    auto tpb = static_cast<uint64_t>(blockDim.x);
    auto idx = threadIdx.x + tpb * blockIdx.x;

    frow_t fL, fR;
    if (idx >= n_cfgs * f0Lsz * f0Rsz) [[unlikely]] return;
    BEFORE(n);
    auto r = cfgs[idx / f0Rsz / f0Lsz];
    AFTER(n);

    if constexpr (Coalesced == 0) { // N - L - R
        BEFORE(lr);
        fL = f0L.data32[idx / f0Rsz % f0Lsz];
        fR = f0R.data32[idx % f0Rsz];
        AFTER(lr);
    } else if constexpr (Coalesced == +1) { // N - L - R
        auto i = idx % f0Rsz;
        BEFORE(lr);
        fL = f0L.data32[idx / f0Rsz % f0Lsz];
        fR = frow32_t{ f0R.dataL[i], f0R.dataH[i], f0R.data0123[i] };
        AFTER(lr);
    } else if constexpr (Coalesced == -1) { // N - R - L
        auto i = idx % f0Lsz;
        BEFORE(lr);
        fR = f0R.data32[idx / f0Lsz % f0Rsz];
        fL = frow32_t{ f0L.dataL[i], f0L.dataH[i], f0L.data0123[i] };
        AFTER(lr);
    }

    BEFORE(tile);
    auto cfg = parse_R<H>(r, ea);
    if (fL.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fR.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fL.shape & fR.shape) [[unlikely]] return;
    BEFORE(comp);
    impl<H>(cfg, fL, fR, ring_buffer, n_outs, n_chunks,
            n_reader_chunk, n_writer_chunk);
    AFTER(comp);
    AFTER(tile);

#ifdef BMARK
    __nv_atomic_fetch_add(&perf[0], perf_lr,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_add(&perf[1], perf_n,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_add(&perf[2], perf_tile,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_add(&perf[3], perf_comp,
            __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#endif
}
