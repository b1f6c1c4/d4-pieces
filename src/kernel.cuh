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
void cache_frow(frow32_t *dst, const frow32_t *src, size_t sz) {
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
template <unsigned H, int W, bool Reverse>
__global__
__launch_bounds__(W, 1536 / W)
void tiled_row_search(unsigned Ltile, unsigned Rtile, K_PARAMS) {

    auto tpb = static_cast<uint64_t>(blockDim.x);
    if (tpb * blockIdx.x >= n_cfgs) // this block shouldn't even exist!
        return;

#ifdef BMARK
    long long perf_lr{}, perf_n{}, perf_tile{}, perf_comp{};
#define BEFORE(X) perf_ ## X -= clock64()
#define AFTER(X)  perf_ ## X += clock64()
#else
#define BEFORE(X)
#define AFTER(X)
#endif

#define CACHE(X) do { \
        BEFORE(lr); \
        cache_frow(X ## cache, f0 ## X.data32 + X ## offset, X ## sz); \
        AFTER(lr); \
    } while (false)

    frow_info_d f0Y, f0X;
    uint32_t f0Ysz, f0Xsz;
    if constexpr (Reverse) {
        f0Y = f0R, f0Ysz = f0Rsz;
        f0X = f0L, f0Xsz = f0Lsz;
    } else {
        f0Y = f0L, f0Ysz = f0Lsz;
        f0X = f0R, f0Xsz = f0Rsz;
    }

    Ltile = min(Ltile, f0Lsz);
    Rtile = min(Rtile, f0Rsz);

    extern __shared__ frow32_t shmem[/* Ltile + Rtile */];
    auto *Lcache = shmem;
    auto *Rcache = shmem + Ltile;

    frow32_t *Ycache;
    frow32_t *Xcache;
    unsigned Ytile, Xtile;
    if constexpr (Reverse) {
        Xcache = Lcache, Ycache = Rcache;
        Xtile = Ltile, Ytile = Rtile;
    } else {
        Ycache = Lcache, Xcache = Rcache;
        Ytile = Ltile, Xtile = Rtile;
    }

    if (f0Xsz <= Xtile) {
        auto Xoffset = 0u;
        auto Xsz = f0Xsz;
        CACHE(X);
    }

    for (auto Yoffset = 0u; Yoffset < f0Ysz; Yoffset += Ytile) {
        auto Ysz = min(Ytile, f0Ysz - Yoffset);
        CACHE(Y);
        if (f0Xsz < Xtile)
            __syncthreads();

        for (auto Xoffset = 0u; Xoffset < f0Xsz; Xoffset += Xtile) {
            auto Xsz = min(Xtile, f0Xsz - Xoffset);
            if (f0Xsz > Xtile) {
                CACHE(X);
                __syncthreads();
            }

            auto idx = threadIdx.x + tpb * blockIdx.x;
            if (idx >= n_cfgs) {
                // this thread still need to exist,
                // otherwise block-level shmem will go wrong
                continue;
            }

            BEFORE(n);
            auto cfg = parse_R<H>(cfgs[idx], ea);
            AFTER(n);

            BEFORE(tile);
            const auto &Lsz = Reverse ? Xsz : Ysz;
            const auto &Rsz = Reverse ? Ysz : Xsz;
            // profiling showed that ALWAYS put (actual) f0R as outer loop in a tile
            for (auto r = 0u; r < Rsz; r++) {
                frow_t fR = Rcache[r];
                if (fR.shape & ~cfg.empty_area) [[unlikely]] continue;

                for (auto l = 0u; l < Lsz; l++) {
                    frow_t fL = Lcache[l];
                    if (fL.shape & ~cfg.empty_area) [[unlikely]] continue;
                    if (fR.shape & fL.shape) [[unlikely]] continue;

                    BEFORE(comp);
                    impl<H>(cfg, fL, fR,
                            ring_buffer, n_outs, n_chunks,
                            n_reader_chunk, n_writer_chunk);
                    AFTER(comp);
                }
            }
            AFTER(tile);
        }
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
