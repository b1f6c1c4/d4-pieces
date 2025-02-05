#include "kernel.h"

#include "util.cuh"
#include "record.cuh"
#include "sn.cuh"

template <unsigned H>
__global__
void d_row_search(K_PARAMS) {
    auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_cfgs * f0Lsz * f0Rsz) [[unlikely]] return;
    auto r = cfgs[idx / f0Rsz / f0Lsz];
    auto fL = f0L[idx / f0Rsz % f0Lsz];
    auto fR = f0R[idx % f0Rsz];
    auto cfg = parse_R<H>(r, ea);
    if (fL.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fR.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fL.shape & fR.shape) [[unlikely]] return;
    d_push(cfg.nm_cnt, cfg.ex, fL.nm0123);
    d_push(cfg.nm_cnt, cfg.ex, fR.nm0123);
    d_sn(cfg.nm_cnt, cfg.ex);
    if (!d_uniq_chk(cfg.nm_cnt, cfg.ex)) [[unlikely]] return;
    cfg.empty_area &= ~fL.shape;
    cfg.empty_area &= ~fR.shape;
    auto ocfg = assemble_R<H - 1>(cfg);
    auto out = __nv_atomic_fetch_add(n_outs, 1,
            __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
spin:
    auto nrc = __nv_atomic_load_n(n_reader_chunk,
            __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM);
    if (out >= (nrc + n_chunks - 1u) * CYC_CHUNK) {
        __nanosleep(1000000);
        goto spin;
    }
    ring_buffer[out % (n_chunks * CYC_CHUNK)] = ocfg; // slice
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
}

template __global__ void d_row_search<8>(K_PARAMS);
template __global__ void d_row_search<7>(K_PARAMS);
template __global__ void d_row_search<6>(K_PARAMS);
template __global__ void d_row_search<5>(K_PARAMS);
template __global__ void d_row_search<4>(K_PARAMS);
template __global__ void d_row_search<3>(K_PARAMS);
template __global__ void d_row_search<2>(K_PARAMS);
template __global__ void d_row_search<1>(K_PARAMS);
