#include "kernel.h"

#include "util.cuh"
#include "record.cuh"
#include "sn.cuh"

#define K_PARAMS_OUT \
        /* output ring buffer */ \
        RX                 *ring_buffer, /* __device__ */ \
        unsigned long long *n_outs, /* __device__ */ \
        unsigned long long n_chunks, \
        unsigned long long *n_reader_chunk, /* __managed__, HtoD */ \
        unsigned long long *n_writer_chunk, /* __managed__, DtoH */ \

#define K_PARAMS \
        unsigned shmem_len,
        K_PARAMS_OUT
        /* input vector */ \
        const R *cfgs, const uint64_t n_cfgs, \
        /* constants */ \
        uint8_t ea, \
        const frow_t *f0L, const uint32_t f0Lsz, \
        const frow_t *f0R, const uint32_t f0Rsz

template <unsigned H>
__device__
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

// N is always coalesced
// Y is never coalesced
// X is coalesced and cached at shmem
template <unsigned H, bool Reverse>
__global__
void row_search(unsigned shmem_len, K_PARAMS) {
    extern __shared__ frow_t shmem[/* shmem_len */];

    uint32_t f0Xsz, f0Ysz;
    const frow_t *f0X, *f0Y;
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
        for (auto i = threadIdx.x; i < n_shmem; i += blockDim.x) {
            shmem[i] = f0X[f0Xoffset + i];
        }

        // each warp, inspect the (warpsId0 + i * wpn)-th warp
        auto warpIdx = threadIdx.x % warpSize;
        auto warpsId0 = threadIdx.x / warpSize + blockIdx.x * wpb;
        for (auto k = 0ull; k < iterations; k++) {
            auto w = warpsId0 + k * wpg;
            auto fY = f0Y[w / wpn];
            auto id = warpIdx + w % wpn;
            if (id >= n_cfgs) [[unlikely]]
                continue;

            auto r = cfgs[id];
            for (auto i = 0ull; i < shmem_len && i < n_shmem; i++) {
                auto fX = shmem[i];
                impl(r, Reverse ? fX : fY, Reverse ? fY : fX,
                    ring_buffer, n_outs, n_chunks,
                    n_reader_chunk, n_writer_chunk, ea);
            }
        }
    }
}

template <unsigned H>
__global__
void simple_row_search<H, KOpt::NONE>(K_PARAMS) {
    auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_cfgs * f0Lsz * f0Rsz) [[unlikely]] return;
    auto r = cfgs[idx / f0Rsz / f0Lsz];
    auto fL = f0L[idx / f0Rsz % f0Lsz];
    auto fR = f0R[idx % f0Rsz];
    if (fL.shape & fR.shape) [[unlikely]] return;
    impl(r, fL, fR, ring_buffer, n_outs, n_chunks,
            n_reader_chunk, n_writer_chunk, ea);
}

void KParams::launch(cudaStream_t stream) {
#define ARGS \
    shmem_len, \
    ring_buffer, n_outs, n_chunks, \
    n_reader_chunk, n_writer_chunk, \
    cfgs, n_cfgs, \
    ea, f0L, f0Lsz, f0R, f0Rsz

#define L(k, t) \
    do { if (height == 8) k<8 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 7) k<7 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 6) k<6 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 5) k<5 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 4) k<4 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 3) k<3 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 2) k<2 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else if (height == 1) k<1 t><<<blocks, threads, shmem_len, stream>>>(ARGS); \
    else throw std::runtime_error{ std::format("height {} not supported", height) }; \
    } while (false)

    if (!shmem_len) L(simple_row_search, );
    else if (reverse) L(row_search, true);
    else L(row_search, false);
}
