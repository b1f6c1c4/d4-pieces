#include "kernel.h"

#include <algorithm>
#include <ranges>
#include <vector>

#include "util.cuh"

template <unsigned H, int>
__global__
void simple_row_search(unsigned, K_PARAMS);

template <unsigned H, bool Reverse>
__global__
__launch_bounds__(768, 2)
void row_search(unsigned shmem_len, K_PARAMS);

extern template __global__ void simple_row_search<8, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<7, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<6, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<5, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<4, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<3, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<2, 0>(unsigned, K_PARAMS);
extern template __global__ void simple_row_search<1, 0>(unsigned, K_PARAMS);
extern template __global__ void row_search<8, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<7, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<6, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<5, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<4, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<3, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<2, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<1, true>(unsigned, K_PARAMS);
extern template __global__ void row_search<8, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<7, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<6, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<5, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<4, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<3, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<2, false>(unsigned, K_PARAMS);
extern template __global__ void row_search<1, false>(unsigned, K_PARAMS);

void KParamsFull::launch(cudaStream_t stream) {
#define ARGS \
    shmem_len, \
    ring_buffer, n_outs, n_chunks, \
    n_reader_chunk, n_writer_chunk, \
    cfgs, n_cfgs, \
    ea, f0L, f0Lsz, f0R, f0Rsz

#define L(k, t) \
    do { if (height == 8) k<8, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 7) k<7, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 6) k<6, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 5) k<5, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 4) k<4, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 3) k<3, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 2) k<2, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else if (height == 1) k<1, t><<<blocks, threads, shmem_len * sizeof(frow_t), stream>>>(ARGS); \
    else throw std::runtime_error{ std::format("height {} not supported", height) }; \
    } while (false)

    if (!shmem_len) L(simple_row_search, 0);
    else if (reverse) L(row_search, true);
    else L(row_search, false);
}

void prepare_kernels() {
#define S(k) \
    C(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)); \
    C(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 4181 * sizeof(frow_t)));
#define COMMA ,
#define SS(k, t) S(k<8 COMMA t>) S(k<7 COMMA t>) S(k<6 COMMA t>) S(k<5 COMMA t>) S(k<4 COMMA t>) S(k<3 COMMA t>) S(k<2 COMMA t>) S(k<1 COMMA t>)
    SS(simple_row_search, 0)
    SS(row_search, true)
    SS(row_search, false)
}

static unsigned known_t[]{ 96, 128, 192, 256, 384, 512, 768 };
static unsigned known_shmem_b[]{ 5120, 7168, 11776, 15872, 24576, 32768, 50176 };

#ifdef BMARK
std::vector<KParams> KSizing::optimize() const {
#else
KParams KSizing::optimize() const {
#endif
    std::vector<KParams> pars;
    auto n = n_cfgs * f0Lsz * f0Rsz;
    for (auto t = 1ull; t <= 512u; t <<= 1)
        if ((n + t - 1) / t <= 2147483647ull)
            pars.emplace_back(*this, false, (n + t - 1) / t, t, 0);
    for (auto t = 3ull; t <= 1024u; t <<= 1)
        if ((n + t - 1) / t <= 2147483647ull)
            pars.emplace_back(*this, false, (n + t - 1) / t, t, 0);
    auto wpn = (n_cfgs + 31) / 32;
    for (auto i = 0; i < 7; i++)
        for (auto b = 1ull; b <= wpn && b <= 2147483647ull; b <<= 1) {
            pars.emplace_back(*this, false, b, known_t[i], known_shmem_b[i] / sizeof(frow_t));
            pars.emplace_back(*this, true, b, known_t[i], known_shmem_b[i] / sizeof(frow_t));
        }
    std::ranges::sort(pars, std::less{}, [](const KParams &kp) { return kp.fom(); });
#ifdef BMARK
    return pars;
#else
    return pars.front();
#endif
}

double KParams::fom() const {
    auto oc = std::min(16u, 1536u / threads) * 84;

    auto v = 0.0;

    if (shmem_len == 0) {
        v = (1.0 + ((threads + 31) / 32 * 32) * 1e-3) * 1e-6;
    } else {
        auto f0Xsz = f0Rsz;
        auto f0Ysz = f0Lsz;
        if (reverse)
            std::swap(f0Xsz, f0Ysz);

        auto wpb = static_cast<uint64_t>(threads) / 32;
        auto wpg = static_cast<uint64_t>(blocks) * wpb;
        auto wpn = (n_cfgs + 32 - 1) / 32;
        auto iterations = (wpn * f0Ysz + wpg - 1) / wpg;

        auto outer_loop = (f0Xsz + shmem_len - 1) / shmem_len;

        auto shmem_loop = std::min(f0Xsz, shmem_len) / blocks;
        v += outer_loop * (4.0 + shmem_loop); // load shmem

        v += outer_loop * iterations * 16.0; // load fY
        v += outer_loop * iterations * 20.0; // load cfgs[id]

        v += outer_loop * iterations * std::min(f0Xsz, shmem_len) * 5.0; // compute

        v *= 3.3e-8;
    }
    return ((blocks + oc - 1) / oc) * v;
}
