#include "kernel.h"

#include <algorithm>
#include <ranges>
#include <vector>
#ifdef BMARK
#include <format>
#include <iostream>
#endif

#include "util.cuh"

template <unsigned H, int>
__global__
void legacy_row_search(unsigned, K_PARAMS);

template <unsigned H, bool Reverse>
__global__
__launch_bounds__(768, 2)
void LR_row_search(unsigned shmem_len, K_PARAMS);

extern template __global__ void legacy_row_search<8, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<7, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<6, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<5, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<4, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<3, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<2, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<1, 0>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<8, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<7, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<6, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<5, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<4, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<3, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<2, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<1, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<8, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<7, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<6, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<5, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<4, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<3, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<2, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<1, false>(unsigned, K_PARAMS);

void KParamsFull::launch(cudaStream_t stream) {
#define ARGS \
    shmem_len, \
    ring_buffer, n_outs, n_chunks, \
    n_reader_chunk, n_writer_chunk, \
    cfgs, n_cfgs, \
    ea, f0L, f0Lsz, f0R, f0Rsz

#define L(k, t) \
    do { if (height == 8) k<8, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 7) k<7, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 6) k<6, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 5) k<5, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 4) k<4, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 3) k<3, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 2) k<2, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 1) k<1, t><<<blocks, threads, shmem_len * sizeof(frow32_t), stream>>>(ARGS); \
    else throw std::runtime_error{ std::format("height {} not supported", height) }; \
    } while (false)

    if (!shmem_len) L(legacy_row_search, 0);
    else if (reverse) L(LR_row_search, true);
    else L(LR_row_search, false);
}

void prepare_kernels() {
#define S(k) \
    C(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)); \
    C(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 50176));
#define COMMA ,
#define SS(k, t) S(k<8 COMMA t>) S(k<7 COMMA t>) S(k<6 COMMA t>) S(k<5 COMMA t>) S(k<4 COMMA t>) S(k<3 COMMA t>) S(k<2 COMMA t>) S(k<1 COMMA t>)
    SS(legacy_row_search, 0)
    SS(LR_row_search, true)
    SS(LR_row_search, false)
}

static unsigned known_t[]{ 96, 128, 192, 256, 384, 512, 768, 1024 };
static unsigned known_shmem_b[]{ 5120, 7168, 11776, 15872, 24576, 32768, 50176, 101376 };

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
    for (auto i = 0; i < 7; i++) { // 1024 disabled
        for (auto b = 1ull; b <= wpn && b <= 2147483647ull; b <<= 1) {
            pars.emplace_back(*this, false, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
            pars.emplace_back(*this, true, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
        }
        for (auto b = 3ull; b <= wpn && b <= 2147483647ull; b <<= 1) {
            pars.emplace_back(*this, false, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
            pars.emplace_back(*this, true, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
        }
        for (auto b = 7ull; b <= wpn && b <= 2147483647ull; b <<= 1) {
            pars.emplace_back(*this, false, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
            pars.emplace_back(*this, true, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
        }
        for (auto b = 21ull; b <= wpn && b <= 2147483647ull; b <<= 1) {
            pars.emplace_back(*this, false, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
            pars.emplace_back(*this, true, b, known_t[i], known_shmem_b[i] / sizeof(frow32_t));
        }
    }
    std::ranges::sort(pars, std::less{}, [](const KParams &kp) { return kp.fom(); });
#ifdef BMARK
    return pars;
#else
    return pars.front();
#endif
}

double KParams::fom() const {
    auto oc = std::min(16u, 1536u / threads) * 84; // max blocks per device
    auto e = ((blocks + oc - 1) / oc);

    if (shmem_len == 0) {
        auto c = (1.0 + ((threads + 31) / 32 * 32) * 1e-3) * 1.63e-6;
        auto v = e * c + blocks * 1e-11;
#ifdef BMARK
        std::cout << std::format("<<<{:10},{:5}>>>  [legacy] {:9.2e}*{:3} + {:9.2f} ={:9.2f}\n",
                blocks, threads,
                c, e, blocks * 1e-11, v);
#endif
        return v;
    }

    uint32_t Ltile, Rtile;
    if (f0Lsz + f0Rsz <= shmem_len) {
        Ltile = f0Lsz, Rtile = f0Rsz;
    } else if (f0Lsz < shmem_len / 2) {
        Ltile = f0Lsz, Rtile = shmem_len - f0Lsz;
    } else if (f0Rsz < shmem_len / 2) {
        Ltile = shmem_len - f0Rsz, Rtile = f0Rsz;
    } else {
        Ltile = shmem_len / 2, Rtile = shmem_len - Ltile;
    }
    auto nL = (f0Lsz + Ltile - 1) / Ltile;
    auto nR = (f0Rsz + Rtile - 1) / Rtile;
    if (reverse) {
        std::swap(Ltile, Rtile);
        std::swap(nL, nR);
    }

    auto tpb = static_cast<uint64_t>(threads);
    auto tpg = static_cast<uint64_t>(blocks) * tpb;
    auto iterations = (n_cfgs + tpg - 1) / tpg;

    auto mem = 1.2e-4;
    auto m = nL * (4e-3 + Ltile * mem); // load Lcache
    if (nR == 1) // load Rcache
        m += (1e-1 + Rtile * mem);
    else
        m += nL * nR * (4e-3 + Rtile * mem);

    auto c = nL * nR * Ltile * Rtile * iterations * 7.2e-8; // compute
    auto n = n_cfgs * 2.3e-5; // load cfgs
    auto v = e * (m + c) + n;
#ifdef BMARK
    std::cout << std::format("<<<{:10},{:5},{:5}B>>>[{}] L{}/{} R{}/{} ({:9.2f} +{:9.2f})*{:3}+{:9.2f}={:9.2f}\n",
            blocks, threads, shmem_len * sizeof(frow32_t),
            reverse ? "L" : "R",
            Ltile, nL, Rtile, nR,
            m, c, e, n, v);
#endif
    return v;
}
