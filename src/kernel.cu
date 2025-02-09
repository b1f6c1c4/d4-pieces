#include "kernel.h"

#include <algorithm>
#include <ranges>
#include <vector>
#include <format>
#include <iostream>

#include "util.cuh"
#include "util.hpp"

template <unsigned H, int>
__global__
void legacy_row_search(unsigned, K_PARAMS);

template <unsigned H, int W, bool Reverse>
__global__
__launch_bounds__(W, 1536 / W)
void LR_row_search(unsigned shmem_len, K_PARAMS);

extern template __global__ void legacy_row_search<8, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<7, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<6, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<5, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<4, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<3, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<2, 0>(unsigned, K_PARAMS);
extern template __global__ void legacy_row_search<1, 0>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<8, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<7, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<6, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<5, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<4, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<3, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<2, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<1, 768, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<8, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<7, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<6, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<5, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<4, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<3, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<2, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<1, 768, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<8, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<7, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<6, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<5, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<4, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<3, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<2, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<1, 1024, true>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<8, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<7, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<6, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<5, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<4, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<3, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<2, 1024, false>(unsigned, K_PARAMS);
extern template __global__ void LR_row_search<1, 1024, false>(unsigned, K_PARAMS);

#define COMMA ,
#define CCMMA ,

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
    else if (threads > 768)
        if (reverse) L(LR_row_search, 1024 COMMA true);
        else L(LR_row_search, 1024 COMMA false);
    else
        if (reverse) L(LR_row_search, 768 COMMA true);
        else L(LR_row_search, 768 COMMA false);
}

void prepare_kernels() {
#define S(...) \
    C(cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)); \
    C(cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
#define SS(k, ...) \
    S(k<8 COMMA __VA_ARGS__>) \
    S(k<7 COMMA __VA_ARGS__>) \
    S(k<6 COMMA __VA_ARGS__>) \
    S(k<5 COMMA __VA_ARGS__>) \
    S(k<4 COMMA __VA_ARGS__>) \
    S(k<3 COMMA __VA_ARGS__>) \
    S(k<2 COMMA __VA_ARGS__>) \
    S(k<1 COMMA __VA_ARGS__>)
    SS(legacy_row_search, 0)
    SS(LR_row_search, 768  COMMA true)
    SS(LR_row_search, 768  COMMA false)
    SS(LR_row_search, 1024 COMMA true)
    SS(LR_row_search, 1024 COMMA false)
}

static unsigned known_t[]{ 96, 128, 192, 256, 384, 512, 768, 1024 };
static unsigned known_shmem_b[]{ 5120, 7168, 11776, 15872, 24576, 32768, 50176, 101376 };

#ifdef BMARK
std::vector<KParams> KSizing::optimize() const {
#else
KParams KSizing::optimize(bool debug) const {
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
    for (auto i = 0; i < 8; i++) {
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
    if (debug) {
        std::cout << std::format("kernel#optimize: best kernel params for {} are:\n",
                to_string());
        for (auto i = 0zu; i < pars.size() && i < 10zu; i++)
            std::cout << std::format("      #{}\n", pars[i].to_string(false));
    }
    return pars.front();
#endif
}

std::string KSizing::to_string() const {
    return std::format("[{:<6}*L{:<5}*R{:<5}]", n_cfgs, f0Lsz, f0Rsz);
}

std::string KParams::to_string(bool full) const {
    std::string s;
    if (!shmem_len)
        s = std::format("<<<{:11},{:5}>>>[legacy]", blocks, threads);
    else
        s = std::format("<<<{:9},{:5},{:6}>>>[{}]", blocks, threads,
                shmem_len * sizeof(frow32_t), reverse ? 'L' : 'R');
    if (full)
        s += KSizing::to_string();
    s += std::format(" ~ {}", display(fom()));
    return s;
}

#ifdef BMARK
double KParams::fom(bool debug) const {
#else
double KParams::fom() const {
#endif
    auto oc = std::min(16u, 1536u / threads) * 84; // max blocks per device
    auto util = 1536.0 / ((1536u / threads) * threads);
    auto e = ((blocks + oc - 1) / oc);

    if (shmem_len == 0) {
        auto c = (1.0 + ((threads + 31) / 32 * 32) * 1e-3) * 1.63e-6;
        auto v = e * c + blocks * 1e-11;
#ifdef BMARK
        if (debug) {
            std::cout << std::format("<<<{:10},{:5}>>>  [legacy] {:9.2e}*{:3} + {:9.2f} ={:9.2f}\n",
                    blocks, threads,
                    c, e, blocks * 1e-11, v);
        }
#endif
        return v + 500e-6;
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
        m += (4e-3 + Rtile * mem);
    else
        m += nL * nR * (4e-3 + Rtile * mem);

    auto c = nL * nR * Ltile * Rtile * iterations * 7.2e-8; // compute
    auto n = n_cfgs * 2.3e-5; // load cfgs
    auto v = e * (m + c * util) + n;
#ifdef BMARK
    if (debug) {
        std::cout << std::format("<<<{:9},{:5},{:6}>>>   {}{}*{}-{}{}*{}   ({:9.2f} +{:9.2f}*{})*{:3}+{:9.2f}={:9.2f}\n",
                blocks, threads, shmem_len * sizeof(frow32_t),
                reverse ? "R" : "L",
                Ltile,
                nL,
                reverse ? "L" : "R",
                Rtile,
                nR,
                m, c, util, e, n, v);
    }
#endif
    return v + 500e-6;
}
