#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>
#include <format>
#include <iostream>

#include "util.cuh"
#include "util.hpp"

#include "kernel.cuh"

#define COMMA ,
#define CCMMA ,

void KParamsFull::launch(cudaStream_t stream) {
#ifdef BMARK
#define ARGS_EX , perf
#else
#define ARGS_EX
#endif
#define ARGS \
    Ltile, Rtile, \
    ring_buffer, n_outs, n_chunks, \
    n_reader_chunk, n_writer_chunk, \
    cfgs, n_cfgs, \
    ea, f0L, f0Lsz, f0R, f0Rsz ARGS_EX

#define L(k, t) \
    do { if (height == 8) k<8, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 7) k<7, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 6) k<6, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 5) k<5, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 4) k<4, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 3) k<3, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 2) k<2, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else if (height == 1) k<1, t><<<blocks(), threads, (Ltile + Rtile) * sizeof(frow32_t), stream>>>(ARGS); \
    else THROW("height {} not supported", height); \
    } while (false)

    switch (ty) {
        case KKind::Legacy:
            L(linear_row_search, 0);
            break;
        case KKind::CoalescedR:
            L(linear_row_search, +1);
            break;
        case KKind::CoalescedL:
            L(linear_row_search, -1);
            break;
        case KKind::TiledStandard:
            if (threads > 768)
                L(tiled_row_search, 1024 COMMA false);
            else
                L(tiled_row_search, 768 COMMA false);
            break;
       case KKind::TiledReversed:
            if (threads > 768)
                L(tiled_row_search, 1024 COMMA true);
            else
                L(tiled_row_search, 768 COMMA true);
            break;
        default:
            THROW("unknown ty");
    }
}

void prepare_kernels() {
#define S(sz, ...) \
    C(cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1)); \
    C(cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
#define SS(sz, k, ...) \
    S(sz, k<8 COMMA __VA_ARGS__>) \
    S(sz, k<7 COMMA __VA_ARGS__>) \
    S(sz, k<6 COMMA __VA_ARGS__>) \
    S(sz, k<5 COMMA __VA_ARGS__>) \
    S(sz, k<4 COMMA __VA_ARGS__>) \
    S(sz, k<3 COMMA __VA_ARGS__>) \
    S(sz, k<2 COMMA __VA_ARGS__>) \
    S(sz, k<1 COMMA __VA_ARGS__>)
    SS(0, linear_row_search, 0)
    SS(0, linear_row_search, +1)
    SS(0, linear_row_search, -1)
    SS(50176, tiled_row_search, 768  COMMA true)
    SS(50176, tiled_row_search, 768  COMMA false)
    SS(101376, tiled_row_search, 1024 COMMA true)
    SS(101376, tiled_row_search, 1024 COMMA false)
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
    if (n <= 256 * 2147483647ull)
        pars.emplace_back(*this, KKind::Legacy, (n + 256 - 1) / 256, 256, 0);
    else if (n <= 768 * 2147483647ull)
        pars.emplace_back(*this, KKind::Legacy, (n + 768 - 1) / 768, 768, 0);
    if (n <= 256 * 2147483647ull)
        pars.emplace_back(*this, KKind::CoalescedR, (n + 256 - 1) / 256, 256, 0);
    else if (n <= 768 * 2147483647ull)
        pars.emplace_back(*this, KKind::CoalescedL, (n + 768 - 1) / 768, 768, 0);
    if (n <= 256 * 2147483647ull)
        pars.emplace_back(*this, KKind::CoalescedR, (n + 256 - 1) / 256, 256, 0);
    else if (n <= 768 * 2147483647ull)
        pars.emplace_back(*this, KKind::CoalescedL, (n + 768 - 1) / 768, 768, 0);
    for (auto i = 0; i < 8; i++) {
        pars.emplace_back(*this, KKind::TiledStandard,
                (n_cfgs + known_t[i] - 1) / known_t[i],
                known_t[i], known_shmem_b[i] / sizeof(frow32_t));
    }
    std::ranges::sort(pars, std::less{}, [](const KParams &kp) { return kp.fom(); });
#ifdef BMARK
    return pars;
#else
    if (debug) {
        std::print("kernel#optimize: best kernel params for {} are:\n",
                to_string());
        for (auto i = 0zu; i < pars.size() && i < 10zu; i++)
            std::print("      #{}\n", pars[i].to_string(false));
    }
    return pars.front();
#endif
}

uint64_t KParams::blocks() const {
    switch (ty) {
        case KKind::Legacy:
        case KKind::CoalescedR:
        case KKind::CoalescedL:
            return n_cfgs * f0Lsz * f0Rsz / threads;
        case KKind::TiledStandard:
        case KKind::TiledReversed:
            return ((n_cfgs + threads - 1) / threads) *
                ((f0Lsz + Ltile - 1) / Ltile) *
                ((f0Rsz + Rtile - 1) / Rtile);
        default:
            THROW("unknown ty");
    }
}

std::string KSizing::to_string() const {
    return std::format("[{:<6}*L{:<5}*R{:<5}]", n_cfgs, f0Lsz, f0Rsz);
}

std::string KParams::to_string(bool full) const {
    std::string s;
    switch (ty) {
        case KKind::Legacy:
            s = std::format("<<<{:11},{:5}>>>[legacy]", blocks(), threads);
            break;
        case KKind::CoalescedR:
        case KKind::CoalescedL:
            s = std::format("<<<{:10},{:4},{:5}>>>[C{}]", blocks(), threads,
                    threads * sizeof(frow32_t), ty == KKind::CoalescedL ? 'L' : 'R');
            break;
        case KKind::TiledStandard:
        case KKind::TiledReversed:
            s = std::format("<<<{:9},{:5},{:4}*{:4}>>>[{}]", blocks(), threads,
                    Ltile, Rtile,
                    ty == KKind::TiledReversed ? 'L' : 'R');
            break;
        default:
            THROW("unknown ty");
    }
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
    auto e = ((blocks() + oc - 1) / oc);

    if (ty == KKind::Legacy) {
        auto c = (1.0 + ((threads + 31) / 32 * 32) * 1e-3) * 2.0e-6;
        auto v = e * c + blocks() * 1e-11;
#ifdef BMARK
        if (debug) {
            std::print("<<<{:10},{:5}>>>  [legacy] {:9.2e}*{:3} + {:9.2f} ={:9.2f}\n",
                    blocks(), threads,
                    c, e, blocks() * 1e-11, v);
        }
#endif
        return v; // + 500e-6;
    }

    auto nL = (f0Lsz + Ltile - 1) / Ltile;
    auto nR = (f0Rsz + Rtile - 1) / Rtile;

    auto m = 1.0 * (nL * nR) * (Ltile + Rtile); // load L/Rcache
    if ((Ltile + Rtile) * sizeof(frow32_t) >= 48 * 1024ull) // beyond 48K penalty
        m += 0.58 * ((Ltile + Rtile) * sizeof(frow32_t) - 48 * 1024ull);
    m *= 5e-4 * std::min(16u, 1536u / threads); // per block

    auto c = nL * nR * Ltile * Rtile * 7.2e-200; // compute

    auto n = n_cfgs * 1.5e-5 * ::pow(nL * nR, 0.87); // load cfgs

    auto v = e * (m + c * util) + n;
#ifdef BMARK
    if (debug) {
        std::print("<<<{:9},{:5},{:4}*{:4}>>>   {}{}*{}-{}{}*{}   ({:9.2f} +{:9.2f}*{})*{:3}+{:9.2f}={:9.2f}\n",
                blocks(), threads, Ltile, Rtile,
                ty == KKind::TiledReversed ? "R" : "L",
                Ltile, nL,
                ty == KKind::TiledReversed ? "L" : "R",
                Rtile, nR,
                m, c, util, e, n, v);
    }
#endif
    return v / 1e-6; // + 500e-6;
}
