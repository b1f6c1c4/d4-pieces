#include "searcher_cuda.h"
#include "util.hpp"
#include "util.cuh"
#include "frow.h"
#include "device.h"
#include "sn.cuh"

#include <cuda.h>
#include <cuda/atomic>
#include <algorithm>
#include <cstring>
#include <memory>
#include <deque>
#include <iostream>
#include <format>
#include <unistd.h>
#include <cstdio>

template <unsigned H>
__global__
void d_row_search(
        // output ring buffer
        RX                 *ring_buffer, // __device__
        unsigned long long *n_outs, // __device__
        unsigned long long n_chunks,
        unsigned long long *n_reader_chunk, // __managed__, HtoD
        unsigned long long *n_writer_chunk, // __managed__, DtoH
        // input vector
        const R *cfgs, const uint64_t n_cfgs,
        // constants
        uint8_t ea,
        const frow_t *f0L, const uint32_t f0Lsz,
        const frow_t *f0R, const uint32_t f0Rsz) {
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

CudaSearcher::CudaSearcher(uint64_t empty_area)
    : solutions{}, height{ (std::bit_width(empty_area) + 8u - 1u) / 8u } {
    auto &r = solutions[empty_area & 0xffu];
    C(cudaMallocManaged(&r.ptr, sizeof(R)));
    r.ptr[0] = RX{ (uint32_t)(empty_area >> 8), (uint32_t)(empty_area >> 8 + 32) };
    r.len = 1;
}

CudaSearcher::~CudaSearcher() {
    free();
}

void CudaSearcher::free() {
    for (auto &r : solutions) {
        if (r.ptr)
            cudaFree(r.ptr);
        r.ptr = nullptr;
        r.len = 0;
    }
}

void CudaSearcher::search_GPU() {
    Sorter sorter{ *this };
    std::vector<std::unique_ptr<Device>> devs;
    for (auto i = 0; i < n_devices; i++)
        devs.emplace_back(std::make_unique<Device>(i));

    for (auto ipos = 0u; ipos <= 255u; ipos++) {
        std::ranges::sort(devs, std::greater{}, [](const std::unique_ptr<Device> &dev) {
            return dev->workload;
        });
        devs.front()->dispatch(ipos, height, solutions[ipos]);
        for (auto &dev : devs) {
            dev->recycle(false);
            dev->collect(sorter);
        }
    }
    bool flag;
    do {
        flag = true;
        for (auto &dev : devs) {
            flag &= dev->c_completed();
            dev->recycle(false);
            dev->collect(sorter);
        }
    } while (!flag);
    for (auto &dev : devs) {
        dev->recycle(true);
        dev->collect(sorter);
    }
    do {
        flag = true;
        for (auto &dev : devs) {
            flag &= dev->m_completed();
            dev->collect(sorter);
        }
    } while (!flag);
    devs.clear();
    sorter.join();
    height--;
}

uint64_t CudaSearcher::next_size(unsigned pos) const {
    auto szid = min(height - 1, 5);
    return solutions[pos].len
        * h_frowInfoL[(pos >> 0) & 0b1111u].sz[szid]
        * h_frowInfoR[(pos >> 4) & 0b1111u].sz[szid];
}

Rg<R> CudaSearcher::write_solution(unsigned pos, size_t sz) {
    auto &r = solutions[pos];
    if (r.ptr) {
        C(cudaFree(r.ptr));
        r.ptr = nullptr, r.len = 0;
    }
    if (sz)
        C(cudaMallocManaged(&r.ptr, sz * sizeof(R), cudaMemAttachHost));
    r.len = sz;
    return r;
}

Rg<R> *CudaSearcher::write_solutions(size_t sz) {
    for (auto pos = 0; pos <= 255; pos++) {
        auto &[ptr, len] = solutions[pos];
        if (ptr) C(cudaFree(ptr));
        ptr = nullptr;
        len = 0;
        C(cudaMallocManaged(&ptr, sz * sizeof(R), cudaMemAttachHost));
    }
    return solutions;
}
