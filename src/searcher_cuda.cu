#include "searcher_cuda.h"
#include "util.hpp"
#include "util.cuh"
#include "frow.h"
#include "device.h"
#include "sn.cuh"

#include <cuda.h>
#include <cuda/atomic>
#include <linux/kcmp.h>
#include <sys/syscall.h>
#include <algorithm>
#include <chrono>
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
    solutions.emplace_back(
        Rg<R>{
            new R[1]{ { (uint32_t)(empty_area >> 8), (uint32_t)(empty_area >> 8 + 32) } },
            1zu,
            RgType::DELETE,
        },
        empty_area & 0xffu);
}

CudaSearcher::~CudaSearcher() {
    free();
}

void CudaSearcher::free() {
    for (auto &r : solutions)
        r.dispose();
}

void CudaSearcher::search_GPU() {
    Sorter sorter{};
    std::vector<std::unique_ptr<Device>> devs;
    for (auto i = 0; i < n_devices; i++)
        devs.emplace_back(std::make_unique<Device>(i, height, sorter));

    std::mutex mtx;
    auto done = false;
    std::condition_variable cv;
    std::jthread monitor{ [&,this] {
        pthread_setname_np(pthread_self(), "monitor");
        auto sep = syscall(SYS_kcmp, getpid(), getpid(), KCMP_FILE, 1, 2) != 0;
        auto last = 0u;
        using namespace std::chrono_literals;
        std::unique_lock lock{ mtx };
        while (true) {
            cv.wait_for(lock, 100ms, [&]{ return done; });
            if (done)
                break;

            if (sep && last)
                std::cerr << std::format("\33[{}F", last);
            last = sorter.print_stats();
            for (auto &dev : devs)
                dev->print_stats(), last++;
            std::cerr << std::format("\33[37mheight = {}\33[K\33[0m\33[E\33[J\n\n", height);
            last += 3;
        }
    } };

    while (!solutions.empty()) {
        std::ranges::sort(devs, std::less{}, [](const std::unique_ptr<Device> &dev) {
            return dev->get_etc();
        });
        // Device::c is responsible for free up
        devs.front()->dispatch(solutions.front());
        solutions.pop_front();
    }
    for (auto &dev : devs) dev->close();
    // ensure all compute are finished & results are removed from GPU
    for (auto &dev : devs) dev->wait();
    solutions = sorter.join();
    // stop printing stats before ~Device()
    done = true, cv.notify_one(), monitor.join();
    devs.clear();
    height--;
}

uint64_t CudaSearcher::next_size(unsigned i) const {
    auto szid = min(height - 1, 5);
    auto &w = solutions[i];
    return w.len
        * h_frowInfoL[(w.pos >> 0) & 0b1111u].sz[szid]
        * h_frowInfoR[(w.pos >> 4) & 0b1111u].sz[szid];
}
