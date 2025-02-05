#include "searcher_cuda.h"
#include "util.hpp"
#include "util.cuh"
#include "frow.h"
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

/**
 * 128 resident grids / device (Concurrent Kernel Execution)
 * 2147483647*65535*65535 blocks / grid
 * 1024*1024*64 <= 1024 threads / block
 * 32 threads / warp
 * 16 blocks / SM
 * 48 threads / warp
 * 1536 threads / SM
 * 65536 regs / SM
 * 255 regs / threads
 * 64KiB constant memory (8KiB cache)
 */

#define CYC_CHUNK (32ull * 1048576ull / sizeof(RX))

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

template <typename ... TArgs>
void launch(unsigned b, unsigned t, cudaStream_t s, unsigned height,
        TArgs && ... args) {
    if (height == 8)
        d_row_search<8><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 7)
        d_row_search<7><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 6)
        d_row_search<6><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 5)
        d_row_search<5><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 4)
        d_row_search<4><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 3)
        d_row_search<3><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 2)
        d_row_search<2><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else if (height == 1)
        d_row_search<1><<<b, t, 0, s>>>(std::forward<TArgs>(args)...);
    else
        throw std::runtime_error{ std::format("height {} not supported", height) };
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

std::pair<uint64_t, uint32_t> balance(uint64_t n) {
    if (n <= 32)
        return { 1, n };
    if (n <= 32 * 84 * 3)
        return { (n + 31) / 32, 32 };
    if (n <= 64 * 84 * 3)
        return { (n + 63) / 64, 64 };
    if (n <= 96 * 84 * 3)
        return { (n + 95) / 96, 96 };
    if (n <= 128 * 84 * 3)
        return { (n + 127) / 128, 128 };
    if (n <= 256 * 84 * 3)
        return { (n + 255) / 256, 256 };
    return { (n + 511) / 512, 512 };
}

struct Device {
    int dev;
    cudaStream_t c_stream, m_stream;

    RX *ring_buffer; // __device__
    unsigned long long n_chunks;
    unsigned long long *counters; // __managed__, n_reader_chunk, n_writer_chunk

    unsigned long long *n_outs; // __device__, owned

    uint64_t workload;

    unsigned long long m_scheduled;
    std::deque<Rg<RX>> m_data;
    std::deque<cudaEvent_t> m_events;

    explicit Device(int d)
        : dev{ d }, c_stream{}, m_stream{}, ring_buffer{},
          n_chunks{}, counters{}, n_outs{}, workload{},
          m_scheduled{}, m_data{}, m_events{} {

        C(cudaMallocManaged(&counters, 2 * sizeof(unsigned long long)));
        cuda::atomic_ref n_reader_chunk{ counters[0] };
        cuda::atomic_ref n_writer_chunk{ counters[1] };
        n_reader_chunk.store(0, cuda::memory_order_release);
        n_writer_chunk.store(0, cuda::memory_order_release);

        C(cudaSetDevice(d));

        size_t sz_free, sz_total;
        C(cudaMemGetInfo(&sz_free, &sz_total));
        n_chunks = (9 * sz_free / 10 / sizeof(RX) + CYC_CHUNK - 1) / CYC_CHUNK;

        C(cudaStreamCreateWithFlags(&c_stream, cudaStreamNonBlocking));
        C(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

        C(cudaMallocAsync(&n_outs, sizeof(unsigned long long), c_stream));
        unsigned long long zero{};
        C(cudaMemcpyAsync(n_outs, &zero, sizeof(zero), cudaMemcpyHostToDevice, c_stream));

        std::cout << std::format("dev#{}: allocating {} * {}B = {}B ring buffer\n",
                dev, n_chunks, display(CYC_CHUNK * sizeof(RX)),
                display(n_chunks * CYC_CHUNK * sizeof(RX)));
        C(cudaMallocAsync(&ring_buffer, n_chunks * CYC_CHUNK * sizeof(RX), c_stream));
    }

    ~Device() {
        C(cudaSetDevice(dev));
        for (auto ev : m_events)
            C(cudaEventDestroy(ev));
        C(cudaStreamSynchronize(c_stream));
        C(cudaStreamDestroy(c_stream));
        C(cudaStreamSynchronize(m_stream));
        C(cudaStreamDestroy(m_stream));
        C(cudaFree(n_outs));
        C(cudaFree(counters));
        C(cudaFree(ring_buffer));
    }

    [[nodiscard]] bool c_completed() const {
        auto res = cudaStreamQuery(c_stream);
        switch (res) {
            case cudaSuccess: return true;
            case cudaErrorNotReady: return false;
            default: C(res); return false;
        }
    }

    [[nodiscard]] bool m_completed() const {
        return m_data.empty();
    }

    void dispatch(unsigned pos, unsigned height, Rg<R> cfgs) {
        auto [ptr, len] = cfgs;
        if (!ptr || !len)
            return;
        auto szid = min(height - 1, 5);
        auto fanoutL = h_frowInfoL[(pos >> 0) & 0b1111u].sz[szid];
        auto fanoutR = h_frowInfoR[(pos >> 4) & 0b1111u].sz[szid];
        auto sz = len * fanoutL * fanoutR;
        auto d_f0L = d_frowDataL[dev][pos >> 0 & 0xfu];
        auto d_f0R = d_frowDataR[dev][pos >> 4 & 0xfu];
        auto [b, t] = balance(sz);
        std::cout << std::format("dev#{}: 0b{:08b}<<<{:8}, {:3}>>> = {:<6}*L{:<5}*R{:<5} => {:>9}B\n",
                dev, pos, b, t,
                len, fanoutL, fanoutR, display(sz * sizeof(R)));
        C(cudaSetDevice(dev));
        C(cudaMemAdvise(ptr, len * sizeof(R), cudaMemAdviseSetReadMostly, dev));
        C(cudaStreamAttachMemAsync(c_stream, ptr, len * sizeof(R)));
        C(cudaMemPrefetchAsync(ptr, len * sizeof(R), dev, c_stream));
        launch(b, t, c_stream, height,
                // output ring buffer
                ring_buffer, n_outs,
                n_chunks,
                &counters[0], &counters[1],
                // input vector
                ptr, len,
                // constants
                pos,
                d_f0L, fanoutL,
                d_f0R, fanoutR);
        workload += b * t;
    }

    void recycle(bool last) {
        C(cudaSetDevice(dev));

        if (last) {
            std::cout << std::format("dev#{}: synchronize\n", dev);
            C(cudaStreamSynchronize(c_stream)); // necessary as kernels may be still finishing
        }

        cuda::atomic_ref n_reader_chunk{ counters[0] };
        cuda::atomic_ref n_writer_chunk{ counters[1] };
        auto nwc = n_writer_chunk.load(cuda::memory_order_acquire);
        while (m_scheduled < nwc) {
            std::cout << std::format("dev#{}: start DtoH chunk #{:0{}}/{} ({} B)\n",
                    dev, m_scheduled, count_digits(n_chunks),
                    n_chunks, display(CYC_CHUNK * sizeof(RX)));
            Rg<RX> r{ new RX[CYC_CHUNK], CYC_CHUNK };
            C(cudaMemcpyAsync(r.ptr,
                        ring_buffer + (m_scheduled % n_chunks) * CYC_CHUNK,
                        CYC_CHUNK * sizeof(RX), cudaMemcpyDeviceToHost, m_stream));
            cudaEvent_t ev;
            C(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            C(cudaEventRecord(ev, m_stream));
            m_data.push_back(r);
            m_events.push_back(ev);
            m_scheduled++;
        }

        if (!last)
            return;

        unsigned long long used;
        C(cudaMemcpyAsync(&used, n_outs, sizeof(used), cudaMemcpyDeviceToHost, c_stream));
        C(cudaStreamSynchronize(c_stream));
        if (used < nwc * CYC_CHUNK)
            throw std::runtime_error{ "internal error" };
        if (used >= (nwc + 1u) * CYC_CHUNK)
            throw std::runtime_error{ "internal error" };
        if (m_scheduled != nwc)
            throw std::runtime_error{ "internal error" };
        if (used == nwc * CYC_CHUNK)
            return;

        auto sz = used - nwc * CYC_CHUNK;
        std::cout << std::format("dev#{}: start tail DtoH chunk #{:0{}}/{} for {} entries ({} B)\n",
                dev, nwc, count_digits(n_chunks),
                n_chunks, sz, display(sz * sizeof(RX)));
        Rg<RX> r{ new RX[sz], sz };
        C(cudaMemcpyAsync(r.ptr, ring_buffer + (nwc % n_chunks) * CYC_CHUNK,
                    sz * sizeof(RX), cudaMemcpyDeviceToHost, m_stream));
        cudaEvent_t ev;
        C(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        C(cudaEventRecord(ev, m_stream));
        m_data.push_back(r);
        m_events.push_back(ev);
    }

    void collect(Sorter &sorter) {
        cuda::atomic_ref n_reader_chunk{ counters[0] };
        while (!m_events.empty()) {
            auto ev = m_events.front();
            auto err = cudaEventQuery(ev);
            if (err == cudaErrorNotReady)
                return;
            C(err);
            C(cudaEventDestroy(ev));
            auto nrc = n_reader_chunk.fetch_add(1, cuda::memory_order_release);
            std::cout << std::format("dev#{}: pushing chunk #{:0{}} ({} entries, {} B) to sorter\n",
                    dev, nrc, count_digits(n_chunks),
                    CYC_CHUNK, display(CYC_CHUNK * sizeof(RX)));
            sorter.push(m_data.front());
            m_events.pop_front();
            m_data.pop_front();
        }
    }
};

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
