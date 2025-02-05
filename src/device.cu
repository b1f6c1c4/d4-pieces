#include "device.h"

#include <cuda/atomic>
#include "util.hpp"
#include "util.cuh"
#include "kernel.h"

template <typename ... TArgs>
static void launch(unsigned b, unsigned t, cudaStream_t s, unsigned height,
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

static std::pair<uint64_t, uint32_t> balance(uint64_t n) {
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

Device::Device(int d)
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

Device::~Device() {
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

bool Device::c_completed() const {
    auto res = cudaStreamQuery(c_stream);
    switch (res) {
        case cudaSuccess: return true;
        case cudaErrorNotReady: return false;
        default: C(res); return false;
    }
}

bool Device::m_completed() const {
    return m_data.empty();
}

void Device::dispatch(unsigned pos, unsigned height, Rg<R> cfgs) {
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

void Device::recycle(bool last) {
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
        Rg<RX> r{
            reinterpret_cast<RX *>(std::aligned_alloc(4096, CYC_CHUNK * sizeof(RX))),
            CYC_CHUNK };
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

void Device::collect(Sorter &sorter, unsigned height) {
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
        sorter.push(m_data.front(), height);
        m_events.pop_front();
        m_data.pop_front();
    }
}
