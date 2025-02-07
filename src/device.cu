#include "device.h"

#include <cuda/atomic>
#include <chrono>
#include <sys/sysinfo.h>
#include <pthread.h>
#include <unistd.h>
#include "util.hpp"
#include "util.cuh"
#include "kernel.h"

using namespace std::chrono_literals;

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

Device::Work::Work(WL work, int dev, unsigned height) : WL{ work }, p{} {
    auto l = pos >> 0 & 0b1111u;
    auto r = pos >> 4 & 0b1111u;
    auto szid = min(height - 1, 5);
    fanoutL = h_frowInfoL[l].sz[szid];
    fanoutR = h_frowInfoR[r].sz[szid];
    sz = len * fanoutL * fanoutR;
    d_f0L = d_frowDataL[dev][l];
    d_f0R = d_frowDataR[dev][r];
    std::tie(b, t) = balance(sz);
    load = b * t;
}

Device::Device(int d, unsigned h, Sorter &s)
    : dev{ d }, height{ h }, sorter{ s } {
    C(cudaSetDevice(d));

    C(cudaMallocManaged(&counters, 2 * sizeof(unsigned long long)));
    cuda::atomic_ref n_reader_chunk{ counters[0] };
    cuda::atomic_ref n_writer_chunk{ counters[1] };
    n_reader_chunk.store(0, cuda::memory_order_release);
    n_writer_chunk.store(0, cuda::memory_order_release);

    C(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    size_t sz_free, sz_total;
    C(cudaMemGetInfo(&sz_free, &sz_total));
    n_chunks = (7 * sz_free / 10 / sizeof(RX) + CYC_CHUNK - 1) / CYC_CHUNK;

    // launch thread AFTER setting up counters and n_chunks
    c_thread = std::jthread{ &Device::c_entry, this };
    m_thread = std::jthread{ &Device::m_entry, this };
}

void Device::c_entry() {
    pthread_setname_np(pthread_self(), std::format("dev#{}.c", dev).c_str());
    C(cudaSetDevice(dev));

    C(cudaStreamCreateWithFlags(&c_stream, cudaStreamNonBlocking));

    C(cudaMallocAsync(&n_outs, sizeof(unsigned long long), c_stream));
    unsigned long long zero{};
    C(cudaMemcpyAsync(n_outs, &zero, sizeof(zero), cudaMemcpyHostToDevice, c_stream));

    std::cout << std::format("dev#{}: allocating {} * {}B = {}B ring buffer\n",
            dev, n_chunks, display(CYC_CHUNK * sizeof(RX)),
            display(n_chunks * CYC_CHUNK * sizeof(RX)));
    C(cudaMallocAsync(&ring_buffer, n_chunks * CYC_CHUNK * sizeof(RX), c_stream));

    std::unique_lock lock{ mtx };
    xc_ready = true;
    cv.notify_all();

again:
    cv.wait_for(lock, 50ms, [this]{ return xc_closed || !xc_queue.empty(); });

    // synchronously free up host inputs copies
    lock.unlock();
    for (auto &work : c_works) {
        if (!work)
            continue;
        auto err = cudaEventQuery(work.ev_m);
        if (err == cudaErrorNotReady)
            continue;
        C(err);
        C(cudaEventDestroy(work.ev_m));
        work.ev_m = cudaEvent_t{};
        std::cout << std::format("dev#{}: free up {}B host input mem ({} entries)\n",
                dev, display(work.len * sizeof(R)), work.len);
        work.dispose();
    }

    // synchronously free up device inputs copies
    while (!c_works.empty()) {
        auto work = c_works.front();
        auto err = cudaEventQuery(work.ev_c);
        if (err == cudaErrorNotReady)
            break;
        C(err);
        C(cudaEventDestroy(work.ev_c));
        work.ev_c = cudaEvent_t{};
        auto wl = c_workload.load(std::memory_order_relaxed);
        auto fin = c_finished.fetch_add(work.load, std::memory_order_relaxed) + work.load;
        std::cout << std::format("dev#{}: {}/{} = {:.2f}% done, free up {}B device mem ({} entries)\n",
                dev, display(fin), display(wl), 100.0 * fin / wl,
                display(work.len * sizeof(R)), work.len);
        C(cudaFree(work.p));
        work.p = nullptr;
        lock.lock();
        xc_pending--;
        c_works.pop_front();
        lock.unlock();
        std::cout << std::format("dev#{}: {}B ({} entries) device mem freed\n",
                dev, display(work.len * sizeof(R)), work.len);
    }

    // dispatch more
    lock.lock();
    while (!xc_queue.empty()) {
        auto work = xc_queue.front();
        xc_queue.pop_front();
        lock.unlock();
        { // dispatch logic
            std::cout << std::format(
                    "dev#{}: 0b{:08b}<<<{:8}, {:3}>>> = {:<6}*L{:<5}*R{:<5} => {:>9}B\n",
                    dev, work.pos, work.b, work.t,
                    work.len, work.fanoutL, work.fanoutR, display(work.sz * sizeof(R)));
            C(cudaMallocAsync(&work.p, work.len * sizeof(R), c_stream));
            C(cudaMemcpyAsync(work.p, work.ptr, work.len * sizeof(R),
                        cudaMemcpyHostToDevice, c_stream));
            C(cudaEventCreateWithFlags(&work.ev_m, cudaEventDisableTiming));
            C(cudaEventRecord(work.ev_m, c_stream));
            launch(work.b, work.t, c_stream, height,
                    // output ring buffer
                    ring_buffer, n_outs,
                    n_chunks,
                    &counters[0], &counters[1],
                    // input vector
                    work.p, work.len,
                    // constants
                    work.pos,
                    work.d_f0L, work.fanoutL,
                    work.d_f0R, work.fanoutR);
            C(cudaEventCreateWithFlags(&work.ev_c, cudaEventDisableTiming));
            C(cudaEventRecord(work.ev_c, c_stream));
            c_works.emplace_back(work);
        }
        lock.lock();
        xc_pending++;
        cv.notify_all();
    }
    if (!xc_closed || !c_works.empty())
        goto again;

    lock.unlock();
    unsigned long long used;
    { // figure out final output count
        C(cudaMemcpyAsync(&used, n_outs, sizeof(used), cudaMemcpyDeviceToHost, c_stream));
        C(cudaStreamSynchronize(c_stream));
        C(cudaStreamDestroy(c_stream));
        c_stream = cudaStream_t{};
        C(cudaFree(n_outs));
        n_outs = nullptr;
    }
    lock.lock();
    std::cout << std::format("dev#{}: finalizing with xc_used={}\n", dev, used);
    xc_used = used;
    cv.notify_all();
}

void Device::m_entry() {
    pthread_setname_np(pthread_self(), std::format("dev#{}.m", dev).c_str());
    C(cudaSetDevice(dev));

    cuda::atomic_ref n_reader_chunk{ counters[0] };
    cuda::atomic_ref n_writer_chunk{ counters[1] };

    C(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

    /*
    auto verify_mem = [=,this] {
        while (true) {
            auto mem = 4096 * get_avphys_pages();
            if (mem / (2 * CYC_CHUNK * sizeof(RX)) >= m_data.size())
                break;
            std::cout << std::format("mem: only {}B memory availabe, wait\n",
                    display(mem));
            if (last) {
                ::usleep(1000000);
            } else {
                ::usleep(500000);
                return false;
            }
        }
        return true;
    };
    */

    // if (!verify_mem()) return;

    std::unique_lock lock{ mtx };
    cv.wait(lock, [this]{ return xc_ready; });

again:
    cv.wait_for(lock, 50ms, [this]{ return xc_used.has_value(); });
    auto used = std::move(xc_used);
    xc_used.reset();
    lock.unlock();

    unsigned long long nwc;
    { // recycle logic
        nwc = n_writer_chunk.load(cuda::memory_order_acquire);
        while (m_scheduled < nwc) {
            std::cout << std::format("dev#{}: start DtoH chunk #{:0{}}/{} ({}B)\n",
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
            // if (!verify_mem()) return;
        }
    }

    if (used) { // tail recycle logic
        if (*used < nwc * CYC_CHUNK)
            throw std::runtime_error{ "internal error" };
        if (*used >= (nwc + 1u) * CYC_CHUNK)
            throw std::runtime_error{ "internal error" };
        if (m_scheduled != nwc)
            throw std::runtime_error{ "internal error" };
        if (*used > nwc * CYC_CHUNK) {
            auto sz = *used - nwc * CYC_CHUNK;
            std::cout << std::format("dev#{}: start tail DtoH chunk #{:0{}}/{} for {} entries ({}B)\n",
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
    }

    // sorter logic
    while (!m_events.empty()) {
        auto ev = m_events.front();
        auto err = cudaEventQuery(ev);
        if (err == cudaErrorNotReady)
            break;
        C(err);
        C(cudaEventDestroy(ev));
        auto nrc = n_reader_chunk.fetch_add(1, cuda::memory_order_release);
        std::cout << std::format("dev#{}: pushing chunk #{:0{}} ({} entries, {}B) to sorter\n",
                dev, nrc, count_digits(n_chunks),
                CYC_CHUNK, display(CYC_CHUNK * sizeof(RX)));
        sorter.push(m_data.front());
        m_events.pop_front();
        m_data.pop_front();
    }

    lock.lock();
    if (!m_events.empty() || !xc_closed || xc_pending)
        goto again;

    xm_completed = true;
    cv.notify_all();
    lock.unlock();
    C(cudaStreamSynchronize(m_stream));
    C(cudaStreamDestroy(m_stream));
    m_stream = cudaStream_t{};
    C(cudaFree(ring_buffer));
    ring_buffer = nullptr;
    C(cudaFree(counters));
    counters = nullptr;
}

void Device::dispatch(WL cfgs) {
    if (!cfgs.ptr || !cfgs.len)
        return;
    std::unique_lock lock{ mtx };
    cv.wait(lock, [this]{ return xc_ready; });
    auto &work = xc_queue.emplace_back(cfgs, dev, height);
    c_workload += work.b * work.t;
    cv.notify_all();
}

void Device::close() {
    std::unique_lock lock{ mtx };
    xc_closed = true;
    cv.notify_all();
}

void Device::wait() {
    std::unique_lock lock{ mtx };
    cv.wait(lock, [this]{ return xm_completed; });
}

uint64_t Device::get_workload() const {
    return c_workload.load(std::memory_order_relaxed);
}

void Device::print_stats() const {
    cuda::atomic_ref n_reader_chunk{ counters[0] };
    cuda::atomic_ref n_writer_chunk{ counters[1] };

    std::stringstream ss;
    ss << "\33[37mdev" << dev << " [";

    std::unique_lock lock{ mtx };
    if (!xc_ready) {
        ss << "initializing]";
    } else if (xm_completed) {
        ss << "completed]";
    } else {
        auto nrc = n_reader_chunk.load(cuda::memory_order_relaxed);
        auto nwc = n_writer_chunk.load(cuda::memory_order_relaxed);
        for (auto i = 0ull; i < n_chunks; i++) {
            auto c = i < nrc ? i + n_chunks : i;
            if (c < nrc)
                ss << " ";
            else if (c < m_scheduled)
                ss << "\33[35mR";
            else if (c < nwc)
                ss << "\33[90m-";
            else if (c == nwc)
                ss << "\33[36mW";
            else
                ss << " ";
        }
        auto wl = c_workload.load(std::memory_order_relaxed);
        auto fin = c_finished.load(std::memory_order_relaxed);
        ss << "\33[37m]";
        if (xc_pending)
            ss << std::format(" {:.02f}%/{:d}", 100.0 * fin / wl, xc_pending);
        if (!xc_queue.empty())
            ss << std::format("/{:d}", xc_queue.size());
        if (xc_closed)
            ss << " closed";
    }
    lock.unlock();
    ss << "\33[K\33[0m\n";
    std::cerr << ss.str();
}
