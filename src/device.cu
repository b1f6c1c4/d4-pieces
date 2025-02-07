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

Device::Input::Input(WL work, int dev, unsigned height) : WL{ work }, p{} {
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

    std::cout << std::format("dev#{}.c: allocating {} * {}B = {}B ring buffer\n",
            dev, n_chunks, display(CYC_CHUNK * sizeof(RX)),
            display(n_chunks * CYC_CHUNK * sizeof(RX)));
    C(cudaMallocAsync(&ring_buffer, n_chunks * CYC_CHUNK * sizeof(RX), c_stream));

    std::unique_lock lock{ mtx };
    xc_ready = true;
    cv.notify_all();

again:
    cv.wait_for(lock, 50ms, [this]{ return xc_closed || !xc_queue.empty(); });

    // synchronously free up original copies (work.ptr)
    lock.unlock();
    for (auto &work : c_works) {
        if (!work || work.device_accessible())
            continue;
        auto err = cudaEventQuery(work.ev_m);
        if (err == cudaErrorNotReady)
            continue;
        C(err);
        C(cudaEventDestroy(work.ev_m));
        work.ev_m = cudaEvent_t{};
        std::cout << std::format("dev#{}.c: free up {}B host input mem ({} entries)\n",
                dev, display(work.len * sizeof(R)), work.len);
        work.dispose();
    }

    // synchronously free up device copies (work.p)
    while (!c_works.empty()) {
        auto work = c_works.front(); // copy; pop_front() anyway
        auto err = cudaEventQuery(work.ev_c);
        if (err == cudaErrorNotReady)
            break;
        C(err);
        C(cudaEventDestroy(work.ev_c));
        work.ev_c = cudaEvent_t{};
        auto wl = c_workload.load(std::memory_order_relaxed);
        auto fin = c_finished.fetch_add(work.load, std::memory_order_relaxed) + work.load;
        std::cout << std::format("dev#{}.c: {}/{} = {:.2f}% done, free up {}B device mem ({} entries)\n",
                dev, display(fin), display(wl), 100.0 * fin / wl,
                display(work.len * sizeof(R)), work.len);
        if (work.device_accessible())
            work.dispose();
        else {
            C(cudaFree(work.p));
            work.p = nullptr;
        }
        lock.lock();
        xc_pending--;
        c_works.pop_front();
        lock.unlock();
        std::cout << std::format("dev#{}.c: {}B ({} entries) device mem freed\n",
                dev, display(work.len * sizeof(R)), work.len);
    }

    // dispatch more
    lock.lock();
    while (!xc_queue.empty()) {
        auto work = std::move(xc_queue.front());
        xc_queue.pop_front();
        lock.unlock();
        { // dispatch logic
            std::cout << std::format(
                    "dev#{}.c: 0b{:08b}<<<{:8}, {:3}>>> = {:<6}*L{:<5}*R{:<5} => {:>9}B\n",
                    dev, work.pos, work.b, work.t,
                    work.len, work.fanoutL, work.fanoutR, display(work.sz * sizeof(R)));
            if (!work.device_accessible()) {
                C(cudaMallocAsync(&work.p, work.len * sizeof(R), c_stream));
                C(cudaMemcpyAsync(work.p, work.ptr, work.len * sizeof(R),
                            cudaMemcpyHostToDevice, c_stream));
                C(cudaEventCreateWithFlags(&work.ev_m, cudaEventDisableTiming));
                C(cudaEventRecord(work.ev_m, c_stream));
            } else {
                C(cudaMemAdvise(work.ptr, work.len * sizeof(R), cudaMemAdviseSetReadMostly, dev));
                C(cudaMemPrefetchAsync(work.ptr, work.len * sizeof(R), dev, c_stream));
            }
            launch(work.b, work.t, c_stream, height,
                    // output ring buffer
                    ring_buffer, n_outs,
                    n_chunks,
                    &counters[0], &counters[1],
                    // input vector
                    work.device_accessible() ? work.ptr : work.p,
                    work.len,
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
    std::cout << std::format("dev#{}.c: c thread quitting xc_used={}\n", dev, used);
    xc_used = used;
    cv.notify_all();
}

#define MAX_DtoH 8

void Device::m_entry() {
    pthread_setname_np(pthread_self(), std::format("dev#{}.m", dev).c_str());
    C(cudaSetDevice(dev));

    C(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, -1));

    cuda::atomic_ref n_reader_chunk{ counters[0] };
    cuda::atomic_ref n_writer_chunk{ counters[1] };

    std::unique_lock lock{ mtx };
    cv.wait(lock, [this]{ return xc_ready; });

    auto tailed = false;

again:
    cv.wait_for(lock, 50ms, [this]{ return xc_used.has_value(); });
    auto used = std::move(xc_used);
    if (xc_used) {
        tailed = true;
        xc_used.reset();
    }
    lock.unlock();

    std::unique_lock lock_m_works{ mtx_m_works };

again2:
    // sorter logic
    auto local = 0ull;
    while (!m_works.empty()) {
        auto pwork = m_works.front();
        std::atomic_ref ptr{ pwork->ptr };
        if (!ptr.load(std::memory_order_relaxed)) {
            m_works.pop_front();
            local++;
        } else {
            break;
        }
    }
    if (local) {
        n_reader_chunk.fetch_add(local, cuda::memory_order_release);
    }

    // recycle logic
    auto nwc = n_writer_chunk.load(cuda::memory_order_acquire);
    while (m_scheduled < nwc && m_works.size() < MAX_DtoH) {
        m_initiate_transfer(CYC_CHUNK);
        m_scheduled++;
    }
    if (m_works.size() >= MAX_DtoH) {
        std::this_thread::sleep_for(5ms);
        goto again2;
    }

    if (used) { // tail recycle logic
        if (*used < nwc * CYC_CHUNK)
            throw std::runtime_error{ "internal error" };
        if (*used >= (nwc + 1u) * CYC_CHUNK)
            throw std::runtime_error{ "internal error" };
        if (m_scheduled != nwc)
            throw std::runtime_error{ "internal error" };
        if (*used > nwc * CYC_CHUNK) {
            m_initiate_transfer(*used - nwc * CYC_CHUNK);
        }
    }

    lock_m_works.unlock();

    lock.lock();
    if (!m_works.empty() || !tailed)
        goto again;

    xm_completed = true;
    cv.notify_all();
    lock.unlock();
    std::cout << std::format("dev#{}.m: m thread quitting\n", dev);
    C(cudaFree(ring_buffer));
    ring_buffer = nullptr;
    C(cudaFree(counters));
    counters = nullptr;
}

void Device_callback_helper(void *raw) {
    auto data = reinterpret_cast<void **>(raw);
    auto *self = static_cast<Device *>(data[0]);
    auto *pwork = static_cast<Device::Output *>(data[1]);
    self->m_callback(pwork);
    delete [] data;
}

void Device::m_initiate_transfer(uint64_t sz) {
    std::cout << std::format("dev#{}.m: start {}DtoH chunk #{:0{}}/{} ({}B)\n",
            dev, sz == CYC_CHUNK ? "" : "tail ",
            m_scheduled, count_digits(n_chunks),
            n_chunks, display(CYC_CHUNK * sizeof(RX)));
    auto pwork = m_works.emplace_back(new Output{ Rg<RX>::make_cpu(sz) });
    C(cudaMemcpyAsync(pwork->ptr,
                ring_buffer + (m_scheduled % n_chunks) * CYC_CHUNK,
                sz * sizeof(RX), cudaMemcpyDeviceToHost, m_stream));
    auto data = new void *[2]{ this, pwork };
    C(cudaLaunchHostFunc(m_stream, &Device_callback_helper, data));
}

void Device::m_callback(Output *pwork) {
    std::cout << std::format("dev#{}.m: pushing a chunk ({} entries, {}B) to sorter\n",
            dev, pwork->len, display(pwork->len * sizeof(RX)));
    sorter.push(*pwork);
    std::atomic_ref ptr{ pwork->ptr };
    ptr.store(nullptr, std::memory_order_relaxed);
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
        lock.unlock();
        auto nwc = n_writer_chunk.load(cuda::memory_order_relaxed);

        std::unique_lock lock_m_works{ mtx_m_works, std::defer_lock_t{} };
        (void)lock_m_works.try_lock_for(5ms);
        auto nrc = n_reader_chunk.load(cuda::memory_order_relaxed);
        for (auto i = 0ull; i < n_chunks; i++) {
            auto c = i < nrc ? i + n_chunks : i;
            if (c < nrc)
                ss << " ";
            else if (c < m_scheduled) {
                if (!lock_m_works || c - nrc >= m_works.size())
                    ss << "\33[31m?";
                else if (m_works[c - nrc]->ptr)
                    ss << "\33[35mR";
                else
                    ss << "\33[95mR";
            } else if (c < nwc)
                ss << "\33[90m-";
            else if (c == nwc)
                ss << "\33[36mW";
            else
                ss << " ";
        }
        if (lock_m_works)
            lock_m_works.unlock();

        auto wl = c_workload.load(std::memory_order_relaxed);
        auto fin = c_finished.load(std::memory_order_relaxed);

        ss << "\33[37m]";
        lock.lock();
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
