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

Device::Input::Input(WL work, int dev, unsigned height)
    : WL{ work }, szid{ min(height - 1, 5) }, kp{ KSizing{ len,
        h_frowInfoL[pos >> 0 & 0b1111u].sz[szid],
        h_frowInfoR[pos >> 4 & 0b1111u].sz[szid] }.optimize(true) },
      p{} { }

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
    c_thread = std::thread{ &Device::c_entry, this };
    m_thread = std::thread{ &Device::m_entry, this };
}

void Device::c_entry() {
    pthread_setname_np(pthread_self(), std::format("dev#{}.c", dev).c_str());
    C(cudaSetDevice(dev));

    C(cudaStreamCreateWithFlags(&c_stream, cudaStreamNonBlocking));

    C(cudaMallocAsync(&n_outs, sizeof(unsigned long long), c_stream));
    C(cudaMemsetAsync(n_outs, 0, 2 * sizeof(unsigned long long), c_stream));

    std::cout << std::format("dev#{}.c: allocating {} * {}B = {}B ring buffer\n",
            dev, n_chunks, display(CYC_CHUNK * sizeof(RX)),
            display(n_chunks * CYC_CHUNK * sizeof(RX)));
    C(cudaMallocAsync(&ring_buffer, n_chunks * CYC_CHUNK * sizeof(RX), c_stream));

    std::unique_lock lock{ mtx };
    boost::upgrade_lock lock_c_works{ mtx_c };

    xc_ready = true;
    cv.notify_all();

again:
    cv.wait_for(lock, 50ms, [this]{ return xc_closed || !xc_queue.empty(); });

    // synchronously free up original copies (work.ptr)
    lock.unlock();

    for (auto &work : c_works) {
        if (work.ev_m == cudaEvent_t{})
            continue;
        auto err = cudaEventQuery(work.ev_m);
        if (err == cudaErrorNotReady)
            continue;
        C(err);
        {
            // since indefinite long time might have passed before actual
            // kernel start and ev_m check, this is only a rough estimate
            boost::upgrade_to_unique_lock xlock_c_work{ lock_c_works };
            work.est_start_time = std::chrono::steady_clock::now();
        }
        C(cudaEventDestroy(work.ev_m));
        work.ev_m = cudaEvent_t{};
        if (!work.device_accessible()) {
            std::cout << std::format("dev#{}.c: free up {}B host input mem ({} entries)\n",
                    dev, display(work.len * sizeof(R)), work.len);
            work.dispose();
        }
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
        std::cout << std::format("dev#{}.c: free up {}B device mem ({} entries)\n",
                dev, display(work.len * sizeof(R)), work.len);
        if (work.device_accessible())
            work.dispose();
        else {
            C(cudaFree(work.p));
            work.p = nullptr;
        }
        {
            boost::upgrade_to_unique_lock xlock_c_work{ lock_c_works };
            c_works.pop_front();
            c_fom_done += work.kp.fom();
            c_actual_done += work.elapsed();
        }
        std::cout << std::format("dev#{}.c: {}B ({} entries) device mem freed\n",
                dev, display(work.len * sizeof(R)), work.len);
    }

    // launch kernels
    lock.lock();
    if (!xc_queue.empty()) { // make sure not to issue too many
        auto work = std::move(xc_queue.front());
        xc_queue.pop_front();
        lock.unlock();
        { // dispatch logic
            KParamsFull kpf{ work.kp, height,
                ring_buffer, n_outs, n_chunks,
                &counters[0], &counters[1],
                work.ptr, (uint8_t)work.pos,
                d_frowDataL[dev][work.pos >> 0 & 0xfu],
                d_frowDataR[dev][work.pos >> 4 & 0xfu] };
            std::cout << std::format(
                    "dev#{}.c: {:08b}<<<{:8},{:4},{:5}B>>>[{}] = {:<6}*L{:<5}*R{:<5} ~ {}\n",
                    dev, work.pos,
                    kpf.blocks, kpf.threads, kpf.shmem_len * sizeof(frow32_t),
                    !kpf.shmem_len ? "-" : kpf.reverse ? "L" : "R",
                    kpf.n_cfgs, kpf.f0Lsz, kpf.f0Rsz, display(kpf.fom()));
            if (!work.device_accessible()) {
                C(cudaMallocAsync(&work.p, work.len * sizeof(R), c_stream));
                C(cudaMemcpyAsync(work.p, work.ptr, work.len * sizeof(R),
                            cudaMemcpyHostToDevice, c_stream));
                kpf.cfgs = work.p;
            } else {
                C(cudaMemAdvise(work.ptr, work.len * sizeof(R), cudaMemAdviseSetReadMostly, dev));
                C(cudaMemPrefetchAsync(work.ptr, work.len * sizeof(R), dev, c_stream));
            }
            C(cudaEventCreateWithFlags(&work.ev_m, cudaEventDisableTiming));
            C(cudaEventRecord(work.ev_m, c_stream));
            kpf.launch(c_stream);
            C(cudaEventCreateWithFlags(&work.ev_c, cudaEventDisableTiming));
            C(cudaEventRecord(work.ev_c, c_stream));

            boost::upgrade_to_unique_lock xlock_c_work{ lock_c_works };
            c_sum_fom += work.kp.fom();
            c_works.emplace_back(work);
        }
        lock.lock();
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
    boost::upgrade_lock lock_m_works{ mtx_m };

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

again2:
    // sorter logic
    auto local = 0ull;
    while (!m_works.empty()) {
        auto pwork = m_works.front();
        std::atomic_ref ptr{ pwork->ptr };
        if (!ptr.load(std::memory_order_relaxed)) {
            boost::upgrade_to_unique_lock xlock_m_works{ lock_m_works };
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
        m_initiate_transfer(CYC_CHUNK, lock_m_works);
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
            m_initiate_transfer(*used - nwc * CYC_CHUNK, lock_m_works);
        }
    }

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

void Device::m_initiate_transfer(uint64_t sz, boost::upgrade_lock<boost::upgrade_mutex> &lock) {
    std::cout << std::format("dev#{}.m: start {}DtoH chunk #{:0{}}/{} {} ({}B)\n",
            dev, sz == CYC_CHUNK ? "" : "tail ",
            m_scheduled, count_digits(n_chunks),
            n_chunks, sz, display(sz * sizeof(RX)));
    Output *pwork;
    {
        boost::upgrade_to_unique_lock xlock_m_works{ lock };
        pwork = m_works.emplace_back(new Output{ Rg<RX>::make_cpu(sz) });
        m_scheduled++;
    }
    C(cudaMemcpyAsync(pwork->ptr,
                ring_buffer + ((m_scheduled - 1) % n_chunks) * CYC_CHUNK,
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
    c_fom_queued.fetch_add(work.kp.fom(), std::memory_order_relaxed);
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

double Device::get_etc() const {
    boost::shared_lock lock_c_works{ mtx_c };
    if (c_works.empty())
        return 0;

    auto &work = c_works.front();
    auto el = work.elapsed();
    auto q = c_fom_queued.load(std::memory_order_relaxed);
    return q + c_sum_fom - std::min(work.kp.fom(), el);
}

unsigned Device::print_stats() const {
    cuda::atomic_ref n_reader_chunk{ counters[0] };
    cuda::atomic_ref n_writer_chunk{ counters[1] };

    auto lines = 1u;
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

        boost::shared_lock lock_m_works{ mtx_m };
        auto nrc = n_reader_chunk.load(cuda::memory_order_relaxed);
        for (auto i = 0ull; i < n_chunks; i++) {
            auto c = i < nrc ? i + n_chunks : i;
            if (c < nrc)
                ss << " ";
            else if (c < m_scheduled) {
                if (m_works[c - nrc]->ptr)
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
        lock_m_works.unlock();

        ss << "\33[37m] ";

        boost::shared_lock lock_c_works{ mtx_c };
        ss << std::format("[E{:7} A{:7}]",
                display(c_fom_done), display(c_actual_done));
        ss << std::format(" ETC{:7}", display(get_etc()));

        lock.lock();
        if (!xc_queue.empty())
            ss << std::format(" Q{:d}", xc_queue.size());
        if (xc_closed)
            ss << " closed";

        if (!c_works.empty()) {
            auto &work = c_works.front();
            lines++;
            ss << std::format("\33[K\n\33[37mdev#{} {:08b}{}/{:7}]",
                        dev, work.pos, work.kp.to_string(true), display(work.elapsed()));
            if (c_works.size() >= 1)
                ss << " + W" << c_works.size() - 1;
        }
    }
    lock.unlock();
    ss << "\33[K\33[0m\n";
    std::cerr << ss.str();
    return lines;
}

Device::~Device() {
    // it is necessary to explicitly join c/m_thread now, because implicitly
    // defined destructors will destroy data members in the reverse order of
    // declaration and clash the execution of tail
    c_thread.join();
    m_thread.join();
}
