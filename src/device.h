#pragma once

#include <cuda.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include "record.h"
#include "frow.h"
#include "sorter.hpp"

struct WL : Rg<R> {
    unsigned pos;
};

class Device {
    struct Work : WL {
        cudaEvent_t ev;
        uint64_t sz;
        uint32_t b, t;
        const frow_t *d_f0L, *d_f0R;
        uint32_t fanoutL, fanoutR;
        uint64_t load;
        Work(WL work, int dev, unsigned height);
        Work(const Work &other) = default;
        Work(Work &&other) = default;

        // c:cudaMallocAsync c:cudaFree
        // __device__
        R *p;
    };

    int dev;
    unsigned height;
    Sorter &sorter;
    std::jthread c_thread, m_thread;

    // xc_ready xc_closed xc_pending xm_completed
    // false   false    0ull      false
    // true    false    0ull      false
    // (c: dispatching/freeing) (m: recycling)       (sorter)
    // true    false    114ull    false
    //           (Device::close())
    // true    true     113ull    false
    // true    true     0ull      false
    //       (c: exit)       (m: tail recycling)       (sorter)
    // true    true     0ull      true
    //       (c: exit)         (m: cudaFree())   (sorter: finalizing)

    // synchronization, protected by mtx {
    mutable std::mutex mtx;
    // cv.notify_all() happens when:
    // 1. xc_ready = true;
    // 2. xc_closed = true;
    // 3. xm_completed = true;
    // 4. xc_queue.push_back(...);
    // 5. xc_used.emplace();
    std::condition_variable cv;
    bool xc_ready{}, xc_closed{}, xm_completed{};
    uint64_t xc_pending{}; // # kernels
    std::optional<uint64_t> xc_used{}; // final output count
    std::deque<Work> xc_queue{}; // before dispatching
    // }

    std::atomic<uint64_t> c_workload{}; // # threads
    std::atomic<uint64_t> c_finished{}; // # threads

    // c:cudaMallocAsync m:cudaFree
    // __device__
    RX *ring_buffer{};

    // constant, set by c; available after xc_ready == true
    unsigned long long n_chunks{};

    // Device():cudaMallocManaged m:cudaFree
    // __managed__
    // [0] = n_reader_chunk, [1] = n_writer_chunk
    // requires cuda::atomic_ref
    unsigned long long *counters{};

    // c:cudaMallocAsync c:cudaFree
    unsigned long long *n_outs{}; // __device__

    // internals, only c can access
    cudaStream_t c_stream;
    std::deque<Work> c_works{}; // after dispatching
    void c_entry();

    // internals, only m can access
    unsigned long long m_scheduled{};
    std::deque<Rg<RX>> m_data{};
    std::deque<cudaEvent_t> m_events{};
    cudaStream_t m_stream;
    void m_entry();

public:
    Device(int d, unsigned h, Sorter &s);

    // thread-safe, callable from anywhere
    void dispatch(WL cfgs);
    void close(); // no more dispatch
    void wait(); // both c/m completed

    // thread-safe, callable from anywhere
    uint64_t get_workload() const;

    // thread-safe, callable from anywhere
    void print_stats() const;
};
