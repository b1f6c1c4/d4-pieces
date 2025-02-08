#pragma once

#include <cuda.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include "record.h"
#include "region.h"
#include "frow.h"
#include "kernel.h"
#include "sorter.hpp"

class Device {
    struct Input : WL { // Rg<X> + pos
        cudaEvent_t ev_m, ev_c;
        unsigned szid;
        KParams kp;
        Input(WL work, int dev, unsigned height);
        Input(const Input &other) = default;
        Input(Input &&other) = default;

        // c:cudaMallocAsync c:cudaFree
        // __device__
        R *p;
    };
    struct Output : Rg<RX> {
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
    std::deque<Input> xc_queue{}; // before dispatching
    // }

    // seconds, estimated by KParams::fom()
    std::atomic<double> c_workload{};
    std::atomic<double> c_finished{};

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
    std::deque<Input> c_works{}; // after dispatching
    void c_entry();

    // internals, only m can access
    unsigned long long m_scheduled{};
    cudaStream_t m_stream;
    mutable std::timed_mutex mtx_m_works; // must not hold mtx
    std::deque<Output *> m_works{};
    void m_entry();
    void m_initiate_transfer(uint64_t sz);
    void m_callback(Output *pwork);
    friend void Device_callback_helper(void *raw);

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
