#pragma once

#include <cuda.h>
#include <atomic>
#include <chrono>
#include <boost/thread/shared_mutex.hpp>
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
        R *p; // __device__
    };
    struct Output : Rg<RX> {
    };

    int dev;
    unsigned height;
    Sorter &sorter;
    std::thread c_thread, m_thread;

    // xc_ready xc_closed c_works xm_completed
    // false      false    empty      false
    // true       false    empty      false
    // (c: dispatching/freeing) (m: recycling)       (sorter)
    // true       false   !empty      false
    //           (Device::close())
    // true       true    !empty      false
    // true       true     empty      false
    //       (c: exit)       (m: tail recycling)       (sorter)
    // true       true     empty      true
    //       (c: exit)         (m: cudaFree())   (sorter: finalizing)

    mutable std::mutex mtx;
    // protected by mtx {
    // notifier waiter     condition(s)
    // c_entry  m/dispatch 1. xc_ready = true;
    // close    c_entry    2. xc_closed = true;
    // m_entry  wait       3. xm_completed = true;
    // dispatch c_entry    4. xc_queue.push_back(...);
    // c_entry  m_entry    5. xc_used.emplace();
    std::condition_variable cv;
    bool xc_ready{}, xc_closed{}, xm_completed{};
    std::optional<uint64_t> xc_used{}; // final output count
    std::deque<Input> xc_queue{}; // before dispatching
    // }

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

    mutable boost::shared_mutex mtx_c;
    cudaStream_t c_stream;
    // protected by mtx_c {
    std::deque<Input> c_works{}; // after dispatching
    double c_sum_fom{}; // in c_works
    double c_fom_done{}, c_actual_done{};
    // } (all r/w must be very short; only c_thread can hold u)
    std::atomic<double> c_fom_queued{};
    void c_entry();

    mutable boost::shared_mutex mtx_m; // must not hold mtx
    cudaStream_t m_stream;
    // protected by mtx_m {
    unsigned long long m_scheduled{};
    std::deque<Output *> m_works{};
    // } (all r/w must be very short; only m_thread can hold u)
    void m_entry();
    void m_initiate_transfer(uint64_t sz, boost::upgrade_lock<boost::upgrade_mutex> &lock);
    void m_callback(Output *pwork);
    friend void Device_callback_helper(void *raw);

public:
    Device(int d, unsigned h, Sorter &s);
    ~Device();

    // thread-safe, callable from anywhere
    void dispatch(WL cfgs);
    void close(); // no more dispatch
    void wait(); // both c/m completed

    // thread-safe, callable from anywhere
    double get_etc() const;

    // thread-safe, callable from anywhere
    unsigned print_stats() const;
};
