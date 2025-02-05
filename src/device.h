#pragma once

#include <cuda.h>
#include <deque>
#include "record.h"
#include "sorter.hpp"

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

    explicit Device(int d);
    ~Device();
    [[nodiscard]] bool c_completed() const;
    [[nodiscard]] bool m_completed() const;
    void dispatch(unsigned pos, unsigned height, Rg<R> cfgs);
    void recycle(bool last);
    void collect(Sorter &sorter, unsigned height);
};
