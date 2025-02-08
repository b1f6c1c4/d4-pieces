#pragma once

#include <cuda.h>
#include <cstdint>
#include "frow.h"
#include "record.h"

#ifdef BMARK
#include <vector>
#endif

struct KParams;
struct KSizing {
    uint64_t n_cfgs;
    uint32_t f0Lsz;
    uint32_t f0Rsz;

#ifdef BMARK
    [[nodiscard]] std::vector<KParams> optimize() const;
#else
    [[nodiscard]] KParams optimize() const;
#endif
};

struct KParams : KSizing {
    bool reverse;
    uint64_t blocks;
    uint32_t threads;
    unsigned shmem_len;

    [[nodiscard]] double fom() const;
};

struct KParamsFull : KParams {
    unsigned height;

    RX                 *ring_buffer;
    unsigned long long *n_outs;
    unsigned long long n_chunks;
    unsigned long long *n_reader_chunk;
    unsigned long long *n_writer_chunk;

    const R *cfgs;

    uint8_t ea;

    const frow_t *f0L;
    const frow_t *f0R;

    void launch(cudaStream_t stream);
};

void prepare_kernels(); // must be called before KParamsFull::launch
