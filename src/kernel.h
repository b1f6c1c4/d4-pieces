#pragma once

#include <cstdint>
#include "frow.h"
#include "record.h"

struct KParams;
struct KSizing {
    uint64_t n_cfgs;
    uint32_t f0Lsz;
    uint32_t f0Rsz;

    [[nodiscard]] KParams optimize() const;
};

struct KParams : KSizing {
    bool reverse;
    uint64_t blocks;
    uint32_t threads;
    unsigned shmem_len;

    [[nodiscard]] double fom() const;
};

#ifdef __CUDA_ARCH__
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
#endif
