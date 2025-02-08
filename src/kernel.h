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

    const frow32_t *f0L;
    const frow32_t *f0R;

    void launch(cudaStream_t stream);
};

void prepare_kernels(); // must be called before KParamsFull::launch

#define K_PARAMS_OUT \
        /* output ring buffer */ \
        RX                 *ring_buffer, /* __device__ */ \
        unsigned long long *n_outs, /* __device__ */ \
        unsigned long long n_chunks, \
        unsigned long long *n_reader_chunk, /* __managed__, HtoD */ \
        unsigned long long *n_writer_chunk /* __managed__, DtoH */ \

#define K_PARAMS \
        K_PARAMS_OUT, \
        /* input vector */ \
        const R *cfgs, const uint64_t n_cfgs, \
        /* constants */ \
        uint8_t ea, \
        const frow32_t *f0L, const uint32_t f0Lsz, \
        const frow32_t *f0R, const uint32_t f0Rsz
