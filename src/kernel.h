#pragma once

#include <cuda.h>
#include <cstdint>
#include <string>
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
    [[nodiscard]] KParams optimize(bool debug = false) const;
#endif
    [[nodiscard]] std::string to_string() const;
};

enum class KKind {
    // shmem_len == 0
    Legacy,        // b*t >= n_cfgs * f0Lsz * f0Rsz
    // shmem_len == (12*threads) B
    CoalescedR,    // b*t >= n_cfgs * f0Lsz * ceil(f0Rsz/threads)
    CoalescedL,    // b*t >= n_cfgs * ceil(f0Lsz/threads) * f0Rsz
    // shmem_len == several KiB
    TiledStandard, // b*t >= n_cfgs
    TiledReversed, // b*t >= n_cfgs; not recommended
};

struct KParams : KSizing {
    KKind ty;
    uint64_t blocks;
    uint32_t threads;
    unsigned shmem_len;

#ifdef BMARK
    double fom(bool debug = false) const;
#else
    [[nodiscard]] double fom() const;
#endif
    [[nodiscard]] std::string to_string(bool full) const;
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

#ifdef BMARK
    unsigned long long *perf;
#endif

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

#ifdef BMARK
#define K_PARAMS_PROF , unsigned long long *perf
#else
#define K_PARAMS_PROF
#endif

#define K_PARAMS \
        K_PARAMS_OUT, \
        /* input vector */ \
        const R *cfgs, const uint64_t n_cfgs, \
        /* constants */ \
        uint8_t ea, \
        const frow32_t *f0L, uint32_t f0Lsz, \
        const frow32_t *f0R, uint32_t f0Rsz K_PARAMS_PROF
