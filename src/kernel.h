#pragma once

#include <cstdint>
#include "frow.h"
#include "record.h"

#define K_PARAMS \
        /* output ring buffer */ \
        RX                 *ring_buffer, /* __device__ */ \
        unsigned long long *n_outs, /* __device__ */ \
        unsigned long long n_chunks, \
        unsigned long long *n_reader_chunk, /* __managed__, HtoD */ \
        unsigned long long *n_writer_chunk, /* __managed__, DtoH */ \
        /* input vector */ \
        const R *cfgs, const uint64_t n_cfgs, \
        /* constants */ \
        uint8_t ea, \
        const frow_t *f0L, const uint32_t f0Lsz, \
        const frow_t *f0R, const uint32_t f0Rsz

template <unsigned H> __global__ void d_row_search(K_PARAMS);
extern template __global__ void d_row_search<8>(K_PARAMS);
extern template __global__ void d_row_search<7>(K_PARAMS);
extern template __global__ void d_row_search<6>(K_PARAMS);
extern template __global__ void d_row_search<5>(K_PARAMS);
extern template __global__ void d_row_search<4>(K_PARAMS);
extern template __global__ void d_row_search<3>(K_PARAMS);
extern template __global__ void d_row_search<2>(K_PARAMS);
extern template __global__ void d_row_search<1>(K_PARAMS);
