#pragma once

#include <cstdint>

// g_nme and g_sym must be set
// required before calling Searcher::Searcher
void compute_frow_on_cpu();
void transfer_frow_to_gpu();
void show_gpu_devices();

struct frow_t {
    uint64_t shape;
    union {
        uint8_t nm[4];
        uint32_t nm0123;
    };
};

struct frow_info_t {
    frow_t *data;
    uint32_t sz[6];
};

extern int n_devices; // defined in frow.cu
extern frow_info_t h_frowInfoL[16], h_frowInfoR[16]; // defined in frow.cpp
extern frow_t *d_frowDataL[128][16], *d_frowDataR[128][16]; // defined in frow.cu
