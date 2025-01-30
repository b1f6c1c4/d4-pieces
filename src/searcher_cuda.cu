#include "searcher_cuda.h"

#include <iostream>

#define MAX_FCFS 16384
#define MAX_SOLUTIONS 1048576

__device__ static tt_t fcf[MAX_FCFS];
__device__ static size_t shps;
__device__ static size_t fcfs;

void fcf_cache(size_t num_shapes) {
    cudaMemcpyToSymbol(fcf, fast_canonical_form, fast_canonical_forms * sizeof(tt_t));
    cudaMemcpyToSymbol(&shps, &num_shapes, sizeof(size_t));
    cudaMemcpyToSymbol(&fcfs, &fast_canonical_forms, sizeof(size_t));
}

template <unsigned D> // 0 ~ 31
__global__
void searcher_impl(uint64_t empty_area, char *output, int *output_sz,
        uint32_t ex0, uint32_t ex1, uint32_t ex2, uint32_t ex3,
        uint32_t ex4, uint32_t ex5, uint32_t ex6, uint32_t ex7) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto covering = empty_area & ~empty_area; // Shape<8>::front_shape();
    auto [shape, nm] = fcf[idx];
    if (!(shape & covering)) [[likely]] return;
    if (shape & ~empty_area) [[likely]] return;
    auto rest = empty_area & ~shape;
    if (!rest) {
        auto pos = atomicAdd(output_sz, 1);
        if (pos >= MAX_SOLUTIONS)
            return; // FIXME: throw
        auto ptr = &output[fcfs * pos];
        ptr[nm] = 1;
#define W(ex) do { \
        ptr[(ex >>  0) & 0xff] = 1; \
        ptr[(ex >>  4) & 0xff] = 1; \
        ptr[(ex >> 12) & 0xff] = 1; \
        ptr[(ex >> 16) & 0xff] = 1; \
} while (false)
        if constexpr (D >  0) W(ex0);
        if constexpr (D >  4) W(ex1);
        if constexpr (D >  8) W(ex2);
        if constexpr (D > 12) W(ex3);
        if constexpr (D > 16) W(ex4);
        if constexpr (D > 20) W(ex5);
        if constexpr (D > 24) W(ex6);
        if constexpr (D > 28) W(ex7);
        return;
    }
    if constexpr (D <= 32) {
        uint32_t nmx = __byte_perm(nm, 0, 0); // nm nm nm nm
        if constexpr (D) {
            if constexpr (D <=  4) if (__vcmpeq4(nm, ex0)) [[unlikely]] return;
            if constexpr (D <=  8) if (__vcmpeq4(nm, ex1)) [[unlikely]] return;
            if constexpr (D <= 12) if (__vcmpeq4(nm, ex2)) [[unlikely]] return;
            if constexpr (D <= 16) if (__vcmpeq4(nm, ex3)) [[unlikely]] return;
            if constexpr (D <= 20) if (__vcmpeq4(nm, ex4)) [[unlikely]] return;
            if constexpr (D <= 24) if (__vcmpeq4(nm, ex5)) [[unlikely]] return;
            if constexpr (D <= 28) if (__vcmpeq4(nm, ex6)) [[unlikely]] return;
            if constexpr (D <= 32) if (__vcmpeq4(nm, ex7)) [[unlikely]] return;
        }
        auto nms = static_cast<uint64_t>(nm) << (D % 4);
        auto nmm = static_cast<uint64_t>(0xff) << (D % 4);
             if constexpr (D <  4) ex0 = ((ex0 & ~nmm) | nms);
        else if constexpr (D <  8) ex1 = ((ex1 & ~nmm) | nms);
        else if constexpr (D < 12) ex2 = ((ex2 & ~nmm) | nms);
        else if constexpr (D < 16) ex3 = ((ex3 & ~nmm) | nms);
        else if constexpr (D < 20) ex4 = ((ex4 & ~nmm) | nms);
        else if constexpr (D < 24) ex5 = ((ex5 & ~nmm) | nms);
        else if constexpr (D < 28) ex6 = ((ex6 & ~nmm) | nms);
        else if constexpr (D < 32) ex7 = ((ex7 & ~nmm) | nms);
        auto fcf_blocks = (fcfs + 512 - 1) / 512;
        auto fcf_threads = 512;
        searcher_impl<D + 1><<<fcf_blocks, fcf_threads, 0, cudaStreamFireAndForget>>>(
                empty_area & ~shape, output, output_sz,
                ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
    }
}

char *searcher_area(size_t num_shapes) {
    char *output;
    cudaMallocManaged(&output, MAX_SOLUTIONS * num_shapes * sizeof(char));
    return output;
}

size_t searcher_step(char *output, uint64_t empty_area) {
    auto fcf_blocks = (fast_canonical_forms + 512 - 1) / 512;
    auto fcf_threads = 512;
    int *output_sz;
    cudaMallocManaged(&output_sz, sizeof(int));
    *output_sz = 0;
    searcher_impl<0><<<fcf_blocks, fcf_threads>>>(empty_area,
        output, output_sz,
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u);
    cudaDeviceSynchronize();
    auto res = *output_sz;
    cudaFree(output_sz);
    return res;
}
