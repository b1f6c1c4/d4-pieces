#include "searcher_cuda.h"

#include <iostream>

__device__ static const tt_t *fcf;

void fcf_cache() {
    cudaMalloc(&fcf, fast_canonical_forms * sizeof(tt_t));
    cudaMemcpy(&fcf, fast_canonical_form, fast_canonical_forms * sizeof(tt_t),
            cudaMemcpyHostToDevice);
}

template <unsigned D> // 0 ~ 31
__global__
void searcher_impl(uint64_t empty_area, uint32_t fcfs,
        uint32_t ex0, uint32_t ex1, uint32_t ex2, uint32_t ex3,
        uint32_t ex4, uint32_t ex5, uint32_t ex6, uint32_t ex7) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto covering = empty_area & ~empty_area; // Shape<8>::front_shape();
    auto [shape, nm] = fcf[idx];
    if (!(shape & covering)) [[likely]] return;
    if (shape & ~empty_area) [[likely]] return;
    auto rest = empty_area & ~shape;
    if (!rest) {
        return; // TODO
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
        searcher_impl<D + 1><<<fcf_blocks, fcf_threads>>>(empty_area & ~shape, fcfs,
            ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
    }
}

void searcher_step(uint64_t empty_area, char *possible_pieces) {
    auto fcf_blocks = (fast_canonical_forms + 512 - 1) / 512;
    auto fcf_threads = 512;
    searcher_impl<0><<<fcf_blocks, fcf_threads>>>(empty_area, fast_canonical_forms,
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u);
}
