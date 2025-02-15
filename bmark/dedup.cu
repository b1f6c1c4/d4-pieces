#include <cstdint>
#include <cuda.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <format>
#include <iostream>
#include <curand.h>
#include "../src/util.cuh"

__device__ unsigned long long r_count, w_count;

__global__
void dedup_n2(
        uint64_t *data,
        unsigned long long n_data) {
    auto idx = threadIdx.x + (unsigned long long)blockIdx.x * blockDim.x;
    if (idx >= n_data)
        return;

    auto datum = data[idx];
    auto limit = __nv_atomic_load_n(&w_count, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
    for (auto i = 0ull; i < limit; i++)
        if (data[i] == datum)
            return;

    auto out = __nv_atomic_fetch_add(&r_count, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    data[out] = datum;
    __nv_atomic_fetch_add(&w_count, 1, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
}

int main(int argc, char *argv[]) {
    auto sz = std::atoll(argv[1]);
    curandGenerator_t gen;
    C(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    C(curandSetPseudoRandomGeneratorSeed(gen, sz));
    uint64_t *ptr;
    C(cudaMalloc(&ptr, sz * sizeof(uint64_t)));
    C(curandGenerateUniformDouble(gen, reinterpret_cast<double *>(ptr), sz));
    dedup_n2<<<sz / 512, 512>>>(ptr, sz);
}
