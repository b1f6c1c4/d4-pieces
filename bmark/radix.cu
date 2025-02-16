#include <chrono>
#include <cub/device/device_radix_sort.cuh>
#include <curand.h>
#include "../src/util.cuh"

struct AoS {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint8_t e;
};

template <auto AoS::*MPtr>
struct Decomposer {
    __host__ __device__
    auto operator()(AoS &obj) const ->
        cuda::std::tuple<decltype(obj.*MPtr) &> {
        return { obj.*MPtr };
    }
};

int main(int argc, char *argv[]) {
    auto sz = std::atoll(argv[1]);
    curandGenerator_t gen;
    C(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    C(curandSetPseudoRandomGeneratorSeed(gen, sz));
    AoS *d_in, *d_out;
    std::print("Allocating 2 * {}B buffer\n", display(sz * sizeof(AoS)));
    C(cudaMalloc(&d_in, sz * sizeof(AoS)));
    C(cudaMalloc(&d_out, sz * sizeof(AoS)));
    std::print("Initializing {} * {} = {}B elements\n", display(sz),
            sizeof(AoS), display(sz * sizeof(AoS)));
    C(curandGenerate(gen, reinterpret_cast<uint32_t *>(d_in), sz * sizeof(AoS) / sizeof(uint32_t)));

    uint8_t *d_temp_storage{};
    size_t temp_storage_bytes{};

    std::print("Prepare sorting {} * {} = {}B elements\n", display(sz),
            sizeof(AoS), display(sz * sizeof(AoS)));
    C(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
            d_in, d_out, sz, Decomposer<&AoS::a>{}));

    std::print("Allocating {}B temp storage\n", display(temp_storage_bytes));
    C(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    std::print("Start sorting & dedup {} * {} = {}B elements\n", display(sz),
            sizeof(AoS), display(sz * sizeof(AoS)));
    auto t1 = std::chrono::steady_clock::now();
    C(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
            d_in, d_out, sz, Decomposer<&AoS::a>{}));
    C(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
            d_out, d_in, sz, Decomposer<&AoS::b>{}));
    C(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
            d_in, d_out, sz, Decomposer<&AoS::c>{}));
    C(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
            d_out, d_in, sz, Decomposer<&AoS::d>{}));
    C(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
            d_in, d_out, sz, Decomposer<&AoS::e>{}));
    auto t2 = std::chrono::steady_clock::now();

    std::print("Copying {}B\n", display(sz * sizeof(AoS)));
    C(cudaFree(d_temp_storage));
    C(cudaFree(d_in));
    auto h_out = new AoS[sz];
    C(cudaMemcpy(h_out, d_out, sz * sizeof(AoS), cudaMemcpyDeviceToHost));
    std::print("verifying\n");
    for (auto i = 1ull; i < sz; i++) {
        const auto &o = h_out[i - 1];
        const auto &n = h_out[i];
#define CHK(f) \
        if (o.f > n.f) THROW("wrong &AoS::{} @{}", #f, i); \
        if (o.f < n.f) continue;
        CHK(e)
        CHK(d)
        CHK(c)
        CHK(b)
        CHK(a)
    }

    auto s = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::print("time = {}, rate = {}/s\n", display(s), display((uint64_t)(sz / s)));
}
